"""Local project memory: atomic SQLite snapshots of source graph and derived summaries.

No pickle, API calls, model downloads or graph reconstruction through an LLM.
The embedding space is part of the storage contract; different projects use namespaces.
"""
from __future__ import annotations

from contextlib import closing
from collections import deque
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np


class MemoryConflict(RuntimeError):
    """Another writer advanced this project's memory; reload before retrying."""


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Memory metadata is not JSON serializable: {type(value).__name__}")


class MemoryStore:
    def __init__(self, path: str | Path, project_id: str):
        if not project_id.strip():
            raise ValueError("project_id must not be empty")
        self.path = Path(path)
        self.project_id = project_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("""CREATE TABLE IF NOT EXISTS project_memory (
                project_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                sha256 TEXT NOT NULL, payload BLOB NOT NULL)""")

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA synchronous=FULL")
        return connection

    def load(self):
        with closing(self.connect()) as connection:
            row = connection.execute("SELECT revision, sha256, payload FROM project_memory WHERE project_id=?",
                                     (self.project_id,)).fetchone()
        if row is None:
            return 0, None
        raw = zlib.decompress(row[2])
        if hashlib.sha256(raw).hexdigest() != row[1]:
            raise ValueError("Long-term memory checksum mismatch")
        return row[0], json.loads(raw)

    def save(self, payload: dict, expected_revision: int):
        raw = json.dumps(payload, ensure_ascii=False, allow_nan=False, default=_json_value).encode("utf-8")
        checksum = hashlib.sha256(raw).hexdigest()
        with closing(self.connect()) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute("SELECT revision FROM project_memory WHERE project_id=?",
                                     (self.project_id,)).fetchone()
            revision = row[0] if row else 0
            if revision != expected_revision:
                raise MemoryConflict(f"Memory revision changed: expected {expected_revision}, found {revision}")
            revision += 1
            connection.execute("""INSERT INTO project_memory VALUES(?,?,?,?)
                ON CONFLICT(project_id) DO UPDATE SET revision=excluded.revision,
                sha256=excluded.sha256, payload=excluded.payload""",
                (self.project_id, revision, checksum, zlib.compress(raw)))
        return revision


def graph_snapshot(engine) -> dict:
    # Normalize JSON before returning so a saved snapshot never shares mutable metadata.
    payload = {
        "format": 1,
        "embedding": {"dimension": engine.dim, "model": engine.model_name,
                      "space_id": engine._memory_space_id},
        "graph_config": {"alpha": engine.alpha, "connect": engine.connect_thresh,
                         "label": engine.label_thresh, "top_k": engine.top_k,
                         "degree_threshold": engine.degree_threshold, "max_cluster_size": engine.max_cluster_size},
        "vertices": [vertex.attributes() for vertex in engine.graph.vs],
        "edges": [{"source": engine.graph.vs[edge.source]["name"],
                   "target": engine.graph.vs[edge.target]["name"],
                   "attributes": edge.attributes()} for edge in engine.graph.es],
        "tombstone_ids": sorted(engine.tombstone_ids),
        "dirty_summary_ids": sorted(engine.dirty_summary_ids),
        "pending_cluster_seeds": sorted(engine.pending_cluster_seeds),
        "fallback_id_counter": engine._fallback_id_counter,
    }
    return json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False, default=_json_value))


def rebuild_index(engine):
    import faiss
    index = faiss.IndexIDMap(faiss.IndexHNSWFlat(engine.dim, 32, faiss.METRIC_INNER_PRODUCT))
    all_ids = [int(v["name"]) for v in engine.graph.vs]
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Duplicate node IDs in long-term memory")
    active = [v for v in engine.graph.vs if v["metadata"].get("status") != "tombstone"]
    if active:
        vectors = np.array([v["dense"] for v in active], dtype=np.float32)
        if vectors.shape != (len(active), engine.dim) or not np.isfinite(vectors).all():
            raise ValueError("Invalid persisted memory vectors")
        ids = np.array([int(v["name"]) for v in active], dtype=np.int64)
        index.add_with_ids(vectors, ids)
    engine.index = index


def restore_graph(engine, payload: dict):
    import igraph as ig
    expected = {"dimension": engine.dim, "model": engine.model_name, "space_id": engine._memory_space_id}
    if payload["format"] != 1 or payload["embedding"] != expected:
        raise ValueError("Long-term memory format or embedding space does not match configuration")
    configuration = {"alpha": engine.alpha, "connect": engine.connect_thresh,
                     "label": engine.label_thresh, "top_k": engine.top_k,
                     "degree_threshold": engine.degree_threshold, "max_cluster_size": engine.max_cluster_size}
    if payload["graph_config"] != configuration:
        raise ValueError("Long-term memory graph configuration does not match")
    graph = ig.Graph(directed=False)
    vertices = payload["vertices"]
    graph.add_vertices(len(vertices))
    for key in {key for vertex in vertices for key in vertex}:
        graph.vs[key] = [vertex.get(key) for vertex in vertices]
    if vertices:
        graph.vs["dense"] = [np.asarray(v["dense"], dtype=np.float32) for v in vertices]
    names = {vertex["name"]: vertex.index for vertex in graph.vs}
    for edge in payload["edges"]:
        graph.add_edge(names[edge["source"]], names[edge["target"]], **edge["attributes"])
    # Build a replacement off to the side; validation failures leave the current graph intact.
    from types import SimpleNamespace
    replacement = SimpleNamespace(graph=graph, dim=engine.dim)
    rebuild_index(replacement)
    tombstones = set(payload["tombstone_ids"])
    dirty = set(payload["dirty_summary_ids"])
    seeds = set(payload["pending_cluster_seeds"])
    counter = payload["fallback_id_counter"]
    engine.graph, engine.index = graph, replacement.index
    engine.tombstone_ids, engine.dirty_summary_ids = tombstones, dirty
    engine.pending_cluster_seeds, engine._fallback_id_counter = seeds, counter


def source_evidence_hits(engine, hits: list[dict], limit: int, query_vector=None) -> list[dict]:
    """PageIndex-style summaries route to evidence; their wording is never a source."""
    if engine._memory_store is None and not any(h["metadata"].get("retrieval_role") == "navigation" for h in hits):
        return hits[:limit]
    result, seen = [], set()
    for hit in hits:
        pending = deque([int(hit["id"])])
        visited = set()
        while pending:
            node_id = pending.popleft()
            if node_id in visited:
                continue
            visited.add(node_id)
            vertex = engine.graph.vs.find(name=str(node_id))
            metadata = vertex["metadata"]
            if metadata.get("status") == "tombstone":
                continue
            if vertex["type"] == "Summary" or metadata.get("retrieval_role") == "navigation":
                children = metadata.get("source_node_ids")
                if children is None:
                    children = [int(v["name"]) for v in engine.graph.vs
                                if v["metadata"].get("parent_id") == node_id]
                pending.extend(children)
            elif node_id not in seen:
                seen.add(node_id)
                score = (float(np.dot(np.asarray(vertex["dense"]), np.asarray(query_vector).reshape(-1)))
                         if query_vector is not None else hit.get("score", 0.0))
                result.append({"id": node_id, "content": vertex["content"], "metadata": metadata, "score": score})
    return sorted(result, key=lambda h: (-h["score"], h["id"]))[:limit]
