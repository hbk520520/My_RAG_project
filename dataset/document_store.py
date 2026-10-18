"""Read-only document views over existing graph snapshots; no model or index build."""
from __future__ import annotations

from copy import deepcopy

from .long_term_memory import MemoryStore


class DocumentReader:
    def __init__(self, vertices: list[dict], *, revision: int = 0):
        self.revision = revision
        self.vertices = [{"name": v["name"], "type": v["type"], "content": v["content"],
                          "metadata": deepcopy(v["metadata"])} for v in vertices]

    @classmethod
    def from_file(cls, path, project_id: str):
        revision, snapshot = MemoryStore(path, project_id, read_only=True).load()
        return cls(snapshot["vertices"] if snapshot else [], revision=revision)

    @classmethod
    def from_engine(cls, engine):
        return cls([v.attributes() for v in engine.graph.vs], revision=engine._memory_revision)

    def list_documents(self) -> list[dict]:
        ids = sorted({v["metadata"]["document_id"] for v in self.vertices
                      if v["metadata"].get("document_id") and v["metadata"].get("status") != "tombstone"})
        return [self._document(doc_id)[0] for doc_id in ids]

    def _document(self, document_id: str, version: str | None = None):
        nodes = [v for v in self.vertices if v["metadata"].get("document_id") == document_id]
        active = [v for v in nodes if v["metadata"].get("status") != "tombstone"]
        if not active:
            raise KeyError(f"Document not found: {document_id}")
        versions = {v["metadata"].get("document_fingerprint") for v in nodes}
        if len(versions) != 1:
            raise ValueError("Document has conflicting versions")
        actual_version = versions.pop()
        if version is not None and version != actual_version:
            raise ValueError("Document version does not match the requested version")
        meta = active[0]["metadata"]
        page_count = max(v["metadata"].get("document_page_count", v["metadata"].get("page_end", 0)) or 0
                         for v in nodes)
        return {"document_id": document_id, "version": actual_version, "source_file": meta.get("source_file"),
                "page_count": page_count, "toc_source": meta.get("toc_source", "imported")}, active

    def read_structure(self, document_id: str, *, version: str | None = None) -> dict:
        document, vertices = self._document(document_id, version)
        sections = {int(v["name"]): v for v in vertices if v["type"] == "Summary"
                    and v["metadata"].get("pageindex_node_id") is not None}
        children = {node_id: [] for node_id in sections}
        roots = []
        for node_id, vertex in sections.items():
            parent = vertex["metadata"].get("parent_id")
            if parent is None:
                roots.append(node_id)
            elif parent in sections:
                children[parent].append(node_id)
            else:
                raise ValueError("Document section references a missing parent")
        visited, visiting = set(), set()
        def build(node_id):
            if node_id in visiting:
                raise ValueError("Cycle in document hierarchy")
            visiting.add(node_id)
            vertex = sections[node_id]
            meta = vertex["metadata"]
            result = {"node_id": meta["pageindex_node_id"], "title": meta["title"],
                      "start_index": meta["page_start"], "end_index": meta["page_end"],
                      "summary": meta.get("summary", vertex["content"]),
                      "key_items": meta.get("key_items", []), "toc_source": meta.get("toc_source", document["toc_source"]),
                      "nodes": [build(child) for child in sorted(children[node_id], key=order)]}
            visiting.remove(node_id)
            visited.add(node_id)
            return result
        def order(node_id):
            return sections[node_id]["metadata"]["page_start"], sections[node_id]["metadata"]["pageindex_node_id"]
        structure = [build(root) for root in sorted(roots, key=order)]
        if len(visited) != len(sections):
            raise ValueError("Document hierarchy has unreachable or cyclic sections")
        return {**document, "structure": structure}

    def read_pages(self, document_id: str, pages: list[int], *, version: str | None = None) -> list[dict]:
        document, vertices = self._document(document_id, version)
        if not pages or any(type(page) is not int or not 1 <= page <= document["page_count"] for page in pages):
            raise ValueError("Require valid 1-based physical PDF pages")
        by_page = {}
        for vertex in vertices:
            meta = vertex["metadata"]
            if vertex["type"] == "Raw" and meta.get("page_start") == meta.get("page_end"):
                page = meta.get("page_start")
                if page in by_page:
                    raise ValueError("Document has duplicate source pages")
                by_page[page] = vertex
        result = []
        for page in dict.fromkeys(pages):
            if page not in by_page:
                raise ValueError(f"Source page {page} is deleted or unavailable")
            vertex = by_page[page]
            result.append({"id": int(vertex["name"]), "content": vertex["content"],
                           "metadata": deepcopy(vertex["metadata"])})
        return result
