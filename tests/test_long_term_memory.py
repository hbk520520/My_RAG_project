"""Persistent project memory and PageIndex-format routing; entirely offline."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from dataset.graph import LegalDenseGraphBuilder, StubEncoder
from dataset.long_term_memory import MemoryConflict, MemoryStore, source_evidence_hits
from dataset.page_index_memory import import_pdf, import_tree


def engine(path=None, project="test", space="stub-v1", dim=8):
    graph = LegalDenseGraphBuilder(embedding_dim=dim, encoder=StubEncoder(dim))
    if path is not None:
        graph.enable_memory(path, project, space)
    return graph


def add(graph, node_id, text="original evidence", **kwargs):
    graph.add_node(node_id, text, "Raw", kwargs)


def tree():
    return {"structure": [{"node_id": "root", "title": "Employment", "start_index": 1,
        "end_index": 2, "summary": "A generated navigation summary, not a source", "nodes": [
            {"node_id": "contract", "title": "Contract", "start_index": 1,
             "end_index": 1, "summary": "Contract navigation"}]}]}


def test_roundtrip_preserves_source_summaries_vectors_edges_and_dirty_flags(tmp_path):
    path = tmp_path / "memory.sqlite"
    first = engine(path)
    add(first, 1, "source evidence", source_file="law.pdf", page_start=3, record_id="law:p3")
    summary = first.generate_summary_node(["source evidence"], lambda texts: "summary", child_ids=[1])
    first.graph.add_edge(0, 1, weight=0.8)
    first.tombstone_node(1)
    second = engine(path)
    assert second.graph.vcount() == 2 and second.graph.ecount() == 1
    assert second.graph.vs.find(name="1")["metadata"]["page_start"] == 3
    assert second.graph.vs.find(name=str(summary))["content"] == "summary"
    assert second.graph.vs.find(name=str(summary))["metadata"]["retrieval_role"] == "navigation"
    assert second.tombstone_ids == {1} and summary in second.dirty_summary_ids
    assert np.array_equal(first.graph.vs["dense"][0], second.graph.vs["dense"][0])


def test_restore_does_not_call_encoder(tmp_path):
    path = tmp_path / "memory.sqlite"
    first = engine(path)
    add(first, 1)
    class Forbidden:
        def encode(self, *args, **kwargs):
            pytest.fail("Loading memory must not re-encode or download models")
    second = LegalDenseGraphBuilder(embedding_dim=8, encoder=Forbidden())
    second.enable_memory(path, "test", "stub-v1")
    assert second.index.ntotal == 1


def test_memory_survives_another_process(tmp_path):
    path = tmp_path / "memory.sqlite"
    first = engine(path)
    add(first, 1, "cross-process evidence")
    code = """from dataset.graph import LegalDenseGraphBuilder
g=LegalDenseGraphBuilder(embedding_dim=8)
g.enable_memory(__import__('sys').argv[1], 'test', 'stub-v1')
print(g.graph.vs[0]['content'])
"""
    result = subprocess.run([sys.executable, "-c", code, str(path)], cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "cross-process evidence"


def test_project_namespaces_do_not_share_evidence(tmp_path):
    path = tmp_path / "memory.sqlite"
    a, b = engine(path, "a"), engine(path, "b")
    add(a, 1, "project a")
    add(b, 1, "project b")
    assert engine(path, "a").graph.vs[0]["content"] == "project a"
    assert engine(path, "b").graph.vs[0]["content"] == "project b"


def test_reader_refresh_sees_newly_committed_evidence(tmp_path):
    path = tmp_path / "memory.sqlite"
    writer = engine(path)
    add(writer, 1)
    reader = engine(path)
    add(writer, 2, "new evidence")
    assert reader.graph.vcount() == 1
    reader.refresh_memory()
    assert reader.graph.vcount() == 2


def test_stale_writer_never_overwrites_a_newer_snapshot(tmp_path):
    path = tmp_path / "memory.sqlite"
    writer = engine(path)
    add(writer, 1)
    stale = engine(path)
    add(writer, 2, "committed")
    with pytest.raises(MemoryConflict):
        add(stale, 3, "not committed")
    with pytest.raises(RuntimeError, match="Unsaved"):
        stale.refresh_memory()
    stale.refresh_memory(discard_unsaved=True)
    assert stale.graph.vs["name"] == ["1", "2"]
    assert "not committed" not in stale.graph.vs["content"]


def test_failed_first_save_can_be_explicitly_discarded(tmp_path, monkeypatch):
    graph = engine(tmp_path / "memory.sqlite")
    def fail(*args):
        raise OSError("disk unavailable")
    monkeypatch.setattr(graph._memory_store, "save", fail)
    with pytest.raises(OSError, match="disk unavailable"):
        add(graph, 1)
    with pytest.raises(RuntimeError, match="Unsaved"):
        graph.refresh_memory()
    graph.refresh_memory(discard_unsaved=True)
    assert graph.graph.vcount() == 0 and graph.index.ntotal == 0
    assert graph._memory_pending is False


def test_embedding_space_and_graph_configuration_must_match(tmp_path):
    path = tmp_path / "memory.sqlite"
    add(engine(path), 1)
    with pytest.raises(ValueError, match="embedding"):
        engine(path, space="different")
    with pytest.raises(ValueError, match="embedding"):
        engine(path, dim=16)
    changed = LegalDenseGraphBuilder(embedding_dim=8, connect_threshold=0.4)
    with pytest.raises(ValueError, match="configuration"):
        changed.enable_memory(path, "test", "stub-v1")


def test_checksum_corruption_fails_loudly(tmp_path):
    path = tmp_path / "memory.sqlite"
    add(engine(path), 1)
    import sqlite3
    from contextlib import closing
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("UPDATE project_memory SET sha256='wrong'")
    with pytest.raises(ValueError, match="checksum"):
        engine(path)


def test_initial_batch_and_duplicate_ids_cannot_replace_loaded_memory(tmp_path):
    graph = engine(tmp_path / "memory.sqlite")
    add(graph, 1)
    with pytest.raises(ValueError, match="already exists"):
        add(graph, 1)
    with pytest.raises(ValueError, match="overwrite"):
        graph.build_initial_graph_batch([{"id": 1, "type": "Raw", "content": "replacement"}])
    with pytest.raises(ValueError, match="already exists"):
        graph.generate_summary_node(["text"], lambda texts: "summary", summary_id=1)


def test_duplicate_initial_batch_ids_fail_before_publishing(tmp_path):
    graph = engine(tmp_path / "memory.sqlite")
    with pytest.raises(ValueError, match="Duplicate"):
        graph.build_initial_graph_batch([{"id": 1, "type": "Raw", "content": "one"},
                                         {"id": 1, "type": "Raw", "content": "two"}])
    assert graph.graph.vcount() == 0 and graph._memory_revision == 0


def test_pageindex_import_is_atomic_and_idempotent(tmp_path):
    path = tmp_path / "memory.sqlite"
    graph = engine(path)
    result = import_tree(graph, ["original page one", "original page two"], tree(),
                         document_id="law-v1", source_file="law.pdf")
    revision = graph._memory_revision
    assert graph.graph.vcount() == 4
    assert revision == 1  # not one commit per imported page/section
    again = import_tree(graph, ["original page one", "original page two"], tree(),
                        document_id="law-v1", source_file="law.pdf")
    assert again["reused"] and graph._memory_revision == revision
    assert len(result["nodes"]) == 4 and engine(path).graph.vcount() == 4


def test_navigation_returns_source_pages_never_summary_wording(tmp_path):
    graph = engine(tmp_path / "memory.sqlite")
    import_tree(graph, ["source one", "source two"], tree(), document_id="law", source_file="law.pdf")
    root = next(v for v in graph.graph.vs if v["metadata"].get("pageindex_node_id") == "root")
    hits = source_evidence_hits(graph, [{"id": int(root["name"]), "metadata": root["metadata"], "score": 0.9}], 5)
    assert {h["content"] for h in hits} == {"source one", "source two"}
    assert all(h["metadata"]["source_file"] == "law.pdf" for h in hits)
    assert all(h["metadata"]["page_start"] in (1, 2) for h in hits)
    assert all("summary" not in h["content"] for h in hits)


def test_tombstoned_pages_remain_in_memory_but_never_return_as_evidence(tmp_path):
    graph = engine(tmp_path / "memory.sqlite")
    import_tree(graph, ["source one", "source two"], tree(), document_id="law", source_file="law.pdf")
    page = next(v for v in graph.graph.vs if v["metadata"].get("page_start") == 1 and v["type"] == "Raw")
    graph.tombstone_node(int(page["name"]))
    root = next(v for v in graph.graph.vs if v["metadata"].get("pageindex_node_id") == "root")
    hits = source_evidence_hits(graph, [{"id": int(root["name"]), "metadata": root["metadata"]}], 5)
    assert [h["content"] for h in hits] == ["source two"]


def test_deleted_vectors_do_not_hide_live_evidence_after_restart(tmp_path):
    from retriever_worker import retrieve_hits
    path = tmp_path / "memory.sqlite"
    graph = engine(path)
    add(graph, 1, "retained source")
    for node_id in range(2, 8):
        add(graph, node_id, "deleted source")
        graph.tombstone_node(node_id)
    assert graph.index.ntotal == 1
    loaded = engine(path)
    assert loaded.index.ntotal == 1 and loaded.graph.vcount() == 7
    assert retrieve_hits(loaded, "deleted source", "GLOBAL_DENSE_WORMHOLE", 1)[0]["id"] == 1


def test_changed_document_requires_explicit_version(tmp_path):
    graph = engine(tmp_path / "memory.sqlite")
    import_tree(graph, ["old", "original"], tree(), document_id="law", source_file="law.pdf")
    with pytest.raises(ValueError, match="new document version"):
        import_tree(graph, ["changed", "original"], tree(), document_id="law", source_file="law.pdf")


@pytest.mark.parametrize("bad", [
    {"structure": [{"node_id": "x", "title": "x", "start_index": 0, "end_index": 1}]},
    {"structure": [{"node_id": "x", "title": "x", "start_index": 1, "end_index": 3}]},
    {"structure": [{"node_id": "x", "title": "x", "start_index": 2, "end_index": 1}]},
])
def test_invalid_tree_does_not_modify_memory(tmp_path, bad):
    graph = engine(tmp_path / "memory.sqlite")
    with pytest.raises(ValueError):
        import_tree(graph, ["one", "two"], bad, document_id="law", source_file="law.pdf")
    assert graph.graph.vcount() == 0 and graph._memory_revision == 0


def test_duplicate_tree_ids_and_scanned_pages_rejected(tmp_path):
    graph = engine(tmp_path / "memory.sqlite")
    bad = tree()
    bad["structure"][0]["nodes"][0]["node_id"] = "root"
    with pytest.raises(ValueError, match="Duplicate"):
        import_tree(graph, ["one", "two"], bad, document_id="law", source_file="law.pdf")
    with pytest.raises(ValueError, match="OCR"):
        import_tree(graph, ["", " "], tree(), document_id="law", source_file="law.pdf")


def test_summary_failure_rolls_back_import_and_does_not_publish_partial_pages(tmp_path):
    path = tmp_path / "memory.sqlite"
    graph = engine(path)
    add(graph, 1, "existing")
    revision = graph._memory_revision
    data = tree()
    data["structure"][0]["nodes"][0]["summary"] = ""
    def fail(texts):
        raise RuntimeError("model failure")
    with pytest.raises(RuntimeError, match="model failure"):
        import_tree(graph, ["one", "two"], data, document_id="law", source_file="law.pdf", summary_fn=fail)
    assert graph.graph.vs["content"] == ["existing"] and graph._memory_revision == revision
    assert engine(path).graph.vs["content"] == ["existing"]


def test_rewrite_failure_restores_dirty_flags_and_old_summary(tmp_path):
    graph = engine(tmp_path / "memory.sqlite")
    add(graph, 1)
    summary = graph.generate_summary_node(["original evidence"], lambda texts: "old summary", child_ids=[1])
    graph.dirty_summary_ids.add(summary)
    graph.persist_memory()
    with pytest.raises(RuntimeError):
        graph.nightly_recalc_summaries(lambda texts: (_ for _ in ()).throw(RuntimeError("failure")))
    assert summary in graph.dirty_summary_ids
    assert graph.graph.vs.find(name=str(summary))["content"] == "old summary"


def test_pdf_import_preserves_original_page_text_and_inspect_cli(tmp_path, capsys):
    import pymupdf as fitz
    from tools.long_term_memory import main
    pdf = tmp_path / "law.pdf"
    with fitz.open() as document:
        document.new_page().insert_text((72, 72), "Original legal source.")
        document.save(pdf)
    data = {"structure": [{"node_id": "1", "title": "Law", "start_index": 1, "end_index": 1}]}
    graph = engine(tmp_path / "memory.sqlite")
    import_pdf(graph, pdf, data, document_id="law")
    raw = next(v for v in graph.graph.vs if v["type"] == "Raw")
    assert raw["content"].strip() == "Original legal source."
    assert main(["--path", str(tmp_path / "memory.sqlite"), "--project-id", "test", "inspect"]) == 0
    assert json.loads(capsys.readouterr().out)["nodes"] == 2
    tree_path = tmp_path / "tree.json"
    tree_path.write_text(json.dumps(data), encoding="utf-8")
    cli_path = tmp_path / "cli.sqlite"
    assert main(["--path", str(cli_path), "--space-id", "stub-v1", "import-pageindex",
                 "--tree", str(tree_path), "--pdf", str(pdf), "--document-id", "law-v1", "--stub-encoder"]) == 0
    assert len(json.loads(capsys.readouterr().out)["nodes"]) == 2


def test_from_config_loads_memory_and_both_retrievers_refresh(tmp_path, monkeypatch):
    from config_loader import cfg
    from retriever_worker import retrieve_hits
    from soul import Reasoner
    path = tmp_path / "memory.sqlite"
    monkeypatch.setitem(cfg._data, "long_term_memory", {"enabled": True, "path": str(path),
                        "project_id": "test", "embedding_space_id": "stub-v1"})
    writer = LegalDenseGraphBuilder.from_config()
    writer.set_encoder(StubEncoder(writer.dim))
    reader = LegalDenseGraphBuilder.from_config()
    reader.set_encoder(StubEncoder(reader.dim))
    add(writer, 1, "original evidence")
    assert retrieve_hits(reader, "original evidence", "GLOBAL_DENSE_WORMHOLE", 5)[0]["content"] == "original evidence"
    add(writer, 2, "new source")
    reasoner = Reasoner(reader, lambda text: reader.encode_text(text)["dense"], SimpleNamespace())
    assert reasoner.retrieve("new source") == "new source"


def test_bridge_cold_start_preserves_unparented_sources_and_id_counter(tmp_path):
    from dataset.IncrementalMemoryManager import IncrementalMemoryManager, MockGraphDB
    from dataset.memory_graph_bridge import MemoryGraphBridge
    path = tmp_path / "memory.sqlite"
    graph = engine(path)
    add(graph, 100, "existing unparented source")
    loaded = engine(path)
    manager = IncrementalMemoryManager(MockGraphDB(), {})
    bridge = MemoryGraphBridge(manager, loaded)
    assert 100 in manager.db.nodes
    new_id = bridge.inject_knowledge("new durable source")
    assert new_id > 100
    assert new_id in [int(v) for v in engine(path).graph.vs["name"]]


def test_bridge_summary_retains_original_evidence_after_restart(tmp_path):
    from dataset.IncrementalMemoryManager import IncrementalMemoryManager, MockGraphDB
    from dataset.memory_graph_bridge import MemoryGraphBridge
    path = tmp_path / "memory.sqlite"
    graph = engine(path)
    manager = IncrementalMemoryManager(MockGraphDB(), {})
    bridge = MemoryGraphBridge(manager, graph)
    source = bridge.inject_knowledge("source evidence")
    bridge.sync_dirty_summaries(lambda texts: "derived summary")
    loaded = engine(path)
    summary = next(v for v in loaded.graph.vs if v["type"] == "Summary")
    hits = source_evidence_hits(loaded, [{"id": int(summary["name"]), "metadata": summary["metadata"]}], 5)
    assert [hit["id"] for hit in hits] == [source]
    rebuilt = IncrementalMemoryManager(MockGraphDB(), {})
    MemoryGraphBridge(rebuilt, loaded)
    assert rebuilt.db.get_children(int(summary["name"])) == [source]


def test_bridge_summary_failure_keeps_runtime_dirty_flags_and_durable_sources(tmp_path):
    from dataset.IncrementalMemoryManager import IncrementalMemoryManager, MockGraphDB
    from dataset.memory_graph_bridge import MemoryGraphBridge
    path = tmp_path / "memory.sqlite"
    graph = engine(path)
    manager = IncrementalMemoryManager(MockGraphDB(), {})
    bridge = MemoryGraphBridge(manager, graph)
    bridge.inject_knowledge("source evidence")
    dirty = manager.db.get_dirty_nodes()
    revision = graph._memory_revision
    with pytest.raises(ValueError, match="empty"):
        bridge.sync_dirty_summaries(lambda texts: "")
    assert manager.db.get_dirty_nodes() == dirty
    assert graph._memory_revision == revision
    assert engine(path).graph.vs["content"] == ["source evidence"]


def test_persistent_reasoner_uses_ingestion_encoder_not_router_vector(tmp_path):
    from soul import Reasoner
    graph = engine(tmp_path / "memory.sqlite")
    add(graph, 1)
    reasoner = Reasoner(graph, lambda text: pytest.fail("Foreign embedding space"), SimpleNamespace())
    assert reasoner.retrieve("original evidence") == "original evidence"


def test_persistent_worker_does_not_automatically_switch_to_lora(tmp_path, monkeypatch):
    import retriever_worker
    graph = engine(tmp_path / "memory.sqlite")
    monkeypatch.setattr(retriever_worker.LegalDenseGraphBuilder, "from_config", lambda: graph)
    monkeypatch.setattr(graph, "load_lora_weights", lambda path: pytest.fail("Encoder switched"))
    assert retriever_worker.build_engine() is graph


def test_memory_mode_bypasses_unversioned_answers(monkeypatch):
    from soul import UnifiedQueryRouter_Soul
    from config_loader import cfg
    monkeypatch.setitem(cfg._data, "long_term_memory", {"enabled": True})
    cache = SimpleNamespace(lookup=lambda vector: pytest.fail("Unversioned cache used"),
                            add=lambda *args: pytest.fail("Unversioned cache updated"))
    router = SimpleNamespace(process=lambda query: {"status": "simple_rag", "response": "fresh"})
    assert UnifiedQueryRouter_Soul(cache=cache, query_router=router).process("query")["response"] == "fresh"


def test_tombstoned_summary_is_not_reactivated_during_bootstrap(tmp_path):
    from dataset.IncrementalMemoryManager import IncrementalMemoryManager, MockGraphDB
    from dataset.memory_graph_bridge import MemoryGraphBridge
    path = tmp_path / "memory.sqlite"
    graph = engine(path)
    add(graph, 1)
    summary = graph.generate_summary_node(["original evidence"], lambda texts: "summary", child_ids=[1])
    graph.tombstone_node(summary)
    manager = IncrementalMemoryManager(MockGraphDB(), {})
    bridge = MemoryGraphBridge(manager, engine(path))
    bridge.sync_dirty_summaries(lambda texts: pytest.fail("Deleted summary rewritten"))
    assert manager.db.nodes[summary]["status"] == "tombstone"
    assert summary not in manager.summary_embeddings
