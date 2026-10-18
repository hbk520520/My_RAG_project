"""PDF layout, hierarchy lifecycle and cost invariants; no external models."""
import pytest
import pymupdf

from dataset.document_index import build_pdf_tree, read_pdf_structure, diagnose_cost
from dataset.document_store import DocumentReader
from dataset.page_index_memory import import_tree
from dataset.summary_tree import postorder
from tests.test_long_term_memory import engine


def hierarchy():
    return {"toc_source": "hybrid", "structure": [{"node_id": "root", "title": "Document",
        "start_index": 1, "end_index": 3, "key_items": ["Important term"], "nodes": [
            {"node_id": "child", "title": "Child", "start_index": 1, "end_index": 1,
             "toc_source": "detected"}]}]}


def test_import_summary_uses_child_summary_and_own_uncovered_text(tmp_path):
    graph = engine(tmp_path / "memory.sqlite")
    calls = []
    def summarize(texts):
        calls.append(texts)
        return "child summary" if len(calls) == 1 else "parent summary"
    import_tree(graph, ["child original", "own original", ""], hierarchy(), document_id="doc",
                source_file="doc.pdf", summary_fn=summarize)
    assert calls == [["child original"], ["Child\nchild summary", "own original"]]
    reader = DocumentReader.from_file(tmp_path / "memory.sqlite", "test")
    root = reader.read_structure("doc")["structure"][0]
    assert root["summary"] == "parent summary" and root["key_items"] == ["Important term"]
    assert root["nodes"][0]["toc_source"] == "detected" and root["toc_source"] == "hybrid"
    assert reader.read_pages("doc", [3])[0]["content"] == ""
    assert not graph.dirty_summary_ids


def test_incremental_rewrite_children_first_and_preserves_parent_own_body(tmp_path):
    graph = engine(tmp_path / "memory.sqlite")
    import_tree(graph, ["child original", "own original", ""], hierarchy(), document_id="doc", source_file="doc.pdf")
    page_id = DocumentReader.from_engine(graph).read_pages("doc", [1])[0]["id"]
    graph._propagate_dirty(page_id)
    calls = []
    def summarize(texts):
        calls.append(texts)
        return "fresh child" if len(calls) == 1 else "fresh parent"
    graph.nightly_recalc_summaries(summarize)
    assert calls == [["child original"], ["Child\nfresh child", "own original"]]
    assert not graph.dirty_summary_ids
    assert DocumentReader.from_file(tmp_path / "memory.sqlite", "test").read_structure("doc")["structure"][0]["summary"] == "fresh parent"


def test_failed_parent_rewrite_restores_child_and_dirty_flags_without_store():
    graph = engine()
    import_tree(graph, ["one", "two", ""], hierarchy(), document_id="doc", source_file="doc.pdf")
    graph._propagate_dirty(DocumentReader.from_engine(graph).read_pages("doc", [1])[0]["id"])
    before = graph.graph.vs["content"], set(graph.dirty_summary_ids)
    calls = []
    def summarize(texts):
        calls.append(texts)
        if len(calls) == 2:
            raise RuntimeError("parent failed")
        return "new child"
    with pytest.raises(RuntimeError, match="parent failed"):
        graph.nightly_recalc_summaries(summarize)
    assert (graph.graph.vs["content"], graph.dirty_summary_ids) == before


def test_overlapping_sections_all_marked_dirty_and_deleted_evidence_not_reused():
    graph = engine()
    data = hierarchy()
    data["structure"][0]["nodes"].append({"node_id": "shared", "title": "Shared", "start_index": 1, "end_index": 1})
    import_tree(graph, ["remove me", "retained", ""], data, document_id="doc", source_file="doc.pdf",
                summary_fn=lambda texts: "OLD removed text")
    graph.tombstone_node(DocumentReader.from_engine(graph).read_pages("doc", [1])[0]["id"])
    assert len(graph.dirty_summary_ids) == 3
    calls = []
    graph.nightly_recalc_summaries(lambda texts: calls.append(texts) or "retained summary")
    assert calls == [["retained"]]
    root = DocumentReader.from_engine(graph).read_structure("doc")["structure"][0]
    assert all(child["summary"] == "" for child in root["nodes"])


def test_bridge_uses_same_section_inputs_and_acknowledges_after_commit(tmp_path):
    from dataset.IncrementalMemoryManager import IncrementalMemoryManager, MockGraphDB
    from dataset.memory_graph_bridge import MemoryGraphBridge
    graph = engine(tmp_path / "memory.sqlite")
    import_tree(graph, ["child original", "own original", ""], hierarchy(), document_id="doc", source_file="doc.pdf")
    manager = IncrementalMemoryManager(MockGraphDB(), {})
    bridge = MemoryGraphBridge(manager, graph)
    graph._propagate_dirty(DocumentReader.from_engine(graph).read_pages("doc", [1])[0]["id"])
    calls = []
    bridge.sync_dirty_summaries(lambda texts: calls.append(texts) or "updated")
    assert calls == [["child original"], ["Child\nupdated", "own original"]]
    assert not graph.dirty_summary_ids
    assert not manager.db.get_dirty_nodes()
    root = DocumentReader.from_engine(graph).read_structure("doc")["structure"][0]
    assert root["summary"] == "updated"


def test_postorder_closes_ancestors_and_rejects_cycles():
    assert postorder({1: [2], 2: [3], 3: [], 4: []}, [3]) == [3, 2, 1]
    with pytest.raises(ValueError, match="Cycle"):
        postorder({1: [2], 2: [1]}, [2])


def test_blank_section_does_not_invoke_model_or_feed_a_fabricated_summary():
    graph = engine()
    calls = []
    import_tree(graph, ["", "own original", ""], hierarchy(), document_id="doc", source_file="doc.pdf",
                summary_fn=lambda texts: calls.append(texts) or "parent summary")
    assert calls == [["own original"]]


def create_pdf(path, bookmarks=True):
    with pymupdf.open() as pdf:
        for heading in ["Chapter 1 Funding", "1.1 Budget", "Chapter 2 Delivery"]:
            page = pdf.new_page()
            page.insert_text((72, 60), heading, fontsize=16)
            page.insert_text((72, 95), "The ordinary source text describes the facts in this section.", fontsize=11)
        if bookmarks:
            pdf.set_toc([[1, "Chapter 1 Funding", 1], [1, "Chapter 2 Delivery", 3]])
        pdf.save(path)


def test_native_pdf_bookmark_and_heading_fusion_preserves_source_and_pages(tmp_path):
    path = tmp_path / "example.pdf"
    create_pdf(path)
    pages, data = read_pdf_structure(path)
    assert len(pages) == 3 and data["toc_source"] == "hybrid"
    root = data["structure"][0]
    assert [node["toc_source"] for node in root["nodes"]] == ["bookmarks", "bookmarks"]
    assert root["nodes"][0]["nodes"][0]["title"] == "1.1 Budget"
    assert root["nodes"][0]["nodes"][0]["toc_source"] == "detected"
    graph = engine()
    import_tree(graph, pages, data, document_id="doc", source_file=path.name)
    assert len(DocumentReader.from_engine(graph).read_pages("doc", [1, 2, 3])) == 3


def test_invalid_bookmarks_are_reported_and_layout_can_still_be_used(tmp_path):
    path = tmp_path / "example.pdf"
    create_pdf(path)
    with pymupdf.open(path) as pdf:
        class InvalidBookmarks:
            def __len__(self): return len(pdf)
            def __iter__(self): return iter(pdf)
            def get_toc(self, **kwargs): return [[2, "Bad level", 99]]
        data = build_pdf_tree(InvalidBookmarks(), title="doc")
    assert data["toc_source"] == "detected" and data["diagnostics"][0]["source"] == "bookmarks"


def test_plain_pdf_uses_flat_root_without_inventing_headings(tmp_path):
    path = tmp_path / "plain.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "An ordinary body paragraph.")
        pdf.save(path)
    _, data = read_pdf_structure(path)
    assert data["toc_source"] == "flat" and not data["structure"][0]["nodes"]


def test_cost_diagnostic_accounts_for_residual_pages_and_is_read_only():
    data = hierarchy()
    report = diagnose_cost(data)
    assert report[-1] == {"node_id": "root", "flat_pages": 3, "tree_pages": 3.0,
                          "residual_pages": 2, "suggest_merge": True}
    assert len(data["structure"][0]["nodes"]) == 1
    with pytest.raises(ValueError):
        diagnose_cost(data, routing_pages=-1)


def test_cli_can_build_and_persist_without_an_exported_tree(tmp_path, capsys):
    from tools.long_term_memory import main
    path = tmp_path / "example.pdf"
    create_pdf(path)
    args = ["--path", str(tmp_path / "memory.sqlite"), "--project-id", "test", "--space-id", "stub-v1"]
    assert main(args + ["import-pageindex", "--pdf", str(path), "--document-id", "doc", "--stub-encoder"]) == 0
    assert '"toc_source": "hybrid"' in capsys.readouterr().out
    assert main(args + ["cost", "--document-id", "doc"]) == 0
    assert '"residual_pages"' in capsys.readouterr().out
