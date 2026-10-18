"""Local PageIndex-inspired document APIs and grounding; offline engineering tests."""
import json
from types import SimpleNamespace

import pytest

from dataset.citations import generate_cited_answer, requires_citations
from dataset.document_store import DocumentReader
from dataset.page_index_memory import import_tree
from causal_evidence.pipeline import analyze_evidence, evidence_from_hit, EvidenceSettings
from tests.test_causal_evidence import ByteEncoding, TEXT, ScriptedCalls
from tests.test_long_term_memory import engine, tree


def imported(path=None, project="test"):
    graph = engine(path, project)
    import_tree(graph, [TEXT, "original second page"], tree(), document_id="news-v1", source_file="news.pdf")
    return graph


def analysis(graph):
    hit = DocumentReader.from_engine(graph).read_pages("news-v1", [1])[0]
    return analyze_evidence([evidence_from_hit(hit)], "资金紧张", ScriptedCalls(),
                            settings=EvidenceSettings(), extractor_id="fixture", encoding=ByteEncoding())


def cited(evidence="E1", quote=TEXT):
    return json.dumps({"answer": f"原文支持该关系。[{evidence}]",
                       "citations": [{"evidence_id": evidence, "quote": quote}]}, ensure_ascii=False)


def test_read_document_structure_and_pages_without_models_or_graph_rebuild(tmp_path, monkeypatch):
    path = tmp_path / "memory.sqlite"
    graph = imported(path)
    revision = graph._memory_revision
    from dataset.graph import LegalDenseGraphBuilder
    monkeypatch.setattr(LegalDenseGraphBuilder, "from_config", lambda **kwargs: pytest.fail("Read rebuilt graph"))
    reader = DocumentReader.from_file(path, "test")
    document = reader.list_documents()[0]
    assert document["document_id"] == "news-v1" and document["page_count"] == 2
    assert len(document["version"]) == 64
    assert reader.read_structure("news-v1")["structure"][0]["nodes"][0]["title"] == "Contract"
    pages = reader.read_pages("news-v1", [2, 1, 2], version=document["version"])
    assert [p["content"] for p in pages] == ["original second page", TEXT]
    assert DocumentReader.from_file(path, "test").revision == revision
    assert DocumentReader.from_file(path, "another-project").list_documents() == []


def test_read_only_memory_cannot_create_or_write_a_database(tmp_path):
    from dataset.long_term_memory import MemoryStore
    path = tmp_path / "absent.sqlite"
    with pytest.raises(FileNotFoundError):
        DocumentReader.from_file(path, "test")
    assert not path.exists()
    imported(path)
    with pytest.raises(PermissionError):
        MemoryStore(path, "test", read_only=True).save({}, 0)


@pytest.mark.parametrize("pages", [[], [0], [3], [True], [-1]])
def test_invalid_page_requests_are_rejected(pages):
    with pytest.raises(ValueError):
        DocumentReader.from_engine(imported()).read_pages("news-v1", pages)


def test_wrong_version_and_deleted_pages_cannot_be_read(tmp_path):
    graph = imported(tmp_path / "memory.sqlite")
    reader = DocumentReader.from_engine(graph)
    with pytest.raises(ValueError, match="version"):
        reader.read_pages("news-v1", [1], version="outdated")
    with pytest.raises(KeyError):
        reader.read_structure("foreign-document")
    page_id = reader.read_pages("news-v1", [1])[0]["id"]
    graph.tombstone_node(page_id)
    with pytest.raises(ValueError, match="unavailable"):
        DocumentReader.from_file(tmp_path / "memory.sqlite", "test").read_pages("news-v1", [1])


def test_cli_document_reads_use_the_same_read_only_contract(tmp_path, capsys):
    from tools.long_term_memory import main
    path = tmp_path / "memory.sqlite"
    imported(path)
    args = ["--path", str(path), "--project-id", "test"]
    assert main(args + ["documents"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["source_file"] == "news.pdf"
    assert main(args + ["pages", "--document-id", "news-v1", "--pages", "1"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["content"] == TEXT


def test_citations_resolve_only_read_evidence_and_keep_original_version():
    evidence = analysis(imported())
    assert requires_citations([evidence])
    result = generate_cited_answer([{"role": "system", "content": "Answer"}, {"role": "user", "content": "Question"}],
                                  [evidence], lambda messages: cited())
    assert "news.pdf，第 1 页" in result["answer"]
    assert result["citations"][0]["record_id"] == "news-v1:page:1"
    assert len(result["citations"][0]["document_version"]) == 64


@pytest.mark.parametrize("reply", [cited("E2"), cited(quote="not in the original"), cited(quote=" "),
    json.dumps({"answer": "[E1] [E0]", "citations": [{"evidence_id": "E1", "quote": TEXT}]}),
    json.dumps({"answer": "No citation tag", "citations": [{"evidence_id": "E1", "quote": TEXT}]}),
    json.dumps({"answer": "[E1]", "citations": []}),
    json.dumps({"answer": "[E1]", "citations": [{"evidence_id": "E1", "quote": TEXT},
                                                {"evidence_id": "E1", "quote": TEXT}]})])
def test_unread_invented_missing_or_duplicate_citations_fail(reply):
    with pytest.raises(ValueError):
        generate_cited_answer([{"role": "system", "content": "Answer"}, {"role": "user", "content": "Question"}],
                              [analysis(imported())], lambda messages: reply)


def test_langgraph_generator_returns_validated_citations_without_changing_report_type():
    from soul import node_generate
    ops = SimpleNamespace(_call_messages=lambda messages, **kwargs: cited(),
                          generate_final_report=lambda *args: pytest.fail("Unvalidated generator used"))
    result = node_generate({"user_query": "预算问题", "task_queue": [], "past_observations": [],
                            "causal_evidence": [analysis(imported())]}, ops)
    assert isinstance(result["final_report"], str) and len(result["citations"]) == 1


def test_simple_query_validates_citations_in_the_actual_answer_path(monkeypatch):
    from tests.test_chain_integration import make_ops, query_router
    from causal_evidence import graph as graph_module
    original = graph_module.chunk_record
    monkeypatch.setattr(graph_module, "chunk_record", lambda record, **kwargs: original(record, encoding=ByteEncoding()))
    graph = imported()
    ops = make_ops()
    call = ops._call_messages
    ops._call_messages = lambda messages, **kwargs: (cited() if "Read evidence catalog:" in messages[-1]["content"]
                                                    else call(messages, **kwargs))
    result = query_router(monkeypatch, graph, ops).process(TEXT)
    assert result["status"] == "simple_rag" and result["citations"][0]["source_file"] == "news.pdf"


@pytest.mark.parametrize("valid", [True, False])
def test_worker_publishes_only_a_citation_validated_final_report(monkeypatch, valid):
    import reasoner_worker
    import causal_evidence.pipeline as pipeline
    from tests.test_causal_pipeline import Consumer, MemoryState
    store = MemoryState({"user_query": "预算问题", "task_queue": [], "past_observations": [
        {"task": "预算问题", "extracted_facts": [TEXT], "causal_evidence": analysis(imported())}]})
    consumer, errors = Consumer(), []
    monkeypatch.setattr(reasoner_worker, "StateManager", lambda **kwargs: store)
    monkeypatch.setattr(reasoner_worker, "create_consumer", lambda *args, **kwargs: consumer)
    monkeypatch.setattr(reasoner_worker, "call_reasoner_llm", lambda *args: "Reasoning")
    monkeypatch.setattr(reasoner_worker, "call_generator_llm", lambda *args: pytest.fail("Unvalidated generator"))
    monkeypatch.setattr(reasoner_worker, "detect_calculation_needed", lambda *args: False)
    monkeypatch.setattr(pipeline, "causal_completion", lambda messages: cited("E1" if valid else "E2"))
    monkeypatch.setattr(reasoner_worker, "quarantine_message", lambda msg, exc, logger: errors.append(exc))
    reasoner_worker.main()
    if valid:
        assert store.state["status"] == "completed" and store.state["citations"][0]["page_start"] == 1
    else:
        assert errors and "final_report" not in store.state and not store.saved
