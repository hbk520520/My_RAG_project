"""Shared document navigation budgets, errors and real execution paths."""
import json
from types import SimpleNamespace

import pytest

from dataset.document_navigation import NavigationSettings, retrieve
from dataset.document_store import DocumentReader
from tests.test_pageindex import imported
from tests.test_causal_evidence import ByteEncoding, TEXT


@pytest.fixture(autouse=True)
def offline_tokens(monkeypatch):
    import tiktoken
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: ByteEncoding())


def scripted(*actions):
    pending = iter(actions)
    calls = []
    def call(messages):
        calls.append(list(messages))
        return json.dumps(next(pending))
    call.calls = calls
    return call


def page_action(graph, pages, **kwargs):
    doc = DocumentReader.from_engine(graph).list_documents()[0]
    return {"action": "read_pages", "document_id": doc["document_id"], "version": doc["version"],
            "pages": pages, **kwargs}


def run(graph, call, vector=None, **settings):
    return retrieve(graph, "budget question", vector or (lambda: pytest.fail("Unexpected vector retrieval")), call,
                    settings=NavigationSettings(navigation="agent", **settings), encoding=ByteEncoding())


def test_agent_reads_structure_then_original_pages_and_finishes():
    graph = imported()
    page = page_action(graph, [1])
    call = scripted({**page, "action": "read_structure", "pages": None}, page, {"action": "finish"})
    result = run(graph, call)
    assert [hit["content"] for hit in result.hits] == [TEXT]
    assert result.trace["model_calls"] == 3 and result.trace["status"] == "complete"
    assert result.trace["read_content_tokens"] == len(TEXT.encode())
    assert "Employment" in call.calls[1][-1]["content"]
    assert '"child_count": 1' in call.calls[1][-1]["content"]
    assert TEXT in call.calls[2][-1]["content"]


def test_small_complete_documents_read_directly_without_model_calls():
    result = retrieve(imported(), "query", lambda: pytest.fail("vector"), lambda messages: pytest.fail("model"),
                      settings=NavigationSettings(), encoding=ByteEncoding())
    assert result.trace["mode"] == "flat" and len(result.hits) == 2
    assert result.trace["model_calls"] == 0


def test_non_document_and_vector_configuration_keep_existing_baseline():
    from tests.test_long_term_memory import engine, add
    graph = engine()
    add(graph, 1, "standalone article", document_id="legacy")
    sentinel = [{"id": 1, "content": "baseline"}]
    for graph, settings in [(graph, NavigationSettings()), (imported(), NavigationSettings(navigation="vector"))]:
        result = retrieve(graph, "query", lambda: sentinel, lambda messages: pytest.fail("model"), settings=settings)
        assert result.hits is sentinel and result.trace["mode"] == "vector"


def test_explicit_vector_action_uses_original_baseline_with_same_read_budget():
    graph = imported()
    original = DocumentReader.from_engine(graph).read_pages("news-v1", [2])
    result = run(graph, scripted({"action": "vector_search"}, {"action": "finish"}), vector=lambda: original)
    assert result.hits == original and result.trace["read_records"] == 1


@pytest.mark.parametrize("settings", [{"max_pages": 1}, {"max_read_tokens": 1}])
def test_budget_rejection_never_silently_truncates_original_pages(settings):
    graph = imported()
    call = scripted(page_action(graph, [1, 2]), {"action": "finish"})
    result = run(graph, call, **settings)
    assert not result.hits and result.trace["status"] == "empty"
    assert result.trace["actions"][0]["status"] == "read_budget_exceeded"
    assert result.trace["read_content_tokens"] == 0


def test_duplicate_reads_do_not_consume_budget_twice():
    graph = imported()
    result = run(graph, scripted(page_action(graph, [1]), page_action(graph, [1]), {"action": "finish"}), max_pages=1)
    assert len(result.hits) == 1 and result.trace["read_records"] == 1
    assert result.trace["actions"][1]["status"] == "already_read"


def test_agent_can_reduce_a_rejected_batch_and_retry_within_budget():
    graph = imported()
    result = run(graph, scripted(page_action(graph, [1, 2]), page_action(graph, [2]), {"action": "finish"}), max_pages=1)
    assert [hit["content"] for hit in result.hits] == ["original second page"]
    assert result.trace["actions"][0]["status"] == "read_budget_exceeded"


def test_document_catalog_paginates_without_losing_documents():
    from dataset.page_index_memory import import_tree
    from tests.test_long_term_memory import tree
    graph = imported()
    import_tree(graph, ["one", "two"], tree(), document_id="second-doc", source_file="second.pdf")
    call = scripted({"action": "list_documents", "offset": 1}, {"action": "finish"})
    result = run(graph, call, max_structure_tokens=280)
    initial = json.loads(call.calls[0][-1]["content"])["catalog"]
    assert len(initial["items"]) == 1 and initial["next_offset"] == 1
    assert "second-doc" in call.calls[1][-1]["content"]
    assert result.trace["read_records"] == 0


def test_turn_limit_returns_only_read_evidence_and_explicit_partial_status():
    graph = imported()
    result = run(graph, scripted(page_action(graph, [2])), max_turns=1)
    assert len(result.hits) == 1 and result.trace["status"] == "turn_budget_exhausted"


@pytest.mark.parametrize("change", [{"document_id": "foreign"}, {"version": "stale"}, {"pages": [0]},
                                    {"pages": [True]}, {"action": "execute"}, {"path": "secret.pdf"},
                                    {"version": None}])
def test_illegal_actions_versions_and_pages_fail_without_fallback(change):
    graph = imported()
    with pytest.raises((ValueError, KeyError)):
        run(graph, scripted({**page_action(graph, [1]), **change}))


def test_model_failure_is_not_reported_as_empty_success():
    def fail(messages):
        raise RuntimeError("model unavailable")
    with pytest.raises(RuntimeError, match="model unavailable"):
        run(imported(), fail)


def test_tool_payload_supports_the_existing_memory_metadata_types():
    from datetime import date
    graph = imported()
    hit = DocumentReader.from_engine(graph).read_pages("news-v1", [1])[0]
    graph.graph.vs.find(name=str(hit["id"]))["metadata"]["timestamp"] = date(2025, 9, 1)
    call = scripted(page_action(graph, [1]), {"action": "finish"})
    assert run(graph, call).hits[0]["metadata"]["timestamp"] == date(2025, 9, 1)
    assert "2025-09-01" in call.calls[1][-1]["content"]


def test_structure_selection_and_catalog_pagination_are_bounded():
    graph = imported()
    page = page_action(graph, [1])
    call = scripted({**page, "action": "read_structure", "pages": None, "node_id": "root"},
                    {"action": "list_documents", "offset": 1}, {"action": "finish"})
    result = run(graph, call)
    assert not result.hits and "Contract" in call.calls[1][-1]["content"]
    assert '"items": []' in call.calls[2][-1]["content"]
    with pytest.raises(ValueError, match="max_structure_tokens"):
        run(graph, scripted(), max_structure_tokens=1)
    with pytest.raises(KeyError, match="section"):
        run(graph, scripted({**page, "action": "read_structure", "pages": None, "node_id": "absent"}))


def test_sync_and_async_retrievers_share_navigation_and_preserve_versions(monkeypatch):
    from config_loader import cfg
    from soul import Reasoner
    import retriever_worker
    import causal_evidence.pipeline as pipeline
    graph = imported()
    monkeypatch.setitem(cfg._data, "pageindex", {"enabled": True, "navigation": "agent"})
    def new_call():
        return scripted(page_action(graph, [1]), {"action": "finish"})
    sync = new_call()
    reasoner = Reasoner(graph, lambda text: pytest.fail("vector"), SimpleNamespace(
        _call_messages=lambda messages, **kwargs: sync(messages)))
    records = reasoner.retrieve_records("budget")
    assert records[0].text == TEXT and records[0].document_version == page_action(graph, [1])["version"]
    assert reasoner.retrieval_trace["model_calls"] == 2
    monkeypatch.setattr(pipeline, "causal_completion", new_call())
    trace = {}
    hits = retriever_worker.retrieve_hits(graph, "budget", "GRAPH_TRAVERSAL", 5, retrieval_trace=trace)
    assert hits[0]["content"] == TEXT and trace == reasoner.retrieval_trace


def test_full_langgraph_navigation_chain_and_citation_path(monkeypatch):
    from config_loader import cfg
    from soul import Reasoner, build_plan_replan_agent
    from tests.harness import initial_state
    from tests.test_chain_integration import make_ops
    from tests.test_pageindex import cited
    from causal_evidence import graph as graph_module
    graph = imported()
    monkeypatch.setitem(cfg._data, "pageindex", {"enabled": True, "navigation": "agent"})
    original = graph_module.chunk_record
    monkeypatch.setattr(graph_module, "chunk_record", lambda record, **kwargs: original(record, encoding=ByteEncoding()))
    navigation = scripted(page_action(graph, [1]), {"action": "finish"})
    ops = make_ops()
    complete = ops._call_messages
    def call(messages, **kwargs):
        if "Document Navigation:" in messages[0]["content"]:
            return navigation(messages)
        if "Read evidence catalog:" in messages[-1]["content"]:
            return cited()
        return complete(messages, **kwargs)
    ops._call_messages = call
    agent = build_plan_replan_agent(Reasoner(graph, None, ops), ops)
    result = agent.invoke(initial_state("预算问题"))
    assert "news.pdf，第 1 页" in result["final_report"]
    assert result["causal_evidence"][0]["retrieval_trace"]["model_calls"] == 2
    assert result["citations"][0]["quote"] == TEXT


@pytest.mark.parametrize("valid", [True, False])
def test_worker_saves_read_trace_and_evidence_before_dispatch(monkeypatch, valid):
    import retriever_worker
    import causal_evidence.pipeline as pipeline
    from config_loader import cfg
    from tests.test_causal_pipeline import Consumer, MemoryState
    graph = imported()
    monkeypatch.setitem(cfg._data, "pageindex", {"enabled": True, "navigation": "agent"})
    call = scripted(page_action(graph, [2] if valid else [9]), {"action": "finish"})
    store = MemoryState({"user_query": "query", "task_queue": ["budget"]})
    consumer, sent, errors = Consumer(), [], []
    producer = SimpleNamespace(send=lambda *args, **kwargs: sent.append(store.state["past_observations"][-1]),
                               flush=lambda: None)
    monkeypatch.setattr(pipeline, "causal_completion", call)
    monkeypatch.setattr(retriever_worker, "StateManager", lambda **kwargs: store)
    monkeypatch.setattr(retriever_worker, "create_consumer", lambda *args, **kwargs: consumer)
    monkeypatch.setattr(retriever_worker, "create_producer", lambda **kwargs: producer)
    monkeypatch.setattr(retriever_worker, "build_engine", lambda: graph)
    monkeypatch.setattr(retriever_worker, "quarantine_message", lambda msg, exc, logger: errors.append(exc))
    retriever_worker.main()
    assert consumer.commits == 1
    if valid:
        assert len(sent) == 1 and not errors
        assert sent[0]["retrieval_trace"]["status"] == "complete"
        assert sent[0]["evidence_records"][0]["page_start"] == 2
    else:
        assert errors and not sent and not store.saved
