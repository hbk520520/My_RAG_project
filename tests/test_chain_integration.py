"""Default application path: original retrieval → CHAIN → grading → answer."""
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from config_loader import cfg
from causal_evidence import graph as causal_graph
from causal_evidence.models import EvidenceRecord
from causal_evidence.pipeline import analyze_evidence, evidence_from_hit, EvidenceSettings
from tests.harness import ScriptedEncoder, make_offline_graph_engine, initial_state
from tests.test_causal_evidence import ByteEncoding, CUTOFF, TEXT, ScriptedCalls
from tests.test_causal_pipeline import Consumer, MemoryState, run_grader


@pytest.fixture(autouse=True)
def offline_tokens(monkeypatch):
    original = causal_graph.chunk_record
    monkeypatch.setattr(causal_graph, "chunk_record",
                        lambda record, **kwargs: original(record, encoding=ByteEncoding()))


@pytest.fixture
def sources():
    return make_offline_graph_engine([{
        "id": 1, "type": "Raw", "content": TEXT,
        "metadata": {"record_id": "news-1", "source_id": "news", "document_id": "news-v1",
                     "source_file": "news.pdf", "page_start": 2, "page_end": 2, "timestamp": "2025-09-01"},
    }], encoder=ScriptedEncoder(topics={"budget": ["预算", "资金"]}))


def test_committed_configuration_enables_chain_and_local_document_memory():
    assert cfg.get("causal_evidence", "enabled") is True
    assert cfg.get("long_term_memory", "enabled") is True
    assert cfg.get("pageindex", "navigation") == "auto"
    assert cfg.get("causal_evidence", "prediction_cutoff") is None
    assert cfg.get("causal_evidence", "calibration_path") is None


def make_ops():
    from soul import AgenticNodesOperator
    class Ops(AgenticNodesOperator):
        def __init__(self):
            self.calls = []
            self.extraction = ScriptedCalls()

        def _call_messages(self, messages, **kwargs):
            self.calls.append(messages)
            system = messages[0]["content"]
            if "Causal Extraction:" in system or "Causal Verification:" in system:
                return self.extraction(messages)
            if "Meta-Planner" in system:
                return json.dumps({"strategy_queue": ["调查预算导致资金紧张"]})
            if "事实调查官" in system:
                assert "news.pdf" in messages[-1]["content"]
                return json.dumps({"status": "sufficient", "extracted_facts": [TEXT], "rationale": "supported"})
            if "Reasoner" in system or "资深律师" in system:
                assert "verified_causal_relations" in messages[-1]["content"]
                return "基于所给原文的回答"
            raise AssertionError(system)
    return Ops()


def query_router(monkeypatch, sources, ops):
    import soul
    from query import UnifiedQueryRouter_Query
    monkeypatch.setattr(soul.LegalDenseGraphBuilder, "from_config", lambda: sources)
    monkeypatch.setattr(soul, "AgenticNodesOperator", lambda **kwargs: ops)
    router = UnifiedQueryRouter_Query.__new__(UnifiedQueryRouter_Query)
    router.route = lambda query: {"intent": "SIMPLE_QA", "source": "fixture"}
    # Routing and evidence must not accidentally share different embedding spaces.
    router.embedder = SimpleNamespace(encode=lambda *args, **kwargs: pytest.fail("Router vectors used for evidence"))
    return router


@pytest.mark.parametrize("cutoff,expected", [(None, "not_scored_without_cutoff"), (CUTOFF, "scored")])
def test_simple_query_uses_real_retrieval_verification_and_answer(monkeypatch, sources, cutoff, expected):
    ops = make_ops()
    router = query_router(monkeypatch, sources, ops)
    result = router.process("预算是否导致资金紧张？", prediction_cutoff=cutoff, causal_targets=["资金紧张"])
    assert result["status"] == "simple_rag" and result["response"] == "基于所给原文的回答"
    analysis = result["causal_evidence"]
    assert analysis["weighting_status"] == expected
    assert analysis["graph"]["causal_edges"][0]["calibrated_support"] is None
    original = json.loads(analysis["context"])["admitted_evidence"][0]
    assert (original["document_id"], original["source_file"], original["page_start"], original["page_end"]) == (
        "news-v1", "news.pdf", 2, 2)
    assert len(ops.extraction.calls) == 2


def test_simple_query_failure_is_explicit_and_never_reports_verified_evidence(monkeypatch, sources):
    ops = make_ops()
    def fail(messages):
        raise RuntimeError("extraction failed")
    ops.extraction = fail
    result = query_router(monkeypatch, sources, ops).process("预算问题")
    assert result["status"] == "agent_error" and "extraction failed" in result["agent_error"]
    assert "causal_evidence" not in result and "response" not in result


def test_default_model_adapter_propagates_transport_failure():
    from soul import LegalLLMBase
    def fail(**kwargs):
        raise ConnectionError("model offline")
    adapter = LegalLLMBase.__new__(LegalLLMBase)
    adapter.model_name = "fixture"
    adapter.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    with pytest.raises(ConnectionError, match="model offline"):
        adapter._call_messages([{"role": "user", "content": "question"}])


def test_default_grading_failure_is_not_replaced_with_ungraded_facts(monkeypatch, sources):
    from soul import Reasoner
    ops = make_ops()
    def fail(*args):
        raise ConnectionError("grader offline")
    monkeypatch.setattr(ops, "grade_facts", fail)
    monkeypatch.setattr(ops, "extract_facts", lambda *args, **kwargs: pytest.fail("Grader failure hidden"))
    with pytest.raises(ConnectionError, match="grader offline"):
        Reasoner(sources, None, ops).answer("资金紧张", "预算问题")


def test_disabled_simple_query_preserves_original_route_only_behavior(monkeypatch):
    from query import UnifiedQueryRouter_Query
    monkeypatch.setitem(cfg._data, "causal_evidence", {"enabled": False})
    router = UnifiedQueryRouter_Query.__new__(UnifiedQueryRouter_Query)
    router.route = lambda query: {"intent": "SIMPLE_QA"}
    assert router.process("question") == {"intent": "SIMPLE_QA", "status": "simple_rag"}


def test_default_langgraph_flow_keeps_original_source_and_weights_in_report(sources):
    from soul import Reasoner, build_plan_replan_agent
    ops = make_ops()
    reasoner = Reasoner(sources, None, ops)
    state = initial_state("预算削减是否导致资金紧张？")
    state.update(prediction_cutoff=CUTOFF.isoformat(), causal_targets=["资金紧张"])
    final = build_plan_replan_agent(reasoner, ops).invoke(state)
    assert final["final_report"] == "基于所给原文的回答"
    analysis = final["causal_evidence"][0]
    assert analysis["weighting_status"] == "scored"
    assert analysis["weighting"]["chains"][0]["score"] == 0.9
    assert "news.pdf" in final["past_observations"][0]


def test_default_retriever_to_grader_worker_flow_preserves_source_records(monkeypatch, sources):
    import retriever_worker
    state = {"user_query": "预算削减是否导致资金紧张？", "task_queue": ["调查预算削减"],
             "prediction_cutoff": CUTOFF.isoformat(), "causal_targets": ["资金紧张"], "past_observations": []}
    store, consumer = MemoryState(state), Consumer()
    sent = []
    def send(topic, **kwargs):
        observation = store.state["past_observations"][-1]
        assert observation["evidence_records"][0]["source_file"] == "news.pdf"
        sent.append((topic, kwargs))
    producer = SimpleNamespace(send=send, flush=lambda: None, close=lambda: None)
    monkeypatch.setattr(retriever_worker, "StateManager", lambda **kwargs: store)
    monkeypatch.setattr(retriever_worker, "create_consumer", lambda *args, **kwargs: consumer)
    monkeypatch.setattr(retriever_worker, "create_producer", lambda **kwargs: producer)
    monkeypatch.setattr(retriever_worker, "build_engine", lambda: sources)
    monkeypatch.setattr(retriever_worker, "quarantine_message", lambda *args: pytest.fail("Retriever failed"))
    retriever_worker.main()
    assert consumer.commits == 1 and sent and store.state["task_queue"] == state["task_queue"]
    def grade(task, context):
        assert "news.pdf" in context and "weighted_support" in context
        return {"status": "sufficient", "extracted_facts": [TEXT]}
    graded, consumed, _, calls = run_grader(monkeypatch, store.state, grade)
    assert consumed.commits == 1 and len(calls.calls) == 2
    assert graded.state["past_observations"][-1]["causal_evidence"]["weighting_status"] == "scored"


def test_unpersisted_generated_summary_is_navigation_not_extraction_source(sources):
    from soul import Reasoner
    from retriever_worker import retrieve_hits
    sources.generate_summary_node([TEXT], lambda texts: "资金充裕，这是摘要中的虚构关系", child_ids=[1])
    assert sources._memory_store is None
    records = Reasoner(sources, None, SimpleNamespace()).retrieve_records("资金")
    assert [r.text for r in records] == [TEXT]
    assert [h["content"] for h in retrieve_hits(sources, "资金", "GLOBAL_DENSE_WORMHOLE", 5)] == [TEXT]


def test_page_location_does_not_grant_pre_cutoff_availability():
    evidence = evidence_from_hit({"id": 1, "content": TEXT, "metadata": {
        "source_file": "law.pdf", "page_start": 1, "page_end": 1}})
    assert evidence.timestamp is None and not evidence.availability_verified
    result = analyze_evidence([evidence], "资金紧张", lambda messages: pytest.fail("Undated evidence admitted"),
                              settings=EvidenceSettings(prediction_cutoff=CUTOFF), extractor_id="test")
    assert result["weighting_status"] == "no_admitted_evidence"


def test_empty_graph_does_not_load_encoder_or_call_models():
    from dataset.graph import LegalDenseGraphBuilder
    from soul import Reasoner
    def forbidden(*args, **kwargs):
        pytest.fail("Empty graph must not encode queries or request extraction")
    empty = LegalDenseGraphBuilder(embedding_dim=8, encoder=SimpleNamespace(encode=forbidden))
    result = Reasoner(empty, None, SimpleNamespace(_call_messages=forbidden)).answer("task", "query")
    assert not result["sufficient"]
    assert result["causal_evidence"]["weighting_status"] == "no_admitted_evidence"


@pytest.mark.parametrize("pages", [{"page_start": 0}, {"page_start": 2, "page_end": 1}])
def test_invalid_source_pages_are_rejected(pages):
    with pytest.raises(ValidationError):
        EvidenceRecord(record_id="source", text=TEXT, **pages)
