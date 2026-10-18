"""Exercise actual LangGraph nodes and Worker loops with offline model/bus adapters."""
import copy
from datetime import date
import json
from types import SimpleNamespace

import pytest

from config_loader import cfg
from causal_evidence import graph as graph_module
from causal_evidence.models import EvidenceRecord
from causal_evidence.pipeline import EvidenceSettings, analyze_evidence, evidence_from_hit
from tests.test_causal_evidence import (ByteEncoding, CUTOFF, ScriptedCalls, TEXT, payload, record)


@pytest.fixture
def causal_mode(monkeypatch):
    monkeypatch.setitem(cfg._data, "causal_evidence", {
        "enabled": True, "prediction_cutoff": CUTOFF.isoformat(), "targets": ["资金紧张"]})
    original = graph_module.chunk_record
    monkeypatch.setattr(graph_module, "chunk_record", lambda source, **kwargs:
                        original(source, encoding=ByteEncoding()))


def test_pipeline_admits_only_pre_cutoff_text_and_returns_weighted_graph(causal_mode):
    future = EvidenceRecord(record_id="future", text="未来泄漏内容", timestamp=date(2025, 11, 1))
    call = ScriptedCalls()
    result = analyze_evidence([record(), future], "资金紧张", call,
                              settings=EvidenceSettings(prediction_cutoff=CUTOFF, targets=["资金紧张"]),
                              extractor_id="fixture")
    assert "未来泄漏内容" not in result["context"]
    assert result["weighting_status"] == "scored"
    assert result["weighting"]["chains"][0]["score"] == 0.9
    assert result["graph"]["source_chunks"][0]["record_id"] == "news-1"
    assert len(call.calls) == 2


def test_no_cutoff_or_target_does_not_invent_weighting(causal_mode):
    first = analyze_evidence([record()], "anything", ScriptedCalls(),
                             settings=EvidenceSettings(), extractor_id="fixture")
    assert first["weighting"] is None and first["weighting_status"] == "not_scored_without_cutoff"
    second = analyze_evidence([record()], "unmatched question", ScriptedCalls(),
                              settings=EvidenceSettings(prediction_cutoff=CUTOFF), extractor_id="fixture")
    assert second["weighting"] is None and second["weighting_status"] == "no_target_anchor"


def test_source_family_prevents_copied_records_inflating_recurrence(causal_mode):
    records = [record().model_copy(update={"record_id": str(i), "source_id": "same-original"}) for i in range(2)]
    result = analyze_evidence(records, "资金紧张", ScriptedCalls(),
                              settings=EvidenceSettings(prediction_cutoff=CUTOFF), extractor_id="fixture")
    assert all(h["support_records"] == 1 for h in result["weighting"]["hyperedges"].values())
    assert len(result["weighting"]["chains"]) == 1


def test_missing_calibration_artifact_fails_loudly(causal_mode):
    with pytest.raises(FileNotFoundError):
        analyze_evidence([record()], "资金紧张", ScriptedCalls(),
                         settings=EvidenceSettings(calibration_path="outputs/does-not-exist.json"),
                         extractor_id="fixture")


def test_retrieval_metadata_is_not_invented():
    result = evidence_from_hit({"id": 7, "content": TEXT, "metadata": {"similarity": 0.7}})
    assert result.record_id == "7" and result.similarity == 0.7
    assert result.timestamp is None and not result.availability_verified


def test_langgraph_executes_verification_before_grading_and_passes_weights_to_report(causal_mode):
    from soul import AgenticNodesOperator, Reasoner, build_plan_replan_agent
    from tests.harness import initial_state

    calls = []
    evidence_call = ScriptedCalls()
    class Ops(AgenticNodesOperator):
        def __init__(self):
            pass  # no real client; all model calls must pass the existing seam

        def _call_messages(self, messages, **kwargs):
            system = messages[0]["content"]
            calls.append(messages)
            if "Causal Extraction:" in system or "Causal Verification:" in system:
                return evidence_call(messages)
            if "Meta-Planner" in system:
                return json.dumps({"strategy_queue": ["调查资金紧张"]})
            if "事实调查官" in system:
                assert "verified_causal_relations" in messages[-1]["content"]
                return json.dumps({"status": "sufficient", "extracted_facts": [TEXT], "rationale": "fixture"})
            if "资深律师" in system:
                assert "weighted_support" in messages[-1]["content"]
                return "基于原文的最终意见。"
            raise AssertionError(system)

    ops = Ops()
    reasoner = Reasoner(None, None, ops)
    reasoner.retrieve_records = lambda query: [record()]
    agent = build_plan_replan_agent(reasoner, ops)
    result = agent.invoke(initial_state("预算削减是否导致资金紧张？"))
    assert result["final_report"] == "基于原文的最终意见。"
    assert len(result["causal_evidence"]) == 1
    assert result["causal_evidence"][0]["weighting_status"] == "scored"
    systems = [m[0]["content"] for m in calls]
    assert next(i for i, s in enumerate(systems) if "Causal Verification:" in s) < next(
        i for i, s in enumerate(systems) if "事实调查官" in s)


def test_disabled_reasoner_preserves_old_behavior(monkeypatch):
    from soul import Reasoner
    monkeypatch.setitem(cfg._data, "causal_evidence", {"enabled": False})
    ops = SimpleNamespace(grade_facts=lambda query, docs:
                          {"status": "sufficient", "extracted_facts": ["原事实"], "rationale": "old"})
    reasoner = Reasoner(None, None, ops)
    reasoner.retrieve = lambda query: "原资料"
    reasoner.retrieve_records = lambda query: pytest.fail("Disabled mode must not run causal extraction")
    result = reasoner.answer("task", "query")
    assert result["fact"] == "原事实" and "causal_evidence" not in result


class MemoryState:
    def __init__(self, state):
        self.state = copy.deepcopy(state)
        self.saved = []

    def load_state(self, session):
        return copy.deepcopy(self.state)

    def save_state(self, session, state):
        self.state = copy.deepcopy(state)
        self.saved.append(copy.deepcopy(state))


class Consumer:
    def __init__(self):
        self.commits = 0

    def __iter__(self):
        return iter([SimpleNamespace(key="session", value={"session_id": "session"})])

    def commit(self):
        self.commits += 1

    def close(self):
        pass


class Producer:
    def __init__(self, store):
        self.messages = []
        self.store = store

    def send(self, topic, **kwargs):
        self.messages.append((topic, kwargs))
        # State handed off to the next worker must already include the audit graph.
        assert "causal_evidence" in self.store.state["past_observations"][-1]

    def flush(self):
        pass

    def close(self):
        pass


def initial_worker_state():
    return {"user_query": "资金紧张", "task_queue": ["调查预算削减"], "retry_context": {},
            "past_observations": [{"docs": [TEXT], "evidence_records": [record().model_dump(mode="json")]}]}


def run_grader(monkeypatch, state, grade):
    import grader_worker
    import causal_evidence.pipeline as pipeline
    store, consumer = MemoryState(state), Consumer()
    producer = Producer(store)
    evidence_call = ScriptedCalls()
    monkeypatch.setattr(pipeline, "causal_completion", evidence_call)
    monkeypatch.setattr(grader_worker, "StateManager", lambda **kwargs: store)
    monkeypatch.setattr(grader_worker, "create_consumer", lambda *args, **kwargs: consumer)
    monkeypatch.setattr(grader_worker, "create_producer", lambda **kwargs: producer)
    monkeypatch.setattr(grader_worker, "call_grader_llm", grade)
    monkeypatch.setattr(grader_worker, "quarantine_message", lambda *args: pytest.fail("Unexpected worker failure"))
    grader_worker.main()
    return store, consumer, producer, evidence_call


@pytest.mark.parametrize("status", ["sufficient", "partial", "irrelevant"])
def test_real_grader_worker_saves_audit_before_dispatch(causal_mode, monkeypatch, status):
    def grade(query, context):
        assert "weighted_support" in context
        return {"status": status, "extracted_facts": [TEXT], "missing_info": "more"}
    store, consumer, producer, evidence_call = run_grader(monkeypatch, initial_worker_state(), grade)
    assert consumer.commits == 1 and len(evidence_call.calls) == 2
    assert store.state["past_observations"][-1]["causal_evidence"]["weighting_status"] == "scored"
    assert producer.messages


def test_post_cutoff_worker_never_calls_extractor_verifier_or_grader(causal_mode, monkeypatch):
    state = initial_worker_state()
    state["past_observations"][0]["evidence_records"][0]["timestamp"] = "2025-11-01"
    store, consumer, _, calls = run_grader(monkeypatch, state,
                                         lambda *args: pytest.fail("No admitted evidence"))
    assert not calls.calls and consumer.commits == 1
    assert store.state["past_observations"][-1]["docs"] == ""


def test_legacy_worker_observation_requires_reretrieval_in_enabled_mode(causal_mode):
    from grader_worker import prepare_grader_evidence
    with pytest.raises(ValueError, match="rerun retrieval"):
        prepare_grader_evidence({}, "task", {"docs": [TEXT]}, ScriptedCalls())


def test_retriever_compatibility_and_metadata_survive():
    from retriever_worker import retrieve_docs, retrieve_hits
    from tests.harness import ScriptedEncoder, make_offline_graph_engine
    engine = make_offline_graph_engine(
        [{"id": 1, "type": "Raw", "content": TEXT,
          "metadata": {"timestamp": "2025-09-01", "source_id": "original"}}],
        encoder=ScriptedEncoder(topics={"budget": ["预算", "资金"]}))
    hits = retrieve_hits(engine, "预算", "GLOBAL_DENSE_WORMHOLE", 5)
    assert hits[0]["metadata"]["timestamp"] == "2025-09-01"
    assert retrieve_docs(engine, "预算", "GLOBAL_DENSE_WORMHOLE", 5) == [TEXT]


def test_real_reasoner_worker_receives_causal_weights(causal_mode, monkeypatch):
    import reasoner_worker
    analysis = analyze_evidence([record()], "资金紧张", ScriptedCalls(),
                                settings=EvidenceSettings(prediction_cutoff=CUTOFF), extractor_id="fixture")
    state = {"user_query": "资金紧张", "task_queue": [], "past_observations": [
        {"task": "调查预算", "extracted_facts": [TEXT], "causal_evidence": analysis}]}
    store, consumer = MemoryState(state), Consumer()
    seen = []
    def reason(query, text):
        assert "weighted_support" in text
        seen.append(text)
        return "推理"
    def generate(query, chain):
        assert chain[0]["causal_evidence"] == analysis
        return "最终意见"
    monkeypatch.setattr(reasoner_worker, "StateManager", lambda **kwargs: store)
    monkeypatch.setattr(reasoner_worker, "create_consumer", lambda *args, **kwargs: consumer)
    monkeypatch.setattr(reasoner_worker, "call_reasoner_llm", reason)
    monkeypatch.setattr(reasoner_worker, "call_generator_llm", generate)
    monkeypatch.setattr(reasoner_worker, "detect_calculation_needed", lambda *args: False)
    reasoner_worker.main()
    assert seen and store.state["final_report"] == "最终意见" and consumer.commits == 1


def test_calibration_cli_saves_reproducible_artifact_and_heldout_report(tmp_path, capsys):
    from tools.calibrate_causal_support import main
    from tests.test_causal_evidence import labeled_rows
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(r.model_dump_json() for r in labeled_rows()), encoding="utf-8")
    output = tmp_path / "model.json"
    assert main([str(labels), "--extractor-id", "fixture", "--trained-through", "2025-01-01",
                 "--out", str(output)]) == 0
    assert output.is_file()
    report = json.loads(capsys.readouterr().out)["held_out"]
    assert report["raw"]["n"] == report["calibrated"]["n"]


def test_causal_mode_bypasses_unversioned_cache_and_forwards_request_context(causal_mode):
    from soul import UnifiedQueryRouter_Soul
    seen = []
    class Cache:
        def lookup(self, vector):
            pytest.fail("Legacy answer might contain post-cutoff information")

        def add(self, vector, answer):
            pytest.fail("Causal result must not enter an unversioned cache")

    class Router:
        def process(self, query, **kwargs):
            seen.append(kwargs)
            return {"status": "agent_execution", "agent_report": "result"}

    router = UnifiedQueryRouter_Soul(cache=Cache(), query_router=Router())
    assert router.process("资金紧张", prediction_cutoff="2025-10-01",
                          causal_targets=["资金紧张"])["agent_report"] == "result"
    assert seen == [{"prediction_cutoff": "2025-10-01", "causal_targets": ["资金紧张"]}]


def test_disabled_wrapper_keeps_existing_cache_contract(monkeypatch):
    from soul import UnifiedQueryRouter_Soul
    monkeypatch.setitem(cfg._data, "causal_evidence", {"enabled": False})
    cache = SimpleNamespace(lookup=lambda vector: "cached")
    router = SimpleNamespace(embedder=SimpleNamespace(encode=lambda *args, **kwargs: [1.0]),
                             process=lambda query: pytest.fail("Cache hit should bypass the router"))
    assert UnifiedQueryRouter_Soul(cache=cache, query_router=router).process("query")["response"] == "cached"
