import json
from types import SimpleNamespace

import pytest
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from typing import TypedDict

from agent_runtime.service import AgentService
from agent_runtime.store import RunStore
from legal_agent.operators import AgenticNodesOperator


class SimpleState(TypedDict):
    user_query: str
    final_report: str
    answer: str


def minimal_graph(saver, *, fail=None, pause=False):
    from agent_runtime.context import current_run, tool_call
    def read(state):
        current_run.get().begin_node("Read", state)
        value = tool_call("lookup", {"query": state["user_query"]}, lambda: "source")
        if fail:
            fail()
        return {"answer": value}
    def finish(state):
        current_run.get().begin_node("Finish", state)
        answer = interrupt("Please supply facts") if pause else state["answer"]
        return {"final_report": str(answer)}
    builder = StateGraph(SimpleState)
    builder.add_node("Read", read)
    builder.add_node("Finish", finish)
    builder.set_entry_point("Read")
    builder.add_edge("Read", "Finish")
    builder.add_edge("Finish", END)
    return builder.compile(checkpointer=saver)


def test_runtime_resumes_after_failure_with_persisted_tool_result(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite")
    def fail():
        raise ConnectionError("worker disappeared")
    service = AgentService(store, workflow_factory=lambda saver: minimal_graph(saver, fail=fail))
    run = service.submit({"query": "task"})["run_id"]
    assert service.execute(run)["error_kind"] == "transport"
    store.requeue(run, retry=True)
    reopened = AgentService(RunStore(store.path), workflow_factory=minimal_graph)
    result = reopened.execute(run)
    assert result["status"] == "succeeded" and result["result"]["final_report"] == "source"
    with store.connection() as db:
        assert db.execute("SELECT attempts FROM effects WHERE tool='lookup'").fetchone()[0] == 1


def test_waiting_input_survives_restart_and_completed_run_is_not_reexecuted(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite")
    factory = lambda saver: minimal_graph(saver, pause=True)
    service = AgentService(store, workflow_factory=factory)
    run = service.submit({"query": "task"})["run_id"]
    assert service.execute(run)["status"] == "waiting_input"
    resumed = AgentService(RunStore(store.path), workflow_factory=factory)
    resumed.resume(run, "user facts")
    result = resumed.execute(run)
    assert result["status"] == "succeeded" and result["result"]["final_report"] == "user facts"
    before = len(store.events(run))
    assert resumed.execute(run)["status"] == "succeeded"
    assert len(store.events(run)) == before


def test_rewind_uses_selected_checkpoint_and_increments_epoch(tmp_path):
    service = AgentService(RunStore(tmp_path / "runs.sqlite"), workflow_factory=minimal_graph)
    run = service.submit({"query": "task"})["run_id"]
    assert service.execute(run)["status"] == "succeeded"
    history = service.checkpoints(run)
    target = next(item["checkpoint_id"] for item in history if item["next"] == ["Finish"])
    assert service.rewind(run, target)["epoch"] == 1
    assert service.execute(run, epoch=0)["status"] == "queued"
    assert service.execute(run, epoch=1)["status"] == "succeeded"


@pytest.fixture
def actual_service(tmp_path, monkeypatch):
    from config_loader import cfg
    from tests.harness import make_offline_graph_engine
    from tests.test_causal_evidence import ByteEncoding
    import tiktoken
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: ByteEncoding())
    plan = {"skeleton": {"nodes": [{"id": "first", "abstract": "核实辞退", "deps": []},
                                     {"id": "second", "abstract": "核实辞退合法性", "deps": ["first"]}]},
            "concretion": {"concretions": {"first": "核实辞退", "second": "核实辞退合法性"}}}
    calls = []
    def complete(**kwargs):
        system = kwargs["messages"][0]["content"]
        calls.append(system)
        if "Meta-Planner" in system:
            text = json.dumps(plan, ensure_ascii=False)
        elif "Causal Extraction:" in system:
            text = '{"propositions":[],"causal_relations":[]}'
        elif "事实调查官" in system:
            text = '{"status":"sufficient","extracted_facts":["原文事实"]}'
        elif "Return a JSON arithmetic plan" in system:
            text = json.dumps({"formula": "a*b", "inputs": {"a": "0.1", "b": "3"},
                               "steps": [{"name": "total", "operation": "multiply", "left": "a", "right": "b"}], "result": "total"})
        else:
            text = "根据原文完成分析。"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
    class Ops(AgenticNodesOperator):
        def __init__(self):
            self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=complete)))
            self.model_name = "offline-model"
    service = AgentService(RunStore(tmp_path / "runs.sqlite"),
                           engine_factory=lambda project: make_offline_graph_engine(), operator_factory=Ops)
    return service, calls


def test_real_workflow_completes_every_task_and_keeps_model_budget(actual_service):
    service, calls = actual_service
    run = service.submit({"query": "核实辞退与劳动关系"})["run_id"]
    result = service.execute(run)
    assert result["status"] == "succeeded", result["error"]
    assert [task["task_id"] for task in result["result"]["completed_tasks"]] == ["first", "second"]
    assert result["usage"]["model_calls"] == len(calls)
    assert result["usage"]["node_steps"] > 0
    assert service.store.pin(run, 0)["snapshot"]["vertices"]


def test_real_workflow_uses_decimal_calculation_without_docker(actual_service):
    service, calls = actual_service
    run = service.submit({"query": "核实辞退并计算金额"})["run_id"]
    row = service.execute(run)
    assert row["status"] == "succeeded", row["error"]
    assert "0.30" in row["result"]["final_report"]
    assert row["result"]["calculation_ok"] is True
    arithmetic = next(i for i, prompt in enumerate(calls) if "Return a JSON arithmetic plan" in prompt)
    generator = next(i for i, prompt in enumerate(calls) if "资深律师" in prompt)
    assert arithmetic < generator


def test_real_workflow_total_model_budget_cannot_be_reported_as_success(actual_service):
    service, _ = actual_service
    run = service.submit({"query": "核实辞退", "budgets": {"model_calls": 1}})["run_id"]
    row = service.execute(run)
    assert row["status"] == "timed_out" and row["error_kind"] == "budget"


def test_query_entry_uses_durable_service_with_legacy_result_fields(actual_service):
    from query import UnifiedQueryRouter_Query
    service, _ = actual_service
    router = UnifiedQueryRouter_Query.__new__(UnifiedQueryRouter_Query)
    router._agent_service = service
    router.route = lambda query: {"intent": "COMPLEX_TASK"}
    result = router.process("核实辞退")
    assert result["status"] == "agent_execution" and result["agent_report"]
    assert service.store.get(result["run_id"])["status"] == "succeeded"


def test_actual_workflow_asks_for_user_facts_and_resumes_without_turning_them_into_sources(actual_service):
    service, _ = actual_service
    ops = service.operator_factory()
    service._operator = ops
    original = ops.client.chat.completions.create
    def complete(**kwargs):
        messages = kwargs["messages"]
        if "事实调查官" in messages[0]["content"] and "公司发来书面通知" not in messages[-1]["content"]:
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                "status": "partial", "missing_info_kind": "user", "missing_info": "公司是否发来书面通知？"})))])
        return original(**kwargs)
    ops.client.chat.completions.create = complete
    run = service.submit({"query": "核实辞退"})["run_id"]
    assert service.execute(run)["status"] == "waiting_input"
    service.resume(run, "公司发来书面通知")
    row = service.execute(run)
    assert row["status"] == "succeeded", row["error"]
    assert row["result"]["user_facts"] == ["公司发来书面通知"]
    for analysis in row["result"]["causal_evidence"]:
        assert all("公司发来书面通知" not in chunk["text"] for chunk in analysis["graph"]["source_chunks"])
