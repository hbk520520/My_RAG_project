from decimal import DecimalException
import json

import pytest

from agent_tools.calculator import calculate
from agent_tools.sandbox.runner import calculate as container_calculate


def plan(left="0.1", right="3", operation="multiply"):
    return {"formula": "a op b", "inputs": {"a": left, "b": right},
            "steps": [{"name": "total", "operation": operation, "left": "a", "right": "b"}], "result": "total"}


def test_decimal_amount_is_exact_and_formula_is_retained():
    value = calculate(plan())
    assert value["amount"] == "0.30" and value["formula"] == "a op b"


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "1e100", "1e-100"])
def test_invalid_numeric_inputs_are_rejected(bad):
    with pytest.raises(ValueError):
        calculate(plan(left=bad))


def test_division_by_zero_and_forward_references_do_not_produce_amounts():
    with pytest.raises(DecimalException):
        calculate(plan(right="0", operation="divide"))
    value = plan()
    value["steps"][0]["left"] = "future"
    with pytest.raises(ValueError):
        calculate(value)


def test_container_runner_supports_result_variable_and_bounds_output():
    assert container_calculate("result = 6*7")["output"] == "42"
    assert "limit" in container_calculate("print('x'*1000)", limit=20)["error"]
    assert container_calculate("raise ValueError('bad')")["error"]


def test_model_invalid_json_propagates_instead_of_becoming_irrelevant():
    from legal_agent.operators import AgenticNodesOperator
    ops = AgenticNodesOperator.__new__(AgenticNodesOperator)
    ops._call_messages = lambda *args, **kwargs: "not-json"
    with pytest.raises(ValueError):
        ops.grade_facts("task", "source")


def test_partial_retry_counter_survives_replanner_transition():
    from legal_agent.nodes import node_replanner
    state = {"user_query": "task", "task_queue": [{"task_desc": "task"}],
             "past_observations": [], "retry_context": {"status": "partial", "fail_count": 2}}
    result = node_replanner(state, None)
    assert result["retry_context"]["fail_count"] == 2


def test_backend_configuration_error_is_terminal_and_does_not_rewrite_code():
    from legal_agent.nodes import node_execute_code
    state = {"calculation_mode": "code", "generated_code": "result = 1",
             "sandbox_retries": 0, "retry_context": {}}
    result = node_execute_code(
        state,
        runner=lambda code, session_id=None: {
            "calc_result": None, "error": "sandbox.backend is invalid",
            "session_id": None, "via": "space", "error_kind": "configuration"})
    assert result["calculation_ok"] is False
    assert result["execution_status"] == "calculation_failed"
    assert result["retry_context"] == {}
    assert result["sandbox_retries"] == 3
    assert "后端不可用" in result["calc_result"]


def test_cleanup_node_keeps_handle_when_cleanup_is_not_proven(monkeypatch):
    import legal_agent.nodes as nodes
    monkeypatch.setattr(nodes, "destroy_sandbox_session", lambda session_id: False)
    result = nodes.node_cleanup_sandbox({"sandbox_session_id": "space:r:0:sbx_x"})
    assert result == {"sandbox_session_id": "space:r:0:sbx_x"}

    monkeypatch.setattr(nodes, "destroy_sandbox_session", lambda session_id: True)
    assert nodes.node_cleanup_sandbox(
        {"sandbox_session_id": "space:r:0:sbx_x"}) == {"sandbox_session_id": None}
