"""Compatibility identities and importable module boundaries."""
import importlib


def test_historical_imports_resolve_to_the_same_implementation():
    import query
    import soul
    import sandbox_exec
    import sandbox_manager
    from legal_agent import routing
    from legal_agent.operators import AgenticNodesOperator
    from legal_agent.workflow import build_plan_replan_agent
    from agent_tools.sandbox import executor, manager

    assert query is routing
    assert soul.AgenticNodesOperator is AgenticNodesOperator
    assert soul.build_plan_replan_agent is build_plan_replan_agent
    assert sandbox_exec is executor
    assert sandbox_manager is manager


def test_domain_modules_import_without_starting_models_or_services():
    for name in ("legal_agent.state", "legal_agent.nodes", "legal_agent.workflow",
                 "legal_agent.operators", "legal_agent.reasoning", "legal_agent.cache",
                 "agent_tools.sandbox.executor"):
        assert importlib.import_module(name)
