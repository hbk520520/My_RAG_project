from __future__ import annotations
import logging
from config_loader import cfg
import prompts
logger = logging.getLogger("LegalAgentPlanReplan")
from langgraph.graph import StateGraph, END
from .state import AgentState
from .operators import AgenticNodesOperator
from .reasoning import Reasoner
from .nodes import (node_l0_gateway, node_planner, node_executor, node_replanner, node_generate, node_write_code, node_execute_code, node_inject_calc_result, node_cleanup_sandbox)
from .nodes import node_await_input
def build_plan_replan_agent(reasoner: Reasoner, agentic_ops: AgenticNodesOperator,
                            checkpointer=None, code_runner=None):
    """
    :param checkpointer: LangGraph checkpointer（P1 起用于"挂起等远端结果 + 跨进程恢复"）。
                         传 None 即纯内存运行，行为与之前完全一致。
    """
    builder = StateGraph(AgentState)

    # 闭包注入依赖
    def l0_gateway(state): return node_l0_gateway(state)
    def planner(state): return node_planner(state, agentic_ops)
    from .scheduling import execute_ready, replan_pending
    def executor(state): return execute_ready(state, reasoner)
    def replanner(state): return replan_pending(state, agentic_ops)
    def generate(state): return node_generate(state, agentic_ops)
    def write_code(state): return node_write_code(state, agentic_ops)
    def execute_code(state): return node_execute_code(state, runner=code_runner)
    def inject_result(state): return node_inject_calc_result(state, agentic_ops)
    def cleanup(state): return node_cleanup_sandbox(state)

    from agent_runtime.context import current_run
    def instrument(name, function):
        def invoke(state):
            context = current_run.get()
            if context:
                context.begin_node(name, state)
            result = function(state)
            if context:
                context.check()
            return result
        return invoke


    # 节点注册
    builder.add_node("L0_Gateway", instrument("L0_Gateway", l0_gateway))
    builder.add_node("Planner", instrument("Planner", planner))
    builder.add_node("Executor", instrument("Executor", executor))
    builder.add_node("Replanner", instrument("Replanner", replanner))
    builder.add_node("Generate", instrument("Generate", generate))
    builder.add_node("WriteCode", instrument("WriteCode", write_code))
    builder.add_node("ExecuteCode", instrument("ExecuteCode", execute_code))
    builder.add_node("InjectResult", instrument("InjectResult", inject_result))
    builder.add_node("Cleanup", instrument("Cleanup", cleanup))
    builder.add_node("AwaitInput", instrument("AwaitInput", node_await_input))

    # 拓扑: L0 → Planner → Executor → Replanner
    builder.set_entry_point("L0_Gateway")
    builder.add_conditional_edges("L0_Gateway", lambda state: "Generate" if state.get("intent") == "CHITCHAT" else "Planner")
    builder.add_edge("Planner", "Executor")
    builder.add_conditional_edges("Executor", lambda state: "AwaitInput" if state.get("pending_question") else "Replanner")
    builder.add_edge("AwaitInput", "Executor")

    # Replanner 后根据队列和熔断状态决定去向
    def route_after_replan(state: AgentState):
        if state.get("retry_context", {}).get("status") == "force_stop":
            return "Generate"
        if len(state.get("task_queue", [])) == 0:
            return "Generate"
        return "Executor"

    builder.add_conditional_edges("Replanner", route_after_replan)

    # Generate 后判断是否需要沙箱计算
    def route_after_generate(state: AgentState):
        if state.get("needs_sandbox_calc"):
            return "WriteCode"
        return "Cleanup"

    builder.add_conditional_edges("Generate", route_after_generate)

    # WriteCode → ExecuteCode
    builder.add_edge("WriteCode", "ExecuteCode")

    # ExecuteCode 后判断是否需要重试
    def route_after_execute(state: AgentState):
        # 已有结果（执行成功、或重试已耗尽并放弃）-> 直接注入。
        # 这是终止条件的权威判据，避免 retry_context 残留导致死循环。
        if state.get("calc_result"):
            return "InjectResult"
        retry_ctx = state.get("retry_context", {})
        sandbox_retries = state.get("sandbox_retries", 0)
        max_retries = cfg.get("sandbox", "max_retries", default=3)
        if "sandbox_error" in retry_ctx and sandbox_retries < max_retries:
            logger.info(f"沙箱重试 {sandbox_retries}/{max_retries}")
            return "WriteCode"
        return "InjectResult"

    builder.add_conditional_edges("ExecuteCode", route_after_execute)

    # InjectResult → Cleanup → END
    builder.add_edge("InjectResult", "Cleanup")
    builder.add_edge("Cleanup", END)

    return builder.compile(checkpointer=checkpointer)
