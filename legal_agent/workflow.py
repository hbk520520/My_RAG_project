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
def build_plan_replan_agent(reasoner: Reasoner, agentic_ops: AgenticNodesOperator,
                            checkpointer=None):
    """
    :param checkpointer: LangGraph checkpointer（P1 起用于"挂起等远端结果 + 跨进程恢复"）。
                         传 None 即纯内存运行，行为与之前完全一致。
    """
    builder = StateGraph(AgentState)

    # 闭包注入依赖
    def l0_gateway(state): return node_l0_gateway(state)
    def planner(state): return node_planner(state, agentic_ops)
    def executor(state): return node_executor(state, reasoner)
    def replanner(state): return node_replanner(state, agentic_ops)
    def generate(state): return node_generate(state, agentic_ops)
    def write_code(state): return node_write_code(state, agentic_ops)
    def execute_code(state): return node_execute_code(state)
    def inject_result(state): return node_inject_calc_result(state)
    def cleanup(state): return node_cleanup_sandbox(state)

    # 节点注册
    builder.add_node("L0_Gateway", l0_gateway)
    builder.add_node("Planner", planner)
    builder.add_node("Executor", executor)
    builder.add_node("Replanner", replanner)
    builder.add_node("Generate", generate)
    builder.add_node("WriteCode", write_code)
    builder.add_node("ExecuteCode", execute_code)
    builder.add_node("InjectResult", inject_result)
    builder.add_node("Cleanup", cleanup)

    # 拓扑: L0 → Planner → Executor → Replanner
    builder.set_entry_point("L0_Gateway")
    builder.add_edge("L0_Gateway", "Planner")
    builder.add_edge("Planner", "Executor")
    builder.add_edge("Executor", "Replanner")

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

