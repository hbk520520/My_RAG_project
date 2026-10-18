from __future__ import annotations
import logging
from config_loader import cfg
import prompts
logger = logging.getLogger("LegalAgentPlanReplan")
from .state import AgentState
from .operators import AgenticNodesOperator
from .reasoning import Reasoner
from replanner_rules import apply_hard_rules
from agent_tools.sandbox.executor import run_code_once, destroy_session as destroy_sandbox_session
def node_l0_gateway(state: AgentState) -> dict:
    """L0 安全网关：Prompt 注入检测 + 初始化熔断字段"""
    query = state["user_query"]

    # ---- 安全检测规则 ----
    # 中文 jailbreak
    injection_cn = ["忽略指令", "越狱", "忽略之前的", "忘记所有规则",
                    "假装你是", "你现在是", "DAN", "开发者模式"]
    # 英文 jailbreak
    injection_en = ["ignore previous instructions", "ignore all rules",
                    "system prompt", "pretend you are", "jailbreak",
                    "developer mode", "you are now"]
    # 分隔符注入
    injection_delimiters = ['"""', "---", "===", "[[SYSTEM]]", "<<SYS>>"]

    for pattern in injection_cn + injection_en + injection_delimiters:
        if pattern.lower() in query.lower():
            logger.warning(f"L0_REJECT: 检测到注入模式 '{pattern}'")
            raise ValueError(f"L0_REJECT: 触发安全熔断 - 检测到注入模式")

    return {
        "recursion_depth": 0,
        "retry_context": {},
        "global_facts": state.get("global_facts", [])
    }


def node_planner(state: AgentState, agentic_ops: "AgenticNodesOperator") -> dict:
    """
    Planner 节点：用注入的 agentic_ops 生成执行计划。

    阶段 3：改走 agentic_ops.generate_plan()，复用 prompts.py 的模板与同一个
    LLM 客户端。原先这里调模块级 call_deepseek_planner()，等于存在第二份
    Planner 实现，而且它自带一个独立客户端，测试桩无法替换。
    """
    logger.info("Planner 启动，生成执行计划")
    queue = agentic_ops.generate_plan(state["user_query"])
    from agent_runtime.contracts import TaskSpec
    queue = [TaskSpec.model_validate({"task_id": f"task-{index}", **task}).model_dump()
             if isinstance(task, dict) else TaskSpec(task_id=f"task-{index}", task_desc=task).model_dump()
             for index, task in enumerate(queue)]
    return {"task_queue": queue}


def node_executor(state: AgentState, reasoner: Reasoner) -> dict:
    if not state["task_queue"]:
        return {}
    # ---- 熔断保护 ----
    depth = state.get("recursion_depth", 0)
    max_depth = cfg.get("agent", "max_recursion_depth", default=5)
    if depth >= max_depth:
        logger.warning(f"🚨 触发算力熔断（depth={depth} > {max_depth}），强制终止")
        return {"retry_context": {"status": "force_stop"}, "execution_status": "budget_exhausted"}

    current_task = state["task_queue"][0]

    # ---- v2: 解析 Dict 任务格式 ----
    if isinstance(current_task, dict):
        actual_query = current_task.get("task_desc", str(current_task))
        engine = current_task.get("engine", "GRAPH_TRAVERSAL")
        rationale = current_task.get("rationale", "")
        is_wormhole = (engine == "GLOBAL_DENSE_WORMHOLE")
    else:
        # 兼容旧字符串格式
        actual_query = str(current_task)
        engine = "GRAPH_TRAVERSAL"
        rationale = ""
        is_wormhole = actual_query.startswith("[WORMHOLE]")
        if is_wormhole:
            actual_query = actual_query[len("[WORMHOLE]"):].strip()

    if is_wormhole:
        logger.info(f"🌌 虫洞穿越模式激活 (engine={engine}): {actual_query[:60]}")
    logger.info(f"Executor 正在处理: {actual_query[:80]}")

    # 使用 Reasoner 进行检索、推理、判断充足性
    causal_args = {}
    if cfg.get("causal_evidence", "enabled", default=False):
        causal_args = {"prediction_cutoff": state.get("prediction_cutoff"),
                       "causal_targets": state.get("causal_targets")}
    completed = state.get("completed_tasks", [])
    if isinstance(current_task, dict):
        done_ids = {task.get("task_id") for task in completed}
        if not set(current_task.get("deps", [])) <= done_ids:
            raise ValueError("Task dependencies have not completed")
    result = reasoner.answer(actual_query, original_query=state["user_query"],
                             engine_kind=engine, user_facts=state.get("user_facts"), **causal_args)
    if result.get("missing_info_kind") == "user" and not result["sufficient"]:
        return {"pending_question": result["suggestion"], "execution_status": "waiting_input"}

    # 构造观察记录（包含引擎信息）
    task_label = actual_query[:60]
    observation = (f"【任务：{task_label}】【引擎：{engine}】\n"
                   f"事实：{result['fact']}\n"
                   f"推理：{result['reasoning']}")
    if "causal_evidence" in result:
        observation += "\n因果证据与文本权重：\n" + result["causal_evidence"]["context"]

    new_queue = state["task_queue"][1:]  # 默认移除当前任务

    # ---- 新增：根据 grader_status 和 sufficient 联合决策 ----
    grader_status = result.get("grader_status", "irrelevant")
    sufficient = result.get("sufficient", False)

    if not sufficient:
        suggestion = result.get("suggestion", "需要更精准的法律检索")
        # v2: 补充任务使用 Dict 格式
        supplement_task = {
            **(current_task if isinstance(current_task, dict) else {}),
            "task_desc": f"{actual_query[:40]}（补充：{suggestion}）",
            "engine": engine,  # 保持原引擎
            "rationale": f"信息不足补搜: {suggestion[:60]}"
        }
        new_queue = [supplement_task] + new_queue
        observation += f"\n⚠️ 信息不足，已添加补充任务：{suggestion}"

        # 新增：设置 retry_context 供 Replanner 决策
        retry_ctx = {
            "status": grader_status,
            "missing_info": suggestion,
            "fail_count": state.get("retry_context", {}).get("fail_count", 0) + 1,
            "is_wormhole": is_wormhole
        }
    else:
        retry_ctx = {"status": "sufficient", "fail_count": 0}

    # ---- 新增：追加全局事实 (只追加，不覆盖) ----
    facts_to_add = []
    if isinstance(result['fact'], str) and result['fact'] != "未检索到相关文档":
        facts_to_add.append(result['fact'])
    elif isinstance(result['fact'], list):
        facts_to_add.extend(result['fact'])

    delta = {
        "task_queue": new_queue,
        "past_observations": [observation],
        "global_facts": state.get("global_facts", []) + facts_to_add,
        "retry_context": retry_ctx,
        "recursion_depth": depth + 1
    }
    if sufficient:
        delta["completed_tasks"] = completed + [current_task if isinstance(current_task, dict)
                                                else {"task_desc": actual_query}]
    if "causal_evidence" in result:
        delta["causal_evidence"] = [result["causal_evidence"]]
    return delta


def node_replanner(state: AgentState,
                   agentic_ops: "AgenticNodesOperator") -> dict:
    """v2: 支持 Pydantic 强类型任务格式 (task_desc + engine + rationale)"""
    obs_text = "\n".join(state.get("past_observations", []))
    queue = state["task_queue"]
    retry_ctx = state.get("retry_context", {})

    # 如果队列为空，结束
    if not queue:
        logger.info("所有任务完成，准备生成最终报告")
        return {}

    status = retry_ctx.get("status") or retry_ctx.get("grader_status")
    fail_count = retry_ctx.get("fail_count", 0)

    # 情况1：强制停止
    if status == "force_stop":
        logger.warning("算力熔断，强制终止")
        return {"task_queue": []}

    # 情况2：无头绪或部分缺失且重试超过阈值 -> LLM 虫洞重规划
    if status == "irrelevant" or (status == "partial" and fail_count > 2):
        logger.info("触发虫洞重规划引擎 (LLM)")

        current_failed = queue[0]
        failed_desc = (
            current_failed.get("task_desc", "")
            if isinstance(current_failed, dict)
            else str(current_failed)
        )
        replan_target = failed_desc if failed_desc else state["user_query"]

        # P1：改走注入的 agentic_ops.replan_with_wormhole()。
        #
        # 原先这里**直连 get_llm_client()**，后果有两条：
        #   1. 绕过了 `_call_messages` 这**唯一**的 LLM 接缝 —— 离线测试无法注入假 LLM，
        #      只能靠 monkeypatch 打补丁（tests/test_e2e_graph.py 曾这么干）；
        #   2. 它自成第二份「构造 prompt + 解析 JSON + 兜底」的实现，
        #      与 Replanner Worker 各写一遍 —— 阶段 3 刚把 prompt 统一到 prompts.py，
        #      解析逻辑这边又分叉了。
        new_tasks = agentic_ops.replan_with_wormhole(
            query=replan_target,
            global_facts=state.get("global_facts", []),
            fail_log=obs_text[-300:],
            fail_count=fail_count,
        )

        # 新任务替换队列头，保留队列尾部（其他未执行的子任务）
        original = queue[0] if isinstance(queue[0], dict) else {}
        for index, task in enumerate(new_tasks):
            task["task_id"] = (original.get("task_id", f"task-{state.get('recursion_depth', 0)}")
                               if index == len(new_tasks) - 1 else
                               f"replan-{state.get('recursion_depth', 0)}-{index}")
            task["deps"] = original.get("deps", []) if index == 0 else [new_tasks[index-1]["task_id"]]
        new_queue = new_tasks + queue[1:]
    else:
        # 情况3：硬规则补充
        # 规则表来自 replanner_rules.py，与 Replanner Worker 共用同一份。
        # 与 Worker 路径的差别：这里只做"追加"，不做 LLM 重规划。
        new_queue = list(queue)
        hard_tasks = apply_hard_rules(obs_text, queue + state.get("completed_tasks", []), logger)
        if hard_tasks:
            logger.info(f"硬规则补充 {len(hard_tasks)} 条任务")
            new_queue = hard_tasks + new_queue

    return {"task_queue": new_queue, "retry_context": {} if status == "sufficient" or
            status == "irrelevant" or (status == "partial" and fail_count > 2) else retry_ctx}


def node_generate(state: AgentState, agentic_ops: AgenticNodesOperator) -> dict:
    if state.get("calculation_mode") == "structured" and any(
            word in state["user_query"] for word in ("赔偿", "补偿", "加班费", "工资", "双倍", "2N", "N+1", "金额", "元")):
        return {"needs_sandbox_calc": True, "sandbox_retries": 0, "final_report": ""}
    logger.info("Generator 生成最终法律意见书")
    accumulated = []
    for idx, obs in enumerate(state.get("past_observations", [])):
        accumulated.append({
            "hop": idx + 1,
            "sub_task": state["task_queue"][idx] if idx < len(state["task_queue"]) else "总结",
            "reasoning": obs
        })
    from dataset.citations import requires_citations, generate_cited_answer
    analyses = state.get("causal_evidence", [])
    citations = None
    if requires_citations(analyses):
        cited = generate_cited_answer(prompts.build_generator_messages(state["user_query"], accumulated), analyses,
                                      lambda messages: agentic_ops._call_messages(messages, require_json=True))
        report, citations = cited["answer"], cited["citations"]
    else:
        report = agentic_ops.generate_final_report(state["user_query"], accumulated)

    # 检测是否需要金额计算 → 触发沙箱流程
    calc_keywords = ["赔偿", "补偿", "加班费", "工资", "双倍", "2N", "N+1", "金额", "元"]
    needs_calc = (state.get("calculation_mode") != "structured" and
                  any(kw in state["user_query"] + report for kw in calc_keywords))

    result = {"final_report": report}
    if citations is not None:
        result["citations"] = citations
    if needs_calc:
        result["needs_sandbox_calc"] = True
        result["sandbox_retries"] = 0
        logger.info("检测到金额计算需求，将进入沙箱执行流程")
    return result


def node_write_code(state: AgentState, agentic_ops: AgenticNodesOperator) -> dict:
    """生成 Python 计算代码"""
    logger.info("📝 沙箱阶段: 生成计算代码")
    if state.get("calculation_mode") == "structured":
        return {"calculation_plan": agentic_ops.generate_calculation_plan(
            state["user_query"], state.get("past_observations", []) + state.get("user_facts", []))}

    # 从 past_observations 提取推理链
    reasoning_chain = []
    obs_list = state.get("past_observations", [])
    for obs in obs_list:
        if isinstance(obs, dict):
            reasoning_chain.append({
                "sub_task": obs.get("task", ""),
                "facts": str(obs.get("extracted_facts", "")),
                "reasoning": str(obs.get("status", ""))
            })
        elif isinstance(obs, str):
            reasoning_chain.append({"sub_task": "", "facts": obs, "reasoning": ""})

    error_ctx = state.get("retry_context", {}).get("sandbox_error", "")
    code = agentic_ops.generate_calculation_code(
        state["user_query"], reasoning_chain, error_context=error_ctx
    )

    return {
        "generated_code": code,
        "sandbox_retries": state.get("sandbox_retries", 0)
    }


def node_execute_code(state: AgentState, runner=None) -> dict:
    """
    执行沙箱阶段生成的代码。

    阶段 5：Docker 优先 + 本进程降级的逻辑统一收敛到
    legal_sandbox/sandbox_exec.py，本节点只负责"按熔断策略决定要不要重试"。
    """
    logger.info("🐳 沙箱阶段: 执行计算代码")
    if state.get("calculation_mode") == "structured":
        from agent_tools.calculator import calculate
        value = calculate(state["calculation_plan"])
        return {"calc_result": f"{value['amount']}；公式：{value['formula']}；假设：{value['assumptions']}",
                "calculation_ok": True}

    code = state.get("generated_code", "")
    if not code:
        return {"retry_context": {"sandbox_error": "无代码可执行"}}

    outcome = (runner or run_code_once)(code, session_id=state.get("sandbox_session_id"))

    if outcome["error"]:
        logger.warning(f"沙箱执行报错（via={outcome['via']}）: {outcome['error']}")
        retries = state.get("sandbox_retries", 0) + 1
        max_retries = cfg.get("sandbox", "max_retries", default=3)
        if retries < max_retries:
            return {
                "retry_context": {"sandbox_error": outcome["error"]},
                "sandbox_retries": retries,
                "sandbox_session_id": outcome["session_id"],
            }
        # 重试耗尽：必须清掉 sandbox_error 并把计数钉在 max_retries。
        # 否则 route_after_execute 的判定条件（retry_context 里残留 sandbox_error
        # 且 sandbox_retries 未推进）会一直成立，导致 ExecuteCode↔WriteCode 死循环。
        return {
            "calc_result": f"沙箱多次执行失败: {outcome['error']}",
            "sandbox_session_id": outcome["session_id"],
            "retry_context": {},
            "sandbox_retries": max_retries,
            "calculation_ok": False,
            "execution_status": "calculation_failed",
        }

    logger.info(f"沙箱执行成功（via={outcome['via']}）: {str(outcome['calc_result'])[:100]}")
    return {
        "calc_result": outcome["calc_result"],
        "sandbox_session_id": outcome["session_id"],
        "calculation_ok": True,
    }


def node_inject_calc_result(state: AgentState, agentic_ops=None) -> dict:
    """将沙箱计算结果注入最终报告"""
    calc_result = state.get("calc_result", "")
    final_report = state.get("final_report", "")
    generated = {}
    if state.get("calculation_mode") == "structured":
        # Generate the report after arithmetic, with the checked result in context.
        report_state = {**state, "calculation_mode": "report_only",
                        "past_observations": state.get("past_observations", []) + ["程序核算结果：" + calc_result]}
        generated = node_generate(report_state, agentic_ops)
        final_report = generated["final_report"]

    if calc_result:
        if state.get("calculation_ok") is False:
            return {"final_report": f"{final_report}\n\n【计算未完成】\n{calc_result}"}
        enriched_report = (
            f"{final_report}\n\n"
            f"【计算明细】\n"
            f"经代码核算，最终结果为：{calc_result}"
        )
        logger.info("沙箱计算结果已注入最终报告")
        return {**generated, "final_report": enriched_report}

    return {}


def node_cleanup_sandbox(state: AgentState) -> dict:
    """清理沙箱资源（走 sandbox_exec 的统一入口，幂等）"""
    session_id = state.get("sandbox_session_id")
    if session_id:
        destroy_sandbox_session(session_id)
    return {"sandbox_session_id": None}


def node_await_input(state):
    from langgraph.types import interrupt
    answer = interrupt({"question": state["pending_question"]})
    if not isinstance(answer, str) or not answer.strip() or len(answer) > 20000:
        raise ValueError("Resume requires a nonempty answer of at most 20000 characters")
    return {"user_facts": state.get("user_facts", []) + [answer.strip()],
            "pending_question": None, "execution_status": "running"}
