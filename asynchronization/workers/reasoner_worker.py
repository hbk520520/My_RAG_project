"""
Reasoner Worker —— 最后一棒：把收集到的事实串成法律意见
=====================================================
前面几棒把事实都查齐了，这里用 Reasoner 模型对每个子任务做逻辑推演，
然后 Generator 综合所有推理链写出最终法律意见书。
如果涉及金额计算，标记 needs_sandbox_calc 让下一环节走沙箱。

技术栈: Kafka / Redis / DeepSeek API / prompts
"""
import sys, os, json, logging

# ---- 路径引导（同 planner_worker，见该文件说明）----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_ASYNC_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
# 阶段 5：本 Worker 要直接调沙箱，需要 legal_sandbox 目录可见
_SANDBOX_DIR = os.path.join(_ROOT_DIR, "multiple-search", "legal_sandbox")
for _p in (_ROOT_DIR, _ASYNC_DIR, _SANDBOX_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config_loader import cfg
from state_manager import StateManager
from kafka_utils import (
    create_consumer, create_producer,
    TOPIC_REASONER_PENDING,
    GROUP_REASONER, quarantine_message,
)

# 阶段 3：Prompt 统一从 prompts.py 导入，不再内联副本
from prompts import (build_reasoner_messages, build_generator_messages,
                     build_code_generator_messages)
# 阶段 5：沙箱执行统一入口（与 soul.py 共用同一份 Docker/降级逻辑）
from sandbox_exec import run_code_once, destroy_session as destroy_sandbox_session

logging.basicConfig(
    level=getattr(logging, cfg.get("observability", "log_level", default="INFO")),
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger("ReasonerWorker")


def call_reasoner_llm(sub_task: str, extracted_facts: str) -> str:
    """
    调用 Reasoner LLM 对子任务进行逻辑推演。

    对应 soul.py 中 AgenticNodesOperator.reason()，
    共用 prompts.REASONER_SYSTEM。
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=cfg.get("llm", "api_key"),
        base_url=cfg.get("llm", "base_url")
    )

    try:
        response = client.chat.completions.create(
            model=cfg.get("llm", "judge_model"),
            messages=build_reasoner_messages(sub_task, extracted_facts),
            temperature=cfg.get("llm", "temperature_reason"),
            max_tokens=cfg.get("llm", "max_tokens_default")
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Reasoner LLM 调用失败: {e}")
        return f"推理异常: {str(e)}"


def call_generator_llm(user_query: str, accumulated_context: list) -> str:
    """
    调用 Generator LLM 生成最终法律意见书。

    对应 soul.py 中 AgenticNodesOperator.generate_final_report()，
    共用 prompts.GENERATOR_SYSTEM 与 prompts.format_evidence_chain 的拼法。
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=cfg.get("llm", "api_key"),
        base_url=cfg.get("llm", "base_url")
    )

    try:
        response = client.chat.completions.create(
            model=cfg.get("llm", "judge_model"),
            messages=build_generator_messages(user_query, accumulated_context),
            temperature=cfg.get("llm", "temperature_generate"),
            max_tokens=cfg.get("llm", "max_tokens_default")
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Generator LLM 调用失败: {e}")
        return f"报告生成异常: {str(e)}"


def detect_calculation_needed(user_query: str, reasoning_chain: list) -> bool:
    """检测是否需要金额计算（赔偿金/补偿金/加班费等）"""
    calc_keywords = ["赔偿", "补偿", "加班费", "工资", "双倍", "2N", "N+1",
                     "金额", "元", "计算", "赔", "罚金", "滞纳金"]
    combined_text = user_query + " ".join(
        r.get("reasoning", "") for r in reasoning_chain
    )
    return any(kw in combined_text for kw in calc_keywords)


def call_code_generator_llm(user_query: str, reasoning_chain: list,
                            error_context: str = "") -> str:
    """
    生成金额计算代码。

    对应 soul.py 中 AgenticNodesOperator.generate_calculation_code()，
    共用 prompts.CODE_GENERATOR_SYSTEM；error_context 非空时构造器会自动
    追加"上次执行报错，请修正"段落。
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=cfg.get("llm", "api_key"),
        base_url=cfg.get("llm", "base_url")
    )

    try:
        response = client.chat.completions.create(
            model=cfg.get("llm", "judge_model"),
            messages=build_code_generator_messages(user_query, reasoning_chain,
                                                   error_context),
            temperature=cfg.get("llm", "temperature_generate", default=0.1),
            max_tokens=cfg.get("llm", "max_tokens_default")
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"代码生成 LLM 调用失败: {e}")
        return ""


def run_calc_with_sandbox(user_query: str, reasoning_chain: list,
                          state: dict):
    """
    生成代码 → 沙箱执行 → 失败则带上错误上下文重试。

    阶段 5：Reasoner Worker 原先只设置 needs_sandbox_calc=True 却**没有任何消费方**，
    金额计算环节在 Kafka 链路里是断的。现在直接内联调用沙箱，
    复用 sandbox_exec.run_code_once（Docker 优先，不可用时降级本进程）。

    :return: (计算明细文本, 沙箱 session_id)
    """
    max_retries = cfg.get("sandbox", "max_retries", default=3)
    session_id = state.get("sandbox_session_id")
    error_context = ""

    for attempt in range(1, max_retries + 1):
        code = call_code_generator_llm(user_query, reasoning_chain, error_context)
        if not code:
            return "代码生成失败，本次未做金额核算。", session_id

        outcome = run_code_once(code, session_id=session_id)
        session_id = outcome.get("session_id") or session_id

        if not outcome["error"]:
            logger.info(f"沙箱执行成功（via={outcome['via']}，第 {attempt} 次尝试）")
            return str(outcome["calc_result"]), session_id

        logger.warning(f"沙箱第 {attempt}/{max_retries} 次执行失败: "
                       f"{str(outcome['error'])[:200]}")
        error_context = outcome["error"]

    return f"沙箱执行 {max_retries} 次均失败，未能给出核算结果。", session_id


def main():
    consumer = create_consumer(
        TOPIC_REASONER_PENDING, GROUP_REASONER,
        bootstrap_servers=cfg.get("kafka", "bootstrap_servers")
    )
    state_manager = StateManager(
        redis_url=cfg.get("redis", "url"),
        expire_seconds=cfg.get("redis", "state_expire_seconds")
    )

    logger.info("Reasoner Worker started, waiting for tasks...")

    for msg in consumer:
        session_id = msg.key or msg.value.get("session_id")
        try:
            state = state_manager.load_state(session_id)
        except KeyError:
            logger.error(f"Session {session_id} not found, skip")
            consumer.commit()
            continue

        # 阶段 6：单条消息的任何未处理异常都在这里收容（写 DLQ + 提交位移 + 继续）
        try:
            user_query = state.get("user_query", "")
            past_observations = state.get("past_observations", [])
            task_queue = state.get("task_queue", [])
            retry_context = state.get("retry_context", {})

            # 强制停止 → 直接生成最终报告
            if retry_context.get("status") == "force_stop":
                logger.info(f"Session {session_id}: 熔断触发，生成降级报告")

            # ---- 逐步推理每个子任务 ----
            reasoning_chain = []
            for idx, obs in enumerate(past_observations):
                if isinstance(obs, dict):
                    sub_task = obs.get("task", f"步骤{idx+1}")
                    facts = obs.get("extracted_facts", [])
                    facts_text = "\n".join(facts) if isinstance(facts, list) else str(facts)

                    if facts_text:
                        reasoning = call_reasoner_llm(sub_task, facts_text)
                    else:
                        reasoning = "无有效事实，跳过推理"

                    reasoning_chain.append({
                        "hop": idx + 1,
                        "sub_task": sub_task,
                        "reasoning": reasoning,
                        "facts": facts_text
                    })
                    logger.info(f"Session {session_id}: Hop {idx+1} 推理完成")

            # ---- 检测是否需要代码计算 ----
            needs_calc = detect_calculation_needed(user_query, reasoning_chain)
            state["needs_sandbox_calc"] = needs_calc
            calc_detail = ""

            if needs_calc:
                # 阶段 5：原先这里只设 needs_sandbox_calc=True，Kafka 链路里没有任何
                # 消费方 —— 金额计算环节实际是断的。现在直接内联调用沙箱。
                logger.info(f"Session {session_id}: 检测到金额计算需求，进入沙箱核算")
                calc_detail, sandbox_session = run_calc_with_sandbox(
                    user_query, reasoning_chain, state)
                state["sandbox_session_id"] = sandbox_session
                logger.info(f"Session {session_id}: 核算明细 = {calc_detail[:120]}")

            # ---- 生成最终报告 ----
            final_report = call_generator_llm(user_query, reasoning_chain)
            if calc_detail:
                final_report = f"{final_report}\n\n【计算明细】\n{calc_detail}"

            # ---- 清理沙箱资源（幂等，清理失败不影响主流程）----
            if state.get("sandbox_session_id"):
                destroy_sandbox_session(state["sandbox_session_id"])
                state["sandbox_session_id"] = None

            # 更新状态
            state["reasoning_chain"] = reasoning_chain
            state["final_report"] = final_report
            state["status"] = "completed"
            state["task_queue"] = []  # 清空任务队列

            state_manager.save_state(session_id, state)

            logger.info(f"Session {session_id}: 推理完成，报告长度={len(final_report)}")
            consumer.commit()

        except Exception as e:
            quarantine_message(msg, e, logger)
            try:
                consumer.commit()
            except Exception as ce:
                logger.error(f"隔离毒消息时提交位移失败，该消息可能被重复投递: {ce}")

    consumer.close()


if __name__ == "__main__":
    main()
