"""
Reasoner Worker —— 基于检索事实进行逻辑推演并生成最终报告
对应 soul.py 中 AgenticNodesOperator.reason() + generate_final_report() 的角色
"""
import sys
import os
import json
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config_loader import cfg
from state_manager import StateManager
from kafka_utils import (
    create_consumer, create_producer,
    TOPIC_REASONER_PENDING
)

logging.basicConfig(
    level=getattr(logging, cfg.get("observability", "log_level", default="INFO")),
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger("ReasonerWorker")


def call_reasoner_llm(sub_task: str, extracted_facts: str) -> str:
    """
    调用 Reasoner LLM 对子任务进行逻辑推演。
    对应 soul.py 中 AgenticNodesOperator.reason().
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=cfg.get("llm", "api_key"),
        base_url=cfg.get("llm", "base_url")
    )

    system_prompt = """你是法官助理（Reasoner）。严格基于给定事实对子任务进行逻辑推演，不得引入外部知识。"""

    try:
        response = client.chat.completions.create(
            model=cfg.get("llm", "judge_model"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"子任务：{sub_task}\n事实：{extracted_facts}"}
            ],
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
    对应 soul.py 中 AgenticNodesOperator.generate_final_report().
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=cfg.get("llm", "api_key"),
        base_url=cfg.get("llm", "base_url")
    )

    system_prompt = """你是资深律师（Generator）。结合已验证的上下文证据链，生成专业、直接回答用户问题的法律意见书。标记计算出的金额。"""

    # 构造证据链文本
    context_str = ""
    for item in accumulated_context:
        hop = item.get("hop", "?")
        sub_task = item.get("sub_task", "")
        reasoning = item.get("reasoning", item.get("data", ""))
        context_str += f"[Hop {hop}] {sub_task} -> {reasoning}\n\n"

    try:
        response = client.chat.completions.create(
            model=cfg.get("llm", "judge_model"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"用户问题：{user_query}\n证据链：\n{context_str}"}
            ],
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


def main():
    consumer = create_consumer(
        TOPIC_REASONER_PENDING, "reasoner-group",
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
        if needs_calc:
            logger.info(f"Session {session_id}: 检测到金额计算需求，触发沙箱执行")
            state["needs_sandbox_calc"] = True

        # ---- 生成最终报告 ----
        final_report = call_generator_llm(user_query, reasoning_chain)

        # 更新状态
        state["reasoning_chain"] = reasoning_chain
        state["final_report"] = final_report
        state["status"] = "completed"
        state["task_queue"] = []  # 清空任务队列

        state_manager.save_state(session_id, state)

        logger.info(f"Session {session_id}: 推理完成，报告长度={len(final_report)}")
        consumer.commit()

    consumer.close()


if __name__ == "__main__":
    main()
