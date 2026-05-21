"""
Grader Worker —— 评估检索结果的信息充分性
对应 soul.py 中 AgenticNodesOperator.grade_facts() 的角色
"""
import sys
import os
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config_loader import cfg
from state_manager import StateManager
from kafka_utils import (
    create_consumer, create_producer,
    TOPIC_GRADER_PENDING, TOPIC_REPLANNER_PENDING, TOPIC_REASONER_PENDING
)

logging.basicConfig(
    level=getattr(logging, cfg.get("observability", "log_level", default="INFO")),
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger("GraderWorker")


def call_grader_llm(task_desc: str, docs: str) -> dict:
    """
    调用 Grader LLM 判断信息充分性。
    直接复用 soul.py 中 AgenticNodesOperator.grade_facts 的 system prompt。
    返回: {"status": "sufficient|partial|irrelevant", "extracted_facts": [...], "missing_info": "..."}
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=cfg.get("llm", "api_key"),
        base_url=cfg.get("llm", "base_url")
    )

    system_prompt = """你是一个极其严苛的事实调查官。不要推理，只对比资料与任务。
输出JSON：
{
  "rationale": "判决说理",
  "status": "sufficient | partial | irrelevant",
  "extracted_facts": ["事实1"] (仅在 sufficient/partial 时输出),
  "missing_info": "缺少的搜索词" (仅在 partial 时输出)
}"""

    try:
        response = client.chat.completions.create(
            model=cfg.get("llm", "judge_model"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"任务：{task_desc}\n资料：{docs}"}
            ],
            response_format={"type": "json_object"},
            temperature=cfg.get("llm", "temperature_extract"),
            max_tokens=cfg.get("llm", "max_tokens_default")
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Grader LLM 调用失败: {e}")
        return {"status": "irrelevant", "rationale": f"Grader 调用异常: {str(e)}"}


import json


def main():
    consumer = create_consumer(
        TOPIC_GRADER_PENDING, "grader-group",
        bootstrap_servers=cfg.get("kafka", "bootstrap_servers")
    )
    producer = create_producer(
        bootstrap_servers=cfg.get("kafka", "bootstrap_servers")
    )
    state_manager = StateManager(
        redis_url=cfg.get("redis", "url"),
        expire_seconds=cfg.get("redis", "state_expire_seconds")
    )

    logger.info("Grader Worker started, waiting for tasks...")

    for msg in consumer:
        session_id = msg.key or msg.value.get("session_id")
        try:
            state = state_manager.load_state(session_id)
        except KeyError:
            logger.error(f"Session {session_id} not found, skip")
            consumer.commit()
            continue

        # 获取当前任务和检索结果（v2: 支持 Dict 任务）
        task_queue = state.get("task_queue", [])
        past_observations = state.get("past_observations", [])

        if not task_queue:
            logger.info(f"Session {session_id}: 无待评估任务，直接转发至 Reasoner")
            producer.send(TOPIC_REASONER_PENDING, key=session_id,
                          value={"session_id": session_id})
            producer.flush()
            consumer.commit()
            continue

        current_task = task_queue[0]
        # 提取实际查询文本和引擎
        if isinstance(current_task, dict):
            task_desc = current_task.get("task_desc", str(current_task))
            engine = current_task.get("engine", "GRAPH_TRAVERSAL")
        else:
            task_desc = str(current_task)
            engine = "GRAPH_TRAVERSAL"

        # 获取最近一次检索的文档
        last_obs = past_observations[-1] if past_observations else {}
        docs = last_obs.get("docs", [])
        docs_text = "\n---\n".join(docs) if isinstance(docs, list) else str(docs)

        # 调用 Grader 评估
        grade_result = call_grader_llm(task_desc, docs_text)
        status = grade_result.get("status", "irrelevant")
        extracted_facts = grade_result.get("extracted_facts", [])
        missing_info = grade_result.get("missing_info", "")

        logger.info(f"Session {session_id}: Grader 评估 [{task_desc}] -> {status}")

        # 更新状态
        retry_context = state.get("retry_context", {})
        retry_context["grader_status"] = status
        retry_context["fail_count"] = retry_context.get("fail_count", 0)

        if status == "sufficient":
            observation = {
                "task": task_desc,
                "engine": engine,
                "docs": docs,
                "extracted_facts": extracted_facts,
                "status": "sufficient"
            }
            past_observations.append(observation)
            task_queue = task_queue[1:]
            state["task_queue"] = task_queue
            state["past_observations"] = past_observations
            retry_context["fail_count"] = 0

            state_manager.save_state(session_id, state)
            producer.send(TOPIC_REASONER_PENDING, key=session_id,
                          value={"session_id": session_id})

        elif status == "partial":
            retry_context["fail_count"] += 1
            observation = {
                "task": task_desc,
                "engine": engine,
                "docs": docs,
                "extracted_facts": extracted_facts,
                "missing_info": missing_info,
                "status": "partial"
            }
            past_observations.append(observation)

            if missing_info:
                # v2: 补充任务使用 Dict 格式
                supplement = {
                    "task_desc": f"{task_desc}（补充：{missing_info}）",
                    "engine": engine,
                    "rationale": f"信息不全补搜: {missing_info[:60]}"
                }
                task_queue = [supplement] + task_queue[1:]
            else:
                task_queue = task_queue[1:]
            state["task_queue"] = task_queue
            state["past_observations"] = past_observations

            if retry_context["fail_count"] > cfg.get("agent", "max_retries_per_task"):
                state["retry_context"] = retry_context
                state_manager.save_state(session_id, state)
                producer.send(TOPIC_REPLANNER_PENDING, key=session_id,
                              value={"session_id": session_id})
            else:
                state["retry_context"] = retry_context
                state_manager.save_state(session_id, state)
                from kafka_utils import TOPIC_RETRIEVER_PENDING
                producer.send(TOPIC_RETRIEVER_PENDING, key=session_id,
                              value={"session_id": session_id})

        else:  # irrelevant
            retry_context["fail_count"] += 1
            observation = {
                "task": task_desc,
                "engine": engine,
                "docs": docs,
                "extracted_facts": [],
                "status": "irrelevant"
            }
                "extracted_facts": [],
                "status": "irrelevant"
            }
            past_observations.append(observation)
            state["past_observations"] = past_observations

            wormhole_threshold = cfg.get("agent", "wormhole_threshold")
            if retry_context["fail_count"] > wormhole_threshold:
                logger.info(f"Session {session_id}: 失败次数超过虫洞阈值，触发 Replanner")
                state["retry_context"] = retry_context
                state_manager.save_state(session_id, state)
                producer.send(TOPIC_REPLANNER_PENDING, key=session_id,
                              value={"session_id": session_id})
            else:
                state["retry_context"] = retry_context
                state_manager.save_state(session_id, state)
                from kafka_utils import TOPIC_RETRIEVER_PENDING
                producer.send(TOPIC_RETRIEVER_PENDING, key=session_id,
                              value={"session_id": session_id})

        producer.flush()
        consumer.commit()
        logger.info(f"Session {session_id}: Grader 完成，状态={status}")

    consumer.close()


if __name__ == "__main__":
    main()
