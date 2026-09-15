"""
Grader Worker —— 第三棒：翻完资料了，够不够用？
===========================================
拿到 Retriever 找回来的文档，调 Grader 模型判断能不能回答当前子任务：
够 → 转 Reasoner；不够但还能补 → 追加检索词回 Retriever；
完全无关 → 计数，超过阈值就喊 Replanner 来救场。

技术栈: Kafka / Redis / DeepSeek API (JSON mode) / prompts
"""
import sys, os, json, logging

# ---- 路径引导（同 planner_worker，见该文件说明）----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_ASYNC_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in (_ROOT_DIR, _ASYNC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config_loader import cfg
from state_manager import StateManager
from kafka_utils import (
    create_consumer, create_producer,
    TOPIC_GRADER_PENDING, TOPIC_REPLANNER_PENDING,
    TOPIC_REASONER_PENDING, TOPIC_RETRIEVER_PENDING,
    GROUP_GRADER, quarantine_message,
)

# 阶段 3：Prompt 统一从 prompts.py 导入，不再内联副本
from prompts import build_grader_messages

logging.basicConfig(
    level=getattr(logging, cfg.get("observability", "log_level", default="INFO")),
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger("GraderWorker")


def call_grader_llm(task_desc: str, docs: str) -> dict:
    """
    调用 Grader LLM 判断信息充分性。

    Prompt 正文来自 prompts.GRADER_SYSTEM，与 soul.py 的
    AgenticNodesOperator.grade_facts 共用同一份定义（阶段 3 起不再内联副本）。

    返回: {"status": "sufficient|partial|irrelevant",
           "extracted_facts": [...], "missing_info": "..."}
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=cfg.get("llm", "api_key"),
        base_url=cfg.get("llm", "base_url")
    )

    try:
        response = client.chat.completions.create(
            model=cfg.get("llm", "judge_model"),
            messages=build_grader_messages(task_desc, docs),
            response_format={"type": "json_object"},
            temperature=cfg.get("llm", "temperature_extract"),
            max_tokens=cfg.get("llm", "max_tokens_default")
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"Grader LLM 调用失败: {e}")
        return {"status": "irrelevant", "rationale": f"Grader 调用异常: {str(e)}"}


def main():
    consumer = create_consumer(
        TOPIC_GRADER_PENDING, GROUP_GRADER,
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

        # 阶段 6：单条消息的任何未处理异常都在这里收容（写 DLQ + 提交位移 + 继续），
        # 不再让一条毒消息终结整个 Worker、也不让它被无限重投。
        try:
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
                state["retry_context"] = retry_context
                state_manager.save_state(session_id, state)

                if retry_context["fail_count"] > cfg.get("agent", "max_retries_per_task"):
                    producer.send(TOPIC_REPLANNER_PENDING, key=session_id,
                                  value={"session_id": session_id})
                else:
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
                past_observations.append(observation)
                state["past_observations"] = past_observations
                state["retry_context"] = retry_context
                state_manager.save_state(session_id, state)

                wormhole_threshold = cfg.get("agent", "wormhole_threshold")
                if retry_context["fail_count"] > wormhole_threshold:
                    logger.info(f"Session {session_id}: 失败次数超过虫洞阈值，触发 Replanner")
                    producer.send(TOPIC_REPLANNER_PENDING, key=session_id,
                                  value={"session_id": session_id})
                else:
                    producer.send(TOPIC_RETRIEVER_PENDING, key=session_id,
                                  value={"session_id": session_id})

            producer.flush()
            consumer.commit()
            logger.info(f"Session {session_id}: Grader 完成，状态={status}")

        except Exception as e:
            quarantine_message(msg, e, logger)
            try:
                consumer.commit()
            except Exception as ce:
                logger.error(f"隔离毒消息时提交位移失败，该消息可能被重复投递: {ce}")

    consumer.close()


if __name__ == "__main__":
    main()
