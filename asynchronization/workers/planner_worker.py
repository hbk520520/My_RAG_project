"""
Planner Worker —— 第一棒：接到新问题，拆成调查步骤
===============================================
从 Kafka 拿到会话，调 Meta-Planner 生成双层蓝图（S_q DAG + C_q 具象化），
展平成按依赖拓扑排序的执行队列，写回 Redis 然后交给下一棒的 Retriever。

技术栈: Kafka / Redis / DeepSeek API / double_layer_plan / prompts
"""
import sys, os, json, logging

# ---- 路径引导 ----
# Worker 以脚本方式直接运行，sys.path[0] 是 workers/ 目录本身。
# 必须显式加入「项目根目录」与「asynchronization 层」，否则
# config_loader / double_layer_plan / prompts 都会导入失败，
# 进而让 kafka_utils 静默回落到环境变量、config.yaml 的 kafka 段形同虚设。
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
    TOPIC_PLANNER_PENDING, TOPIC_RETRIEVER_PENDING,
    GROUP_PLANNER, quarantine_message,
)

# 阶段 3：Prompt 与蓝图解析统一从公共模块导入，不再内联副本
from prompts import build_meta_planner_messages
from double_layer_plan import parse_double_layer_plan

logging.basicConfig(
    level=getattr(logging, cfg.get("observability", "log_level", default="INFO")),
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger("PlannerWorker")


def call_planner_llm(user_query: str) -> list:
    """
    调用 Meta-Planner LLM 生成双层蓝图 P_q={S_q,C_q}，展平为可执行任务队列。

    Prompt 正文来自 prompts.META_PLANNER_SYSTEM（阶段 3 起不再内联副本）。
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=cfg.get("llm", "api_key"),
        base_url=cfg.get("llm", "base_url")
    )

    try:
        response = client.chat.completions.create(
            model=cfg.get("llm", "judge_model"),
            messages=build_meta_planner_messages(user_query),
            response_format={"type": "json_object"},
            temperature=cfg.get("llm", "temperature_plan"),
            max_tokens=cfg.get("llm", "max_tokens_default")
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)

        # 解析双层蓝图 → 展平
        plan = parse_double_layer_plan(data)
        flat = plan.to_flat_task_queue(respect_deps=True)
        logger.info(f"双层蓝图: {len(plan.skeleton.nodes)}节点 DAG → {len(flat)}步拓扑队列")
        return flat

    except Exception as e:
        logger.error(f"Planner LLM 调用失败: {e}")
        return [{"task_desc": "核查劳动关系基础事实", "engine": "GRAPH_TRAVERSAL", "rationale": "异常降级"}]


def main():
    consumer = create_consumer(
        TOPIC_PLANNER_PENDING, GROUP_PLANNER,
        bootstrap_servers=cfg.get("kafka", "bootstrap_servers")
    )
    producer = create_producer(
        bootstrap_servers=cfg.get("kafka", "bootstrap_servers")
    )
    state_manager = StateManager(
        redis_url=cfg.get("redis", "url"),
        expire_seconds=cfg.get("redis", "state_expire_seconds")
    )

    logger.info("Planner Worker started, waiting for tasks...")

    for msg in consumer:
        session_id = msg.key or (msg.value.get("session_id") if isinstance(msg.value, dict) else None)
        if not session_id:
            logger.error("No session_id in message, skip")
            consumer.commit()
            continue

        try:
            state = state_manager.load_state(session_id)
        except KeyError:
            logger.error(f"Session {session_id} not found, skip")
            consumer.commit()
            continue

        # 阶段 6：单条消息的任何未处理异常都在这里收容（写 DLQ + 提交位移 + 继续），
        # 不再让一条毒消息终结整个 Worker、也不让它被无限重投。
        try:
            # 执行规划
            plan = call_planner_llm(state["user_query"])
            state["task_queue"] = plan
            state["current_step"] = "planner_done"
            state["recursion_depth"] = 0

            # 脱水保存
            state_manager.save_state(session_id, state)

            # 发送给 Retriever
            producer.send(TOPIC_RETRIEVER_PENDING, key=session_id,
                          value={"session_id": session_id})
            producer.flush()

            # 手动提交位移
            consumer.commit()
            logger.info(f"Session {session_id}: plan generated ({len(plan)} tasks)")

        except Exception as e:
            quarantine_message(msg, e, logger)
            try:
                consumer.commit()
            except Exception as ce:
                logger.error(f"隔离毒消息时提交位移失败，该消息可能被重复投递: {ce}")

    consumer.close()


if __name__ == "__main__":
    main()
