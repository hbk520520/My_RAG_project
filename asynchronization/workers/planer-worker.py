import sys
import logging
sys.path.append("../../")   # 确保能找到 legal_graph_engine

from infrastructure.state_manager import StateManager
from infrastructure.kafka_utils import (
    create_consumer, create_producer,
    TOPIC_PLANNER_PENDING, TOPIC_RETRIEVER_PENDING
)
from agentic_operators import AgenticNodesOperator   # 你的 LLM 算子

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PlannerWorker")

def main():
    consumer = create_consumer(TOPIC_PLANNER_PENDING, "planner-group")
    producer = create_producer()
    state_manager = StateManager()

    # 加载模型（只加载一次，全局复用）
    agentic = AgenticNodesOperator(api_key="your-key")  # 只使用其 generate_plan

    logger.info("Planner Worker started, waiting for tasks...")
    for msg in consumer:
        session_id = msg.key
        try:
            state = state_manager.load_state(session_id)
        except KeyError:
            logger.error(f"Session {session_id} not found, skip")
            consumer.commit()
            continue

        # 执行规划
        plan = agentic.generate_plan(state["user_query"])
        state["task_queue"] = plan
        state["current_step"] = "planner_done"

        # 脱水保存
        state_manager.save_state(session_id, state)

        # 发送给 Retriever
        producer.send(TOPIC_RETRIEVER_PENDING, key=session_id, value={"session_id": session_id})
        producer.flush()

        # 手动提交位移
        consumer.commit()
        logger.info(f"Session {session_id}: plan generated")

if __name__ == "__main__":
    main()