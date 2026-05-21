import sys
sys.path.append("../../")

from infrastructure.state_manager import StateManager
from infrastructure.kafka_utils import (
    create_consumer, create_producer,
    TOPIC_RETRIEVER_PENDING, TOPIC_GRADER_PENDING
)
from legal_graph_engine import LegalDenseGraphBuilder   # 你的图引擎

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RetrieverWorker")

def main():
    consumer = create_consumer(TOPIC_RETRIEVER_PENDING, "retriever-group")
    producer = create_producer()
    state_manager = StateManager()

    # 加载图谱引擎（使用 BGE‑M3，保持与原有一致）
    engine = LegalDenseGraphBuilder(alpha_dense=0.3)
    # 需要确保已加载数据，或动态建立索引（此处假设已有数据）

    logger.info("Retriever Worker started, waiting for tasks...")
    for msg in consumer:
        session_id = msg.value["session_id"]
        try:
            state = state_manager.load_state(session_id)
        except KeyError:
            logger.error(f"Session {session_id} not found, skip")
            consumer.commit()
            continue

        # 取当前待检索的子任务（v2: 支持 Dict 格式）
        if not state.get("task_queue"):
            producer.send(TOPIC_REASONER_PENDING, key=session_id, value={"session_id": session_id})
            consumer.commit()
            continue

        current_task = state["task_queue"][0]

        # ---- v2: 解析 Dict 任务 ----
        if isinstance(current_task, dict):
            actual_query = current_task.get("task_desc", str(current_task))
            engine = current_task.get("engine", "GRAPH_TRAVERSAL")
        else:
            actual_query = str(current_task)
            engine = "GRAPH_TRAVERSAL"
            # 兼容旧 [WORMHOLE] 前缀
            if actual_query.startswith("[WORMHOLE]"):
                actual_query = actual_query[len("[WORMHOLE]"):].strip()
                engine = "GLOBAL_DENSE_WORMHOLE"

        # 使用图引擎检索（这里简化，实际可复用 Reasoner 或直接调 search）
        # 实际中应调用 engine 的检索方法
        retrieved_docs = []
        logger.info(f"Retriever [{engine}]: {actual_query[:60]}")

        # 将检索结果写入状态
        state.setdefault("past_observations", []).append({
            "task": actual_query,
            "engine": engine,
            "docs": retrieved_docs
        })
        # 移除已完成任务（或由 Grader 决定）
        state["task_queue"] = state["task_queue"][1:]

        state_manager.save_state(session_id, state)

        # 发送给 Grader 评估
        producer.send(TOPIC_GRADER_PENDING, key=session_id, value={"session_id": session_id})
        producer.flush()
        consumer.commit()
        logger.info(f"Session {session_id}: retrieved for '{current_task}'")

if __name__ == "__main__":
    main()