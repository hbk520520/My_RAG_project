"""
Retriever Worker —— 第二棒：带着子任务去图引擎里找答案
===================================================
从 Redis 拿到当前会话的任务队列，取队头子任务，在图引擎里检索：
  - GRAPH_TRAVERSAL       : 向量召回 + 沿语义边扩展一跳（局部游走，成本低）
  - GLOBAL_DENSE_WORMHOLE : 只做全局稠密召回（跨领域，成本高）
把命中的文档写进 `past_observations[-1]["docs"]`，再交给 Grader 评判质量。

阶段 5 修复（两个都是"链路实际不工作"级别的问题）：
  1. 原先 `engine = LegalDenseGraphBuilder(alpha_dense=0.3)` 用了一个**从未 import**
     的名字，跑起来直接 NameError；而且 `retrieved_docs = []` 是硬编码空列表 ——
     整条链路的检索是空转的，Grader 永远拿到空资料、永远判 irrelevant。
     现在真正接上 `dataset.graph.LegalDenseGraphBuilder`：编码查询 → FAISS 召回 →
     图游走扩展 → 写回文档。
  2. 原先这里会把队头任务从 `task_queue` 移除，而 Grader 又用 `task_queue[0]`
     取"当前任务"，两者错位一格 —— Grader 评估的是「下一个任务」+「上一个任务的
     资料」。现在 Retriever **只检索、不动队列**，由 Grader 在判定 sufficient 后移除。

技术栈: Kafka / Redis / FAISS-HNSW / igraph / BGE-M3
"""
import sys, os, logging

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
    TOPIC_RETRIEVER_PENDING, TOPIC_GRADER_PENDING, TOPIC_REASONER_PENDING,
    GROUP_RETRIEVER, quarantine_message,
)
# 阶段 5：真正接上图引擎（原先这个名字根本没被 import）
# 阶段 10：一并引入顶点名边界转换器（图内顶点名恒为 str(int)）
from dataset.graph import LegalDenseGraphBuilder, vname, as_num_id

logging.basicConfig(
    level=getattr(logging, cfg.get("observability", "log_level", default="INFO")),
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger("RetrieverWorker")


# ============================================================================
# 图引擎装载
# ============================================================================
def build_engine() -> LegalDenseGraphBuilder:
    """
    构造图引擎，并尝试挂载检索器 LoRA（对齐 model/training/train_retriever.py 的产物）。

    LoRA 目录不存在或加载失败时会退回原始 BGE-M3 权重，不阻断 Worker 启动。
    """
    engine = LegalDenseGraphBuilder.from_config()
    lora_path = cfg.get("retriever", "lora_path")

    if not lora_path:
        logger.info("未配置 retriever.lora_path，使用原始 BGE-M3 权重")
    elif not os.path.isdir(lora_path):
        logger.warning(f"retriever.lora_path 不存在({lora_path})，使用原始 BGE-M3 权重")
    elif engine.load_lora_weights(lora_path):
        logger.info(f"检索器 LoRA 已加载并对齐训练产物: {lora_path}")

    return engine


# ============================================================================
# 检索
# ============================================================================
def _vector_recall(engine: LegalDenseGraphBuilder, query: str, k: int) -> list:
    """稠密向量召回，返回 [{id, content, score}]（已过滤 tombstone）"""
    enc = engine.encode_text(query)
    q_vec = enc["dense"].reshape(1, -1).astype("float32")

    k = int(min(max(k, 1), engine.index.ntotal))
    sims, ids = engine.index.search(q_vec, k)

    hits = []
    for score, nid in zip(sims[0], ids[0]):
        if nid == -1:
            continue
        try:
            v = engine.graph.vs.find(name=vname(nid))
        except ValueError:
            continue
        if v["metadata"].get("status") == "tombstone":
            continue
        hits.append({"id": int(nid), "content": v["content"], "score": float(score)})
    return hits


def _graph_expand(engine: LegalDenseGraphBuilder, seeds: list, limit: int) -> list:
    """沿语义边从种子节点扩展一跳，返回新增的 [{id, content, score}]"""
    # 阶段 10：`seen` 与返回的 id 必须是**数值 ID**（外部接口口径），
    # 而 igraph 顶点名是 str(int) —— 混用会让去重判断恒为 False，同一邻居被反复加入。
    seen = {int(h["id"]) for h in seeds}
    expanded = []
    for seed in seeds:
        try:
            v = engine.graph.vs.find(name=vname(seed["id"]))
        except ValueError:
            continue
        for nb in v.neighbors():
            nb_id = as_num_id(nb["name"])
            if nb_id in seen:
                continue
            if nb["metadata"].get("status") == "tombstone":
                continue
            seen.add(nb_id)
            # 邻居得分按一跳衰减，便于 Grader 分辨直接命中与扩展命中
            expanded.append({"id": nb_id, "content": nb["content"],
                             "score": seed["score"] * 0.9})
            if len(expanded) >= limit:
                return expanded
    return expanded


def retrieve_docs(engine: LegalDenseGraphBuilder, query: str,
                  engine_kind: str, top_k: int) -> list:
    """
    按引擎类型检索，返回命中的文档文本列表。

    GRAPH_TRAVERSAL       : 向量召回 top_k，再沿语义边各扩展一跳
    GLOBAL_DENSE_WORMHOLE : 只做全局稠密召回（语义边不参与，用于跨领域找依据）
    """
    if engine.index.ntotal == 0:
        logger.warning(
            "图引擎索引为空（FAISS 0 条），检索结果必然为空。"
            "请先注入语料：python dataset/prepare_corpus.py --corpus-dir <目录> "
            "--out dataset/corpus_out，再把产物交给 "
            "LegalDenseGraphBuilder.build_initial_graph_batch()。")
        return []

    try:
        hits = _vector_recall(engine, query, top_k)
    except Exception as e:
        logger.error(f"查询编码/召回失败，本次返回空结果: {e}")
        return []

    if not hits:
        logger.info(f"[{engine_kind}] 向量召回 0 条: {query[:50]}")
        return []

    if engine_kind == "GRAPH_TRAVERSAL":
        extra = _graph_expand(engine, hits, top_k)
        if extra:
            logger.info(f"[{engine_kind}] 向量召回 {len(hits)} 条，图游走扩展 {len(extra)} 条")
            hits.extend(extra)

    return [h["content"] for h in hits if h.get("content")]


# ============================================================================
# 主循环
# ============================================================================
def main():
    consumer = create_consumer(
        TOPIC_RETRIEVER_PENDING, GROUP_RETRIEVER,
        bootstrap_servers=cfg.get("kafka", "bootstrap_servers")
    )
    producer = create_producer(bootstrap_servers=cfg.get("kafka", "bootstrap_servers"))
    state_manager = StateManager(
        redis_url=cfg.get("redis", "url"),
        expire_seconds=cfg.get("redis", "state_expire_seconds")
    )

    top_k = cfg.get("retriever", "top_k", default=5)

    logger.info("正在装载图引擎与检索器权重…")
    engine = build_engine()
    logger.info(f"Retriever Worker started "
                f"(索引节点数={engine.index.ntotal}, top_k={top_k}, "
                f"lora_loaded={engine.is_lora_loaded})")

    for msg in consumer:
        session_id = msg.value.get("session_id") if isinstance(msg.value, dict) else None
        if not session_id:
            logger.error(f"消息缺少 session_id，跳过: {msg.value}")
            consumer.commit()
            continue

        try:
            state = state_manager.load_state(session_id)
        except KeyError:
            logger.error(f"Session {session_id} not found, skip")
            consumer.commit()
            continue

        # 阶段 6：单条消息的任何未处理异常都在这里被收容（写 DLQ + 提交位移 + 继续），
        # 不再让一条毒消息终结整个 Worker、也不让它被无限重投。
        try:
            task_queue = state.get("task_queue", [])
            if not task_queue:
                logger.info(f"Session {session_id}: 任务队列为空，直接转 Reasoner")
                producer.send(TOPIC_REASONER_PENDING, key=session_id,
                              value={"session_id": session_id})
                producer.flush()
                consumer.commit()
                continue

            current_task = task_queue[0]
            if isinstance(current_task, dict):
                actual_query = current_task.get("task_desc", str(current_task))
                engine_kind = current_task.get("engine", "GRAPH_TRAVERSAL")
            else:
                actual_query = str(current_task)
                engine_kind = "GRAPH_TRAVERSAL"
                # 兼容旧 [WORMHOLE] 前缀
                if actual_query.startswith("[WORMHOLE]"):
                    actual_query = actual_query[len("[WORMHOLE]"):].strip()
                    engine_kind = "GLOBAL_DENSE_WORMHOLE"

            logger.info(f"Retriever [{engine_kind}]: {actual_query[:60]}")
            docs = retrieve_docs(engine, actual_query, engine_kind, top_k)

            # 关键：只追加 observation，**不动 task_queue**。
            # Grader 用 task_queue[0] 取当前任务、past_observations[-1] 取资料，
            # 两者必须指向同一个任务；在这里 pop 会造成评估对象错位。
            state.setdefault("past_observations", []).append({
                "task": actual_query,
                "engine": engine_kind,
                "docs": docs,
                "retriever_status": "ok" if docs else "empty",
            })

            state_manager.save_state(session_id, state)
            producer.send(TOPIC_GRADER_PENDING, key=session_id,
                          value={"session_id": session_id})
            producer.flush()
            consumer.commit()
            logger.info(f"Session {session_id}: 检索到 {len(docs)} 条文档 -> Grader")

        except Exception as e:
            quarantine_message(msg, e, logger)
            try:
                consumer.commit()
            except Exception as ce:
                logger.error(f"隔离毒消息时提交位移失败，该消息可能被重复投递: {ce}")

    consumer.close()


if __name__ == "__main__":
    main()
