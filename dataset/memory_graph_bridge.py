"""
IncrementalMemoryManager ↔ LegalDenseGraphBuilder 桥接适配器
============================================================
将 GMM 动态阈值系统的增量知识注入到 igraph + FAISS 图引擎中，
实现两个子系统之间的数据互通。
"""
import sys
import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Callable

sys.path.insert(0, os.path.dirname(__file__))

# 阶段 10：图内顶点名恒为 str(int)，内部逻辑仍用 int ID —— 边界转换统一走这两个函数
from graph import vname, as_num_id

logger = logging.getLogger("MemoryGraphBridge")


class MemoryGraphBridge:
    """
    桥接器：连接 IncrementalMemoryManager（GMM 动态挂载）
    和 LegalDenseGraphBuilder（igraph + FAISS）
    """

    def __init__(self,
                 memory_manager,       # IncrementalMemoryManager 实例
                 graph_engine,         # LegalDenseGraphBuilder 实例
                 embedding_fn: Optional[Callable[[str], Any]] = None):
        """
        :param memory_manager: IncrementalMemoryManager 实例
        :param graph_engine:   LegalDenseGraphBuilder 实例（来自 graph.py 或 soul.py）
        :param embedding_fn:   文本 → 向量。**建议留空**（默认直接用引擎自己的编码器），
                               此时 GMM 与图引擎天然共用同一个向量。

        阶段 9 起支持三种返回形态：
          1. 不传（None）              → 走 `graph_engine.encode_text()`，dense + sparse 都有
          2. {"dense": …, "sparse": …} → 两个信号都拿到，与形态 1 等价
          3. np.ndarray                → 只有 dense；**没有稀疏信号**，混合相似度会退化为
                                         纯余弦（会打一条 WARNING，不再静默）
        """
        self.mm = memory_manager
        self.graph = graph_engine
        self.embed = embedding_fn        # None 表示"用引擎自己的编码器"
        self._pending_sync: List[Dict] = []  # 待同步到图引擎的节点
        self._warned_no_sparse = False

    # ------------------------------------------------------------------
    # 0. 统一的「文本 → (dense, sparse)」
    # ------------------------------------------------------------------
    def _normalize(self, dense: np.ndarray, content: str) -> np.ndarray:
        """维度校验 + L2 归一化。复用引擎的 `_check_dim`，保证与 encode_text 同一口径。"""
        self.graph._check_dim(dense, f"embedding_fn('{content[:20]}…')")
        norm = float(np.linalg.norm(dense))
        return dense / norm if norm > 0 else dense

    def _encode(self, content: str):
        """
        返回 `(dense, sparse)`。

        阶段 9 的关键点：**一次编码同时喂给两个子系统**。
        此前 `embedding_fn` 的结果只交给 GMM，图引擎那边由 `add_node()` 自己再编码
        一次 —— 两边向量不一致时也不报错（静默分叉）。现在统一走这里，并把结果
        透传给 `graph.add_node(dense=…, sparse=…)`，从根上消除"同一节点两个向量"。
        """
        if self.embed is None:
            enc = self.graph.encode_text(content)
            return enc["dense"], enc["sparse"]

        out = self.embed(content)

        # 形态：BGE-M3 原生输出 {"dense_vecs": …, "lexical_weights": …}
        if isinstance(out, dict) and "dense_vecs" in out:
            dense = np.asarray(out["dense_vecs"][0], dtype=np.float32)
            lexical = out.get("lexical_weights") or [{}]
            return self._normalize(dense, content), lexical[0]

        # 形态 2：{"dense": …, "sparse": …}
        if isinstance(out, dict):
            dense = np.asarray(out["dense"], dtype=np.float32)
            sparse = out.get("sparse")
            if not sparse and not self._warned_no_sparse:
                self._warned_no_sparse = True
                logger.warning(
                    "embedding_fn 未提供稀疏权重，混合相似度将退化为纯余弦"
                    "（如需完整信号，请把 embedding_fn 留空以使用引擎编码器）"
                )
            return self._normalize(dense, content), (sparse or {})

        # 形态 3：裸稠密向量
        if not self._warned_no_sparse:
            self._warned_no_sparse = True
            logger.warning(
                "embedding_fn 只返回稠密向量，缺少稀疏权重 —— 混合相似度将退化为纯余弦"
                "（如需完整信号，请把 embedding_fn 留空以使用引擎编码器）"
            )
        return self._normalize(np.asarray(out, dtype=np.float32), content), {}

    # ------------------------------------------------------------------
    # 1. 注入新知识（双写：GMM 挂载 + 图引擎入索引）
    # ------------------------------------------------------------------
    def inject_knowledge(self,
                         content: str,
                         node_type: str = "Raw",
                         metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        注入一条新知识，同时写入两个子系统：
        - IncrementalMemoryManager：GMM 动态阈值挂载
        - LegalDenseGraphBuilder：igraph + FAISS 连边
        返回新节点 ID
        """
        if metadata is None:
            metadata = {}

        # 阶段 9：一次编码，dense 与 sparse 同时喂给两个子系统
        vec, sparse = self._encode(content)

        # ---- 写入 GMM 记忆系统（用 dense）----
        gmm_node_id = self.mm.inject_new_knowledge(content, vec)

        # ---- 写入图引擎（透传同一个 dense/sparse，不再重新编码）----
        # 生成唯一 ID（优先使用图引擎的计数逻辑）
        graph_node_id = gmm_node_id  # 复用同一 ID
        try:
            self.graph.add_node(
                node_id=graph_node_id,
                content=content,
                node_type=node_type,
                metadata=metadata,
                dense=vec,
                sparse=sparse,
            )
            logger.info(f"双写成功: 节点 {graph_node_id} ('{content[:30]}...')")
        except Exception as e:
            logger.error(f"图引擎写入失败 (节点 {graph_node_id}): {e}")
            # 记录待同步。
            # 阶段 9：必须把向量一起暂存 —— 否则重试时 add_node 会自己重新编码，
            # 写进图引擎的就是另一个向量，GMM 与图引擎又分叉了（与本次修复同一个坑）。
            self._pending_sync.append({
                "node_id": graph_node_id,
                "content": content,
                "node_type": node_type,
                "metadata": metadata,
                "dense": vec,
                "sparse": sparse,
            })

        return graph_node_id

    # ------------------------------------------------------------------
    # 2. 从 MockGraphDB 同步脏摘要 → 图引擎 Summary 节点
    # ------------------------------------------------------------------
    def sync_dirty_summaries(self, summary_fn: Callable[[List[str]], str]):
        """
        将 IncrementalMemoryManager 中的脏摘要节点重算后，
        同步到 LegalDenseGraphBuilder 的对应 Summary 节点。

        :param summary_fn: 输入子节点文本列表，返回新摘要文本的函数（通常为 LLM 调用）
        """
        dirty_nodes = self.mm.db.get_dirty_nodes()
        if not dirty_nodes:
            logger.info("无脏节点需要同步")
            return

        for node_id in dirty_nodes:
            # 从 MockGraphDB 获取子节点文本
            children = self.mm.db.get_children(node_id)
            child_texts = [
                self.mm.get_node_text(c)
                for c in children
                if self.mm.get_node_text(c)
            ]
            if not child_texts:
                continue

            # 生成新摘要
            new_summary = summary_fn(child_texts)

            # 更新 MockGraphDB
            self.mm.db.nodes[node_id]["content"] = new_summary
            self.mm.db.clean_node(node_id)

            # 阶段 9：编码提到 try 之外 ——
            #   1. 只编一次，两个分支（更新已有顶点 / 新建顶点）共用同一份向量；
            #   2. 原先 encode_text 在 try 内部，它一旦抛 ValueError 会被
            #      `except ValueError` 误判成"顶点不存在"，进而创建重复节点。
            enc = self.graph.encode_text(new_summary)
            try:
                v = self.graph.graph.vs.find(name=vname(node_id))
                if v["type"] == "Summary":
                    v["content"] = new_summary
                    v["dense"] = enc["dense"]
                    v["sparse"] = enc["sparse"]
                    v["metadata"]["dirty"] = False
                    logger.info(f"Summary {node_id} 同步完成")
            except ValueError:
                # 图引擎中不存在，创建新的 Summary 节点
                new_id = self.graph.add_node(
                    node_id=node_id,
                    content=new_summary,
                    node_type="Summary",
                    metadata={"source": "memory_sync", "dirty": False},
                    dense=enc["dense"],
                    sparse=enc["sparse"],
                )
                logger.info(f"Summary {node_id} 在图引擎中新建 (ID={new_id})")

    # ------------------------------------------------------------------
    # 3. 从图引擎节点构建 MockGraphDB 结构（用于冷启动 GMM）
    # ------------------------------------------------------------------
    def bootstrap_from_graph(self):
        """
        如果 IncrementalMemoryManager 的 MockGraphDB 为空，
        从图引擎中读取已有 Summary 节点来初始化 GMM 的摘要索引。

        阶段 10 修复三处问题：
          1. `v["name"]` 现在是 `str(int)`，直接当 ID 用会让 GMM 侧的键变成字符串，
             而 `inject_new_knowledge` 一侧全是 int —— 两个子系统 ID 空间不一致；
          2. `child["metadata"]["parent_id"]` 存的是**数值 ID**，与字符串 `node_id`
             比较恒为 False，导致叶子关联循环**一次都没进过**；
          3. `MockGraphDB.add_node` 是自增计数，会另分配一个 ID，使
             `add_edge(父, 子)` 指向不存在的节点。现在显式指定 `node_id=`。
        """
        if self.mm.summary_embeddings:
            logger.info("GMM 摘要索引已有数据，跳过 bootstrap")
            return

        # 遍历图引擎中的 Summary 节点
        summary_count = 0
        for v in self.graph.graph.vs:
            if v["type"] == "Summary":
                node_id = as_num_id(v["name"])
                self.mm.db.add_node({
                    "type": "summary",
                    "content": v["content"],
                    "embedding": v["dense"],
                    "level": v["metadata"].get("level", 1)
                }, node_id=node_id)
                self.mm.summary_embeddings[node_id] = v["dense"]
                self.mm.set_node_text(node_id, v["content"])
                summary_count += 1

                # 添加叶子节点关联
                for child in self.graph.graph.vs:
                    if (child["metadata"].get("parent_id") == node_id
                            and child["type"] != "Summary"):
                        child_id = as_num_id(child["name"])
                        if child_id not in self.mm.db.nodes:
                            self.mm.db.add_node({
                                "type": "leaf",
                                "content": child["content"],
                                "embedding": child["dense"]
                            }, node_id=child_id)
                            self.mm.set_node_text(child_id, child["content"])
                        self.mm.db.add_edge(node_id, child_id, "CONTAINS")

        logger.info(f"从图引擎 bootstrap 完成: {summary_count} 个 Summary 节点")

    # ------------------------------------------------------------------
    # 4. 失效节点同步：tombstone 双向传播
    # ------------------------------------------------------------------
    def tombstone_knowledge(self, node_id: int) -> bool:
        """
        同时软删除两个子系统中的节点。
        图引擎中的 tombstone 会触发脏传播。
        """
        success = True

        # 图引擎 tombstone
        try:
            self.graph.tombstone_node(node_id)
        except Exception as e:
            logger.error(f"图引擎 tombstone 失败 (节点 {node_id}): {e}")
            success = False

        # GMM 系统标记失效
        if node_id in self.mm.db.nodes:
            self.mm.db.nodes[node_id]["status"] = "tombstone"
            # 触发向上脏传播
            parents = self.mm.db.get_parents(node_id)
            for p in parents:
                self.mm.db.mark_dirty(p)

        return success

    # ------------------------------------------------------------------
    # 5. 重试待同步节点
    # ------------------------------------------------------------------
    def retry_pending_sync(self):
        """重试之前因异常未成功写入图引擎的节点"""
        if not self._pending_sync:
            return

        logger.info(f"重试同步 {len(self._pending_sync)} 个挂起节点")
        still_pending = []
        for item in self._pending_sync:
            try:
                # 阶段 9：带上当初算好的向量，保证重试写入的与 GMM 侧是同一个
                self.graph.add_node(
                    node_id=item["node_id"],
                    content=item["content"],
                    node_type=item["node_type"],
                    metadata=item["metadata"],
                    dense=item.get("dense"),
                    sparse=item.get("sparse"),
                )
                logger.info(f"挂起节点 {item['node_id']} 重试成功")
            except Exception as e:
                logger.error(f"挂起节点 {item['node_id']} 重试仍失败: {e}")
                still_pending.append(item)

        self._pending_sync = still_pending

    # ------------------------------------------------------------------
    # 6. 获取全局统计信息
    # ------------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        """返回两个子系统的统计信息"""
        return {
            "graph_nodes": self.graph.graph.vcount(),
            "graph_edges": self.graph.graph.ecount(),
            "faiss_total": self.graph.index.ntotal,
            "gmm_summaries": len(self.mm.summary_embeddings),
            "gmm_nodes": len(self.mm.db.nodes),
            "gmm_dirty": len(self.mm.db.get_dirty_nodes()),
            "pending_sync": len(self._pending_sync),
        }


# ============================================================================
# 使用示例
# ============================================================================
if __name__ == "__main__":
    import numpy as _np
    import sys as _sys
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

    # 模拟初始化两个子系统
    from IncrementalMemoryManager import MockGraphDB, IncrementalMemoryManager

    db = MockGraphDB()
    gmm = IncrementalMemoryManager(db, {})

    # 从 graph.py 导入图引擎。
    # 阶段 8：改用 make_offline_engine()（内部注入 StubEncoder）。
    # 原先这里直接构造 then 在 add_node 里触发 encode_text → 懒加载 BGE-M3
    # → 无网环境下卡在 huggingface.co 重试 5 次然后超时，demo 根本跑不完。
    from graph import make_offline_engine
    engine = make_offline_engine()

    # 模拟嵌入函数（维度必须与引擎 embedding_dim 一致，否则 _check_dim 会拒绝）
    def mock_embed(text):
        v = _np.random.randn(engine.dim).astype(_np.float32)
        return v / _np.linalg.norm(v)

    # 构建桥接器。
    # 阶段 9：**推荐留空 embedding_fn** —— 此时直接用引擎自己的编码器，
    # GMM 与 igraph 拿到的必然是同一个向量（dense + sparse 都齐全）。
    bridge = MemoryGraphBridge(memory_manager=gmm, graph_engine=engine)

    # 注入新知识
    new_id = bridge.inject_knowledge(
        content="2025年最新司法解释：试用期辞退需支付赔偿金",
        node_type="Raw"
    )
    print(f"注入新知识: ID={new_id}")

    # 查看统计
    stats = bridge.get_stats()
    print(f"系统状态: {stats}")

    # 对照：如果一定要自己传 embedding_fn，且只返回裸向量，
    # 稀疏信号就没了（会打 WARNING），混合相似度退化为纯余弦。
    db2 = MockGraphDB()
    bridge2 = MemoryGraphBridge(
        memory_manager=IncrementalMemoryManager(db2, {}),
        graph_engine=make_offline_engine(),
        embedding_fn=mock_embed,          # 形态 3：只有 dense
    )
    print(f"自定义 embedding_fn 注入: ID={bridge2.inject_knowledge('试用期工资不得低于本单位相同岗位最低档工资的80%')}")
    print(f"自定义 embedding_fn 后系统状态: {bridge2.get_stats()}")
