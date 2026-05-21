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

logger = logging.getLogger("MemoryGraphBridge")


class MemoryGraphBridge:
    """
    桥接器：连接 IncrementalMemoryManager（GMM 动态挂载）
    和 LegalDenseGraphBuilder（igraph + FAISS）
    """

    def __init__(self,
                 memory_manager,       # IncrementalMemoryManager 实例
                 graph_engine,         # LegalDenseGraphBuilder 实例
                 embedding_fn: Callable[[str], np.ndarray]):
        """
        :param memory_manager: IncrementalMemoryManager 实例
        :param graph_engine:   LegalDenseGraphBuilder 实例（来自 graph.py 或 soul.py）
        :param embedding_fn:   文本 → 向量的编码函数
        """
        self.mm = memory_manager
        self.graph = graph_engine
        self.embed = embedding_fn
        self._pending_sync: List[Dict] = []  # 待同步到图引擎的节点

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

        vec = self.embed(content)

        # ---- 写入 GMM 记忆系统 ----
        gmm_node_id = self.mm.inject_new_knowledge(content, vec)

        # ---- 写入图引擎 ----
        # 生成唯一 ID（优先使用图引擎的计数逻辑）
        graph_node_id = gmm_node_id  # 复用同一 ID
        try:
            self.graph.add_node(
                node_id=graph_node_id,
                content=content,
                node_type=node_type,
                metadata=metadata
            )
            logger.info(f"双写成功: 节点 {graph_node_id} ('{content[:30]}...')")
        except Exception as e:
            logger.error(f"图引擎写入失败 (节点 {graph_node_id}): {e}")
            # 记录待同步
            self._pending_sync.append({
                "node_id": graph_node_id,
                "content": content,
                "node_type": node_type,
                "metadata": metadata
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

            # 同步到图引擎：查找同名 Summary 节点并更新
            try:
                v = self.graph.graph.vs.find(name=node_id)
                if v["type"] == "Summary":
                    enc = self.graph.encode_text(new_summary)
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
                    metadata={"source": "memory_sync", "dirty": False}
                )
                logger.info(f"Summary {node_id} 在图引擎中新建 (ID={new_id})")

    # ------------------------------------------------------------------
    # 3. 从图引擎节点构建 MockGraphDB 结构（用于冷启动 GMM）
    # ------------------------------------------------------------------
    def bootstrap_from_graph(self):
        """
        如果 IncrementalMemoryManager 的 MockGraphDB 为空，
        从图引擎中读取已有 Summary 节点来初始化 GMM 的摘要索引。
        """
        if self.mm.summary_embeddings:
            logger.info("GMM 摘要索引已有数据，跳过 bootstrap")
            return

        # 遍历图引擎中的 Summary 节点
        summary_count = 0
        for v in self.graph.graph.vs:
            if v["type"] == "Summary":
                node_id = v["name"]
                self.mm.db.add_node({
                    "type": "summary",
                    "content": v["content"],
                    "embedding": v["dense"],
                    "level": v["metadata"].get("level", 1)
                })
                self.mm.summary_embeddings[node_id] = v["dense"]
                self.mm.set_node_text(node_id, v["content"])
                summary_count += 1

                # 添加叶子节点关联
                for child in self.graph.graph.vs:
                    if (child["metadata"].get("parent_id") == node_id
                            and child["type"] != "Summary"):
                        child_id = child["name"]
                        if child_id not in self.mm.db.nodes:
                            self.mm.db.add_node({
                                "type": "leaf",
                                "content": child["content"],
                                "embedding": child["dense"]
                            })
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
                self.graph.add_node(
                    node_id=item["node_id"],
                    content=item["content"],
                    node_type=item["node_type"],
                    metadata=item["metadata"]
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
    # 模拟初始化两个子系统
    from IncrementalMemoryManager import MockGraphDB, IncrementalMemoryManager

    db = MockGraphDB()
    gmm = IncrementalMemoryManager(db, {})

    # 从 graph.py 导入图引擎
    from graph import LegalDenseGraphBuilder
    engine = LegalDenseGraphBuilder(alpha_dense=0.3)

    # 模拟嵌入函数
    def mock_embed(text):
        v = np.random.randn(1024).astype(np.float32)
        return v / np.linalg.norm(v)

    # 构建桥接器
    bridge = MemoryGraphBridge(
        memory_manager=gmm,
        graph_engine=engine,
        embedding_fn=mock_embed
    )

    # 注入新知识
    new_id = bridge.inject_knowledge(
        content="2025年最新司法解释：试用期辞退需支付赔偿金",
        node_type="Raw"
    )
    print(f"注入新知识: ID={new_id}")

    # 查看统计
    stats = bridge.get_stats()
    print(f"系统状态: {stats}")
