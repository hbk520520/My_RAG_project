"""
增量记忆注入引擎 —— 新知识来了不用全量重建
========================================
用一维高斯混合模型 (GMM) 动态决定新知识该挂到哪些簇上，
挂载后向上级联标记脏摘要，等夜间无人时统一重算。

核心思路：
  仿生学记忆整合 —— 新信息不需要重新整理整个记忆库，
  找到最相关的那几个簇挂上去就行，被影响的上层摘要标记"需要重写"。

技术栈: sklearn (GaussianMixture) / numpy / 模拟图数据库
"""
import numpy as np
from sklearn.mixture import GaussianMixture
from typing import List, Tuple, Dict, Set
import logging

logger = logging.getLogger("MemorySniffer")
logging.basicConfig(level=logging.INFO)

# ============================================================================
# 模拟图数据库 (实际环境替换为 Neo4j / Milvus 操作)
# ============================================================================
class MockGraphDB:
    """模拟一个树状结构的图存储，支持节点、边、属性"""
    def __init__(self):
        self.nodes: Dict[int, dict] = {}
        self.edges: List[Tuple[int, int, str]] = []  # (src, dst, type)
        self._id_counter = 0

    def add_node(self, props: dict, node_id: int = None) -> int:
        """
        新增节点，返回其 ID（**int**，与图引擎的顶点 ID 同一空间）。

        :param node_id: 显式指定 ID（阶段 10 新增）。默认 None 表示自增分配。
            桥接器从图引擎 bootstrap 时**必须**指定：否则自增出来的 ID 与
            图引擎的顶点 ID 不同，`add_edge(父, 子)` 会指向不存在的节点、
            `set_node_text(id)` 也会挂在错误的键上。
        """
        if node_id is not None:
            node_id = int(node_id)
            self._id_counter = max(self._id_counter, node_id)
            self.nodes[node_id] = {**props, "dirty": False}
            return node_id
        self._id_counter += 1
        self.nodes[self._id_counter] = {**props, "dirty": False}
        return self._id_counter

    def add_edge(self, src: int, dst: int, rel_type: str):
        self.edges.append((src, dst, rel_type))

    def get_children(self, parent_id: int) -> List[int]:
        return [dst for src, dst, t in self.edges if src == parent_id and t == "CONTAINS"]

    def get_parents(self, child_id: int) -> List[int]:
        return [src for src, dst, t in self.edges if dst == child_id and t == "CONTAINS"]

    def mark_dirty(self, node_id: int):
        if node_id in self.nodes:
            self.nodes[node_id]["dirty"] = True

    def is_dirty(self, node_id: int) -> bool:
        return self.nodes.get(node_id, {}).get("dirty", False)

    def clean_node(self, node_id: int):
        if node_id in self.nodes:
            self.nodes[node_id]["dirty"] = False

    def get_dirty_nodes(self) -> List[int]:
        return [nid for nid, props in self.nodes.items() if props.get("dirty")]


# ============================================================================
# GMM 动态阈值核心
# ============================================================================
class DynamicThresholder:
    """使用一维高斯混合模型将相似度分为 Accept/Reject 两类"""

    # 少于此样本数时 GMM 不可靠（n_components=2，样本太少会直接抛异常）
    MIN_SAMPLES = 4

    @staticmethod
    def split(similarity_scores: np.ndarray) -> Tuple[List[int], float]:
        """
        输入：新节点与所有候选簇的余弦相似度数组
        返回：
            accept_indices: 被接受挂载的簇索引列表
            dynamic_threshold: 动态决策边界 (Accept 分布的最小分数)

        阶段 6 兜底（原先没有，任一情况都会让整条增量注入流程抛异常）：
          1. 样本数 < MIN_SAMPLES   -> GMM 无法拟合两个分量
          2. 所有得分几乎相同       -> 两个高斯分量没有意义
          3. GMM 拟合本身失败       -> 数值问题
          4. 拟合结果为空簇         -> 退化为兜底策略
        """
        scores = np.asarray(similarity_scores, dtype=np.float64).ravel()
        if scores.size == 0:
            return [], 0.0

        # 兜底 1：样本太少
        if scores.size < DynamicThresholder.MIN_SAMPLES:
            logger.info(f"相似度样本仅 {scores.size} 条（< {DynamicThresholder.MIN_SAMPLES}），"
                        f"GMM 不适用，改用中位数阈值")
            return DynamicThresholder._fallback_split(scores)

        # 兜底 2：得分几乎无差异
        if float(scores.max() - scores.min()) < 1e-6:
            logger.info("相似度得分几乎相同，GMM 无区分意义，改用中位数阈值")
            return DynamicThresholder._fallback_split(scores)

        # 正常路径：GMM 动态分界
        try:
            X = scores.reshape(-1, 1)
            gmm = GaussianMixture(n_components=2, covariance_type='spherical',
                                  random_state=42, reg_covar=1e-5)
            labels = gmm.fit_predict(X)
        except Exception as e:
            logger.warning(f"GMM 拟合失败({e})，改用中位数阈值")
            return DynamicThresholder._fallback_split(scores)

        means = gmm.means_.flatten()
        accept_label = int(np.argmax(means))
        accept_indices = np.where(labels == accept_label)[0].tolist()
        if not accept_indices:
            logger.warning("GMM 未分出 accept 簇，改用中位数阈值")
            return DynamicThresholder._fallback_split(scores)

        return accept_indices, float(scores[accept_indices].min())

    @staticmethod
    def _fallback_split(scores: np.ndarray) -> Tuple[List[int], float]:
        """稳健兜底：中位数及以上视为 Accept"""
        threshold = float(np.median(scores))
        accept_indices = np.where(scores >= threshold)[0].tolist()
        return accept_indices, threshold


# ============================================================================
# 记忆嗅探微服务主类
# ============================================================================
class IncrementalMemoryManager:
    """
    增量知识挂载引擎：
    - 基于聚类摘要向量的局部相似度计算
    - GMM 自适应挂载
    - 脏标记级联传播
    - 孤儿节点新簇创建
    - 夜间清算接口
    """

    ABSOLUTE_FLOOR_THRESHOLD = 0.4  # 绝对保底阈值，防止误挂载

    def __init__(self, graph_db: MockGraphDB, summary_embeddings: Dict[int, np.ndarray],
                 floor_threshold: float = None):
        """
        :param graph_db: 图数据库实例 (需实现 children/parents/mark_dirty 等方法)
        :param summary_embeddings: {cluster_id: summary_vector} 的映射，仅包含底层摘要层
        :param floor_threshold: 覆盖绝对保底阈值；None 时沿用类常量
                                ABSOLUTE_FLOOR_THRESHOLD（阶段 6 让它可注入，便于测试与调参）
        """
        self.db = graph_db
        self.summary_embeddings = summary_embeddings  # Layer 1 簇摘要向量
        # 记录节点文本内容，用于夜间重写（简化演示）
        self.text_storage: Dict[int, str] = {}

        if floor_threshold is not None:
            self.ABSOLUTE_FLOOR_THRESHOLD = float(floor_threshold)

    def set_node_text(self, node_id: int, text: str):
        self.text_storage[node_id] = text

    def get_node_text(self, node_id: int) -> str:
        return self.text_storage.get(node_id, "")

    def _compute_similarities(self, new_vector: np.ndarray) -> Tuple[np.ndarray, List[int]]:
        """
        计算新向量与所有簇摘要的余弦相似度。
        返回 (scores, cluster_ids) 按原顺序。
        """
        ids = list(self.summary_embeddings.keys())
        if not ids:
            return np.array([]), ids
        summaries = np.array([self.summary_embeddings[cid] for cid in ids])
        # 归一化 (假设向量已归一化，再点积即余弦相似度)
        dot = np.dot(summaries, new_vector)  # (n_clusters,)
        return dot, ids

    def inject_new_knowledge(self, content: str, embedding: np.ndarray, node_id: int = None) -> int:
        """
        新知识节点挂载主流程：
        1. 创建新节点
        2. 计算与所有簇的相似度
        3. GMM 动态分界
        4. 绝对保底检查
        5. 挂载 + 脏传播
        """
        # 创建原子节点（最底层叶子）
        if node_id is None:
            node_id = self.db.add_node({"type": "leaf", "content": content, "embedding": embedding})
        self.set_node_text(node_id, content)

        # 如果没有簇存在，自动创建第一个簇
        if not self.summary_embeddings:
            logger.info("无现有簇，自动创建根簇")
            # 阶段 8 修复：原先这里只调 _create_new_cluster() 而没有挂载叶子，
            # 结果第一个叶子在树里**没有任何父边**，get_children(根簇) 恒为空，
            # nightly_rewrite / memory_graph_bridge.sync_dirty_summaries 永远看不到它。
            # 下方的孤儿分支一直是建簇 + 挂载两步都做，两处行为应一致。
            root_cluster_id = self._create_new_cluster(parent_of_new=None, new_leaf_id=node_id)
            self._mount_leaf_to_cluster(node_id, root_cluster_id, mark_dirty=True)
            return node_id

        scores, cluster_ids = self._compute_similarities(embedding)
        accept_indices, dynamic_threshold = DynamicThresholder.split(scores)

        # 检查孤儿情况：accept 中最高分仍低于绝对阈值
        accept_scores = [scores[i] for i in accept_indices]
        max_score = max(accept_scores) if accept_scores else 0.0
        if max_score < self.ABSOLUTE_FLOOR_THRESHOLD:
            logger.info(f"孤儿节点检测 (最高相似度 {max_score:.4f} < {self.ABSOLUTE_FLOOR_THRESHOLD})，创建新簇")
            # 阶段 6：用 _create_new_cluster 的返回值作为新簇 ID。
            # 原先靠 `list(self.summary_embeddings.keys())[-1]` 猜"最后一个就是新建的"，
            # 一旦 dict 顺序被后续插入打乱（或新簇没写进索引）就会挂错父节点。
            new_cluster_id = self._create_new_cluster(parent_of_new=None, new_leaf_id=node_id)
            self._mount_leaf_to_cluster(node_id, new_cluster_id, mark_dirty=True)
            return node_id

        # 挂载到所有被接受的簇
        for idx in accept_indices:
            cluster_id = cluster_ids[idx]
            self._mount_leaf_to_cluster(node_id, cluster_id, mark_dirty=True)
            # 级联脏标
            self._propagate_dirty_upwards(cluster_id)

        logger.info(f"节点 {node_id} 动态阈值 {dynamic_threshold:.4f}，挂载到 {len(accept_indices)} 个簇: {[cluster_ids[i] for i in accept_indices]}")
        return node_id

    def _mount_leaf_to_cluster(self, leaf_id: int, cluster_id: int, mark_dirty: bool):
        """在数据库中建立叶子到簇的边，并标记簇为脏"""
        self.db.add_edge(cluster_id, leaf_id, "CONTAINS")
        if mark_dirty:
            self.db.mark_dirty(cluster_id)

    def _propagate_dirty_upwards(self, cluster_id: int):
        """递归向上标记所有祖先节点为脏"""
        parents = self.db.get_parents(cluster_id)
        for p in parents:
            if not self.db.is_dirty(p):
                self.db.mark_dirty(p)
                self._propagate_dirty_upwards(p)  # 继续向上

    def _create_new_cluster(self, parent_of_new: int = None, new_leaf_id: int = None):
        """
        创建一个新的簇（父节点），可选的父簇用于多层树。
        新簇的摘要临时设置为叶子内容（若提供），embedding 也为叶子向量。
        """
        summary_text = ""
        new_emb = None
        if new_leaf_id is not None:
            summary_text = self.get_node_text(new_leaf_id)
            new_emb = self.db.nodes[new_leaf_id].get("embedding")
        cluster_id = self.db.add_node({
            "type": "summary",
            "content": summary_text,
            "embedding": new_emb,
            "level": 1
        })
        # 将新簇加入摘要索引
        if new_emb is not None:
            self.summary_embeddings[cluster_id] = new_emb
        # 如果指定了父簇（如更高层），则建立边
        if parent_of_new is not None:
            self.db.add_edge(parent_of_new, cluster_id, "CONTAINS")
        return cluster_id

    def nightly_rewrite(self, get_summary_fn=None):
        """
        夜间清算：扫描所有脏节点，从底层向高层逐层重写摘要并刷新向量。
        get_summary_fn: 函数，输入子节点内容列表，返回新摘要文本。
        此处仅做模拟，实际需调用 LLM。
        """
        dirty_nodes = self.db.get_dirty_nodes()
        if not dirty_nodes:
            logger.info("无脏节点，跳过清算")
            return

        # 按树层级从底层向高层排序（简单起见，此处假设 level 属性存在）
        nodes_with_level = [(nid, self.db.nodes[nid].get("level", 1)) for nid in dirty_nodes]
        nodes_with_level.sort(key=lambda x: x[1])  # 低层级优先

        for node_id, level in nodes_with_level:
            children = self.db.get_children(node_id)
            child_texts = [self.get_node_text(c) for c in children if c in self.text_storage]
            if not child_texts:
                continue
            # 生成新摘要 (模拟)
            if get_summary_fn:
                new_summary = get_summary_fn(child_texts)
            else:
                new_summary = " ".join(child_texts)[:200]  # 简单拼接截断

            # 更新节点内容
            self.db.nodes[node_id]["content"] = new_summary
            # 刷新 embedding (此处假设有函数可用，演示跳过)
            # self.db.nodes[node_id]["embedding"] = embed_fn(new_summary)
            # 更新 summary_embeddings (若该节点在索引中)
            if node_id in self.summary_embeddings and "embedding" in self.db.nodes[node_id]:
                self.summary_embeddings[node_id] = self.db.nodes[node_id]["embedding"]

            self.db.clean_node(node_id)
            logger.info(f"节点 {node_id} (level {level}) 已重写摘要")

    # ---------- 辅助：重建摘要向量索引 ----------
    def rebuild_summary_index(self):
        """从数据库中重新读取所有 summary 类型节点的 embedding 到索引"""
        self.summary_embeddings.clear()
        for nid, props in self.db.nodes.items():
            if props.get("type") == "summary" and "embedding" in props:
                self.summary_embeddings[nid] = props["embedding"]


# ============================================================================
# 示例用法
# ============================================================================
if __name__ == "__main__":
    # 初始化图数据库
    db = MockGraphDB()

    # 预置两个簇（Layer 1 摘要节点），每个簇有一个叶节点
    c1 = db.add_node({"type": "summary", "content": "劳动法 辞退规定", "embedding": np.array([0.9, 0.1, 0.0]), "level": 1})
    c2 = db.add_node({"type": "summary", "content": "合同法 违约金", "embedding": np.array([0.1, 0.9, 0.0]), "level": 1})
    leaf1 = db.add_node({"type": "leaf", "content": "用人单位单方解除...", "embedding": np.array([0.85, 0.2, 0.0])})
    leaf2 = db.add_node({"type": "leaf", "content": "违约金过高可调整...", "embedding": np.array([0.15, 0.85, 0.0])})
    db.add_edge(c1, leaf1, "CONTAINS")
    db.add_edge(c2, leaf2, "CONTAINS")

    summary_embs = {c1: db.nodes[c1]["embedding"], c2: db.nodes[c2]["embedding"]}
    manager = IncrementalMemoryManager(db, summary_embs)
    manager.set_node_text(leaf1, "用人单位单方解除...")
    manager.set_node_text(leaf2, "违约金过高可调整...")
    manager.set_node_text(c1, "劳动法 辞退规定")
    manager.set_node_text(c2, "合同法 违约金")

    # 注入新知识：一个偏劳动法的新法条
    new_text = "最新司法解释：试用期辞退需支付赔偿金"
    new_emb = np.array([0.88, 0.12, 0.0])  # 与 c1 高度相似
    new_id = manager.inject_new_knowledge(new_text, new_emb)

    # 检查脏节点
    print("脏节点列表:", db.get_dirty_nodes())

    # 注入一个完全无关的知识（孤儿）
    orphan_text = "航空法：无人机登记新规"
    orphan_emb = np.array([-0.2, -0.3, 0.9])
    orphan_id = manager.inject_new_knowledge(orphan_text, orphan_emb)
    print("当前簇数量:", len(manager.summary_embeddings))

    # 模拟夜间清算
    def dummy_summary_fn(texts):
        return "合并摘要: " + "; ".join(texts)[:100]
    manager.nightly_rewrite(get_summary_fn=dummy_summary_fn)
    print("清算后脏节点:", db.get_dirty_nodes())