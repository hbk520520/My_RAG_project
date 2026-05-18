import time
import logging
import numpy as np
import faiss
import igraph as ig
from typing import Dict, Any, Callable, List, Optional, Set
from FlagEmbedding import BGEM3FlagModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegalGraphEngine")


class LegalDenseGraphBuilder:
    def __init__(self,
                 model_name: str = 'BAAI/bge-m3',
                 use_fp16: bool = True,
                 embedding_dim: int = 1024,
                 alpha_dense: float = 0.3,
                 connect_threshold: float = 0.85,
                 label_threshold: float = 0.99,
                 top_k_search: int = 50,
                 degree_threshold: int = 15):
        self.dim = embedding_dim
        self.alpha = alpha_dense
        self.connect_thresh = connect_threshold
        self.label_thresh = label_threshold
        self.top_k = top_k_search
        self.degree_threshold = degree_threshold

        # 图引擎
        self.graph = ig.Graph(directed=False)

        # FAISS 索引
        base_index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.index = faiss.IndexIDMap(base_index)

        # 加载 BGE‑M3
        logger.info(f"正在加载 BGE‑M3 模型: {model_name} ...")
        self.encoder = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        logger.info("BGE‑M3 模型加载完成。")

        # ---------- 新增：法律更新维护相关属性 ----------
        self.tombstone_ids: Set[int] = set()          # 已软删除的节点 ID
        self.dirty_summary_ids: Set[int] = set()      # 需要重算的摘要节点 ID

    # ------------------------------------------------------------------
    # 原有编码与相似度方法（完全保留）
    # ------------------------------------------------------------------
    def encode_text(self, text: str) -> Dict[str, Any]:
        out = self.encoder.encode([text], return_dense=True, return_sparse=True)
        dense = out['dense_vecs'][0]
        dense = dense / np.linalg.norm(dense)
        sparse = out['lexical_weights'][0]
        return {"dense": dense.astype(np.float32), "sparse": sparse}

    def hybrid_similarity(self,
                          dense_a: np.ndarray,
                          sparse_a: Dict[str, float],
                          dense_b: np.ndarray,
                          sparse_b: Dict[str, float]) -> float:
        cos_sim = float(np.dot(dense_a, dense_b))
        common = set(sparse_a.keys()) & set(sparse_b.keys())
        lex_score = sum(sparse_a[t] * sparse_b[t] for t in common)
        return self.alpha * cos_sim + (1.0 - self.alpha) * lex_score

    def _generate_unique_id(self) -> int:
        if self.graph.vcount() == 0:
            return 1
        return max(self.graph.vs["name"]) + 1

    # ------------------------------------------------------------------
    # 原有批量构建（不动）
    # ------------------------------------------------------------------
    def build_initial_graph_batch(self,
                                  nodes_data: List[Dict[str, Any]],
                                  search_batch_size: int = 10000) -> None:
        # ... 完整保留原始实现，此处省略以节省篇幅，代码未改动 ...
        pass

    # ------------------------------------------------------------------
    # 增量添加节点（修改：集成挂载与脏传播，并过滤 tombstone 候选）
    # ------------------------------------------------------------------
    def add_node(self,
                 node_id: int,
                 content: str,
                 node_type: str,
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        if metadata is None:
            metadata = {}

        # 编码新节点
        enc = self.encode_text(content)
        dense_vec = enc["dense"]
        sparse_dict = enc["sparse"]

        # 初始化元数据中的状态与父节点
        metadata.setdefault("status", "active")
        metadata.setdefault("parent_id", None)

        # 注册到图
        self.graph.add_vertex(name=node_id,
                              content=content,
                              type=node_type,
                              metadata=metadata,
                              dense=dense_vec,
                              sparse=sparse_dict)

        if self.index.ntotal == 0:
            self.index.add_with_ids(dense_vec.reshape(1, -1), np.array([node_id], dtype=np.int64))
            return True

        # 检索 top_k 候选（排除 tombstone 节点）
        k = min(self.top_k * 2, self.index.ntotal)   # 多取一些，过滤后可能不够
        sims, n_ids = self.index.search(dense_vec.reshape(1, -1), k)
        cos_scores = sims[0]
        cand_ids = n_ids[0]

        # 过滤：排除自身、-1，以及 tombstone 节点
        valid_mask = (cand_ids != node_id) & (cand_ids != -1)
        for i, cid in enumerate(cand_ids):
            if valid_mask[i] and cid in self.tombstone_ids:
                valid_mask[i] = False
        filt_cos = cos_scores[valid_mask]
        filt_ids = cand_ids[valid_mask].astype(int)
        if len(filt_ids) == 0:
            self.index.add_with_ids(dense_vec.reshape(1, -1), np.array([node_id], dtype=np.int64))
            return True

        # 限制到 top_k 个有效候选
        if len(filt_ids) > self.top_k:
            top_idx = np.argpartition(filt_cos, -self.top_k)[-self.top_k:]
            filt_ids = filt_ids[top_idx]
            filt_cos = filt_cos[top_idx]

        # 计算混合得分
        scores = []
        for cid, cos in zip(filt_ids, filt_cos):
            try:
                v = self.graph.vs.find(name=cid)
                c_sparse = v["sparse"]
                lex = sum(sparse_dict[t] * c_sparse[t] for t in set(sparse_dict) & set(c_sparse))
                scores.append(self.alpha * cos + (1.0 - self.alpha) * lex)
            except ValueError:
                scores.append(0.0)
        scores = np.array(scores)

        # 高相似标签
        high_mask = scores >= self.label_thresh
        if np.any(high_mask):
            best_idx = np.argmax(scores * high_mask)
            similar_target = int(filt_ids[best_idx])
            self.graph.vs.find(name=node_id)["metadata"]["similar_to"] = similar_target
            self.graph.vs.find(name=node_id)["metadata"]["has_similar_label"] = True

        # 择优连边
        conn_mask = (scores >= self.connect_thresh) & (scores < self.label_thresh)
        if np.any(conn_mask):
            valid_scores = scores.copy()
            valid_scores[~conn_mask] = -np.inf
            best_idx = np.argmax(valid_scores)
            target_id = int(filt_ids[best_idx])
            src_idx = self.graph.vs.find(name=node_id).index
            dst_idx = self.graph.vs.find(name=target_id).index
            self.graph.add_edge(src_idx, dst_idx, weight=float(scores[best_idx]))
            logger.info(f"节点 {node_id} 连边至 {target_id}, 得分 {scores[best_idx]:.4f}")

        # 更新 FAISS
        self.index.add_with_ids(dense_vec.reshape(1, -1), np.array([node_id], dtype=np.int64))

        # ---------- 新增功能：增量挂载与脏传播 ----------
        # 仅对非 Summary 节点执行自动挂载（Summary 节点自身不需要父节点）
        if node_type != "Summary":
            self.mount_to_parent(node_id)

        # 新节点加入，它的父摘要需要标记为脏
        self._propagate_dirty(node_id)

        # 原有聚类触发器（保留）
        self.check_and_trigger_clustering(node_id)
        return True

    # ------------------------------------------------------------------
    # 新增：软删除节点
    # ------------------------------------------------------------------
    def tombstone_node(self, node_id: int) -> bool:
        """将指定节点标记为 tombstone，并向上传播脏标记"""
        try:
            v = self.graph.vs.find(name=node_id)
        except ValueError:
            logger.error(f"节点 {node_id} 不存在，无法软删除")
            return False

        if v["metadata"].get("status") == "tombstone":
            logger.info(f"节点 {node_id} 已经是 tombstone 状态")
            return True

        v["metadata"]["status"] = "tombstone"
        self.tombstone_ids.add(node_id)
        logger.info(f"节点 {node_id} 已软删除（tombstone），对前端隐身")

        # 向上传播脏标记
        self._propagate_dirty(node_id)
        return True

    # ------------------------------------------------------------------
    # 新增：增量挂载（找到最合适的父 Summary 节点）
    # ------------------------------------------------------------------
    def mount_to_parent(self, child_id: int):
        """为 child_id 节点寻找最相似的 Summary 类型节点作为父节点"""
        try:
            child_v = self.graph.vs.find(name=child_id)
        except ValueError:
            return

        # 收集所有状态为 active 的 Summary 节点
        summary_vertices = [
            v for v in self.graph.vs
            if v["type"] == "Summary" and v["metadata"].get("status") != "tombstone"
        ]
        if not summary_vertices:
            return

        child_dense = child_v["dense"]
        child_sparse = child_v["sparse"]
        best_score = -1
        best_parent_id = None
        for sv in summary_vertices:
            score = self.hybrid_similarity(child_dense, child_sparse,
                                           sv["dense"], sv["sparse"])
            if score > best_score:
                best_score = score
                best_parent_id = sv["name"]

        if best_parent_id is not None and best_score >= self.connect_thresh:  # 至少达到连边阈值
            child_v["metadata"]["parent_id"] = best_parent_id
            logger.info(f"节点 {child_id} 已挂载到父 Summary {best_parent_id} (得分 {best_score:.4f})")
        else:
            logger.info(f"节点 {child_id} 未找到合适的父 Summary，保持无父节点")

    # ------------------------------------------------------------------
    # 新增：脏节点向上传播
    # ------------------------------------------------------------------
    def _propagate_dirty(self, start_node_id: int):
        """从给定节点开始，沿 parent_id 链向上将摘要节点标记为 dirty"""
        current_id = start_node_id
        visited = set()
        while current_id is not None:
            if current_id in visited:
                break  # 防止环路
            visited.add(current_id)
            try:
                v = self.graph.vs.find(name=current_id)
            except ValueError:
                break
            # 仅对 Summary 节点进行 dirty 标记
            if v["type"] == "Summary":
                v["metadata"]["dirty"] = True
                self.dirty_summary_ids.add(current_id)
                logger.info(f"摘要节点 {current_id} 标记为 dirty")
            # 继续向上
            parent_id = v["metadata"].get("parent_id")
            if parent_id and parent_id != current_id:
                current_id = parent_id
            else:
                break

    # ------------------------------------------------------------------
    # 新增：夜间局部重算
    # ------------------------------------------------------------------
    def nightly_recalc_summaries(self,
                                 llm_generate_fn: Callable[[List[str]], str]):
        """重新生成所有被标记为 dirty 的 Summary 节点，并清除 dirty 标记"""
        if not self.dirty_summary_ids:
            logger.info("无脏摘要节点，跳过重算")
            return

        logger.info(f"开始夜间局部重算，共 {len(self.dirty_summary_ids)} 个脏摘要节点")
        recalc_list = list(self.dirty_summary_ids)
        self.dirty_summary_ids.clear()

        for summary_id in recalc_list:
            try:
                v = self.graph.vs.find(name=summary_id)
            except ValueError:
                continue
            if v["type"] != "Summary":
                continue

            # 收集所有直接子节点（parent_id 等于该 summary_id 的节点）
            child_texts = []
            for child in self.graph.vs:
                if child["metadata"].get("parent_id") == summary_id and child["metadata"].get("status") != "tombstone":
                    child_texts.append(child["content"])
            if not child_texts:
                logger.warning(f"摘要节点 {summary_id} 无有效子节点，跳过重算")
                continue

            # 生成新摘要
            new_summary = llm_generate_fn(child_texts)
            # 更新该节点的 content 并重新编码向量
            enc = self.encode_text(new_summary)
            v["content"] = new_summary
            v["dense"] = enc["dense"]
            v["sparse"] = enc["sparse"]
            v["metadata"]["dirty"] = False
            logger.info(f"摘要节点 {summary_id} 重算完成")

            # 更新 FAISS 中的向量（覆盖写入需要先删除再添加，FAISS 不支持直接修改）
            # 简单做法：不更新 FAISS，因为摘要节点通常不参与初始检索（由子节点检索后汇聚）
            # 若需要更新，可重建索引，此处从略

    # ------------------------------------------------------------------
    # 原有方法保留（generate_summary_node, check_and_trigger_clustering）
    # ------------------------------------------------------------------
    def generate_summary_node(self, ...):  # 同原实现，保留不变
        pass

    def check_and_trigger_clustering(self, target_node_id: int) -> bool:  # 同原实现，保留不变
        pass


# ------------------------------------------------------------------
# 使用示例（演示法律更新）
# ------------------------------------------------------------------
if __name__ == "__main__":
    engine = LegalDenseGraphBuilder(alpha_dense=0.3)

    # 模拟旧法条入库
    engine.add_node(101, "旧法条：故意杀人，处十年以上有期徒刑。", "article")
    engine.add_node(102, "新法条：故意杀人，处死刑、无期徒刑或十年以上有期徒刑。", "article")
    engine.add_node(200, "Summary 节点：暴力犯罪量刑标准", "Summary")

    # 手动设置父节点（实际中由 mount_to_parent 自动完成）
    engine.graph.vs.find(name=101)["metadata"]["parent_id"] = 200
    engine.graph.vs.find(name=102)["metadata"]["parent_id"] = 200

    # 软删除旧法条
    engine.tombstone_node(101)

    # 此时 Summary 200 已被标记为 dirty
    print("脏摘要节点集合:", engine.dirty_summary_ids)

    # 模拟 LLM 重生成函数
    def fake_llm(texts):
        return "根据最新法条，暴力犯罪量刑更新为..."

    engine.nightly_recalc_summaries(fake_llm)
    print("Summary 200 新内容:", engine.graph.vs.find(name=200)["content"])