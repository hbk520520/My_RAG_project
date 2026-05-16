import time
import logging
import numpy as np
import faiss
import igraph as ig
from typing import Dict, Any, Callable, List, Optional
import jieba
from rank_bm25 import BM25Okapi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegalGraphEngine")


class LegalDenseGraphBuilder:
    def __init__(self,
                 embedding_dim: int = 768,
                 similarity_threshold: float = 0.85,      # 加权和下限
                 dedup_threshold: float = 0.99,           # 现在作为“高相似标签”阈值
                 top_k_search: int = 50,                  # 候选邻居数（用于BM25计算）
                 degree_threshold: int = 15):
        self.dim = embedding_dim
        self.threshold = similarity_threshold             # 0.85
        self.high_sim_threshold = dedup_threshold         # 0.99
        self.top_k = top_k_search
        self.degree_threshold = degree_threshold

        # 图引擎（无向图）
        self.graph = ig.Graph(directed=False)

        # FAISS 索引（内积度量，向量需归一化）
        base_index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.index = faiss.IndexIDMap(base_index)

        logger.info(f"Engine initialized: dim={self.dim}, threshold={self.threshold}, "
                    f"high_sim_label={self.high_sim_threshold}, top_k={self.top_k}")

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _l2_normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def _generate_unique_id(self) -> int:
        if self.graph.vcount() == 0:
            return 1
        return max(self.graph.vs["name"]) + 1

    def _tokenize(self, text: str) -> List[str]:
        """中文分词，用于BM25"""
        return list(jieba.cut(text))

    def _compute_bm25_scores(self, query_text: str, doc_texts: List[str]) -> np.ndarray:
        """
        计算查询与一组文档的 BM25 分数。
        返回与 doc_texts 同长度的 numpy 数组。
        """
        if not doc_texts:
            return np.array([])
        tokenized_docs = [self._tokenize(doc) for doc in doc_texts]
        tokenized_query = self._tokenize(query_text)
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(tokenized_query)   # 长度 = len(doc_texts)
        return np.array(scores, dtype=np.float32)

    def _normalize_array(self, arr: np.ndarray) -> np.ndarray:
        """将数组归一化到 [0,1]（除以最大值，若最大值0则全0）"""
        max_val = np.max(arr)
        if max_val == 0:
            return arr
        return arr / max_val

    # ------------------------------------------------------------------
    # 批量初始化方法（新增混合相似度择优连边）
    # ------------------------------------------------------------------
    def build_initial_graph_batch(self,
                                  nodes_data: List[Dict[str, Any]],
                                  embeddings: np.ndarray,
                                  search_batch_size: int = 10000) -> None:
        """
        离线全量批量注入节点，并根据余弦+BM25加权和择优连一条边。
        """
        total_nodes = len(nodes_data)
        if total_nodes != embeddings.shape[0]:
            raise ValueError("节点数量与矩阵维度不匹配！")

        logger.info(f"开始批量构建底层图谱（混合相似度），总节点数: {total_nodes}")

        # 1. 注册节点
        self.graph.add_vertices(total_nodes)
        self.graph.vs["name"] = [n["id"] for n in nodes_data]
        self.graph.vs["content"] = [n["content"] for n in nodes_data]
        self.graph.vs["type"] = [n["type"] for n in nodes_data]
        self.graph.vs["metadata"] = [n["metadata"] for n in nodes_data]

        # 2. 批量注入FAISS
        all_ids = np.array([n["id"] for n in nodes_data], dtype=np.int64)
        self.index.add_with_ids(embeddings.astype(np.float32), all_ids)
        logger.info("FAISS 索引全量注入完成。")

        # 3. 逐节点搜索 top_k 邻居，计算混合相似度并决定连边/打标签
        edges_to_add = []
        edge_weights = []
        similar_labels = {}   # node_id -> similar_to_id

        for start in range(0, total_nodes, search_batch_size):
            end = min(start + search_batch_size, total_nodes)
            batch_emb = embeddings[start:end].astype(np.float32)
            batch_ids = all_ids[start:end]

            sims, n_ids = self.index.search(batch_emb, self.top_k)

            for row_idx, query_id in enumerate(batch_ids):
                cos_scores = sims[row_idx]
                cand_ids = n_ids[row_idx]
                # 过滤掉自身及无效ID
                mask = (cand_ids != query_id) & (cand_ids != -1)
                filtered_cos = cos_scores[mask]
                filtered_ids = cand_ids[mask]
                if len(filtered_ids) == 0:
                    continue   # 无有效邻居，保持孤立

                # 获取候选节点的 content
                cand_contents = []
                for cid in filtered_ids:
                    try:
                        v = self.graph.vs.find(name=int(cid))
                        cand_contents.append(v["content"])
                    except ValueError:
                        cand_contents.append("")   # 容错

                # 计算 BM25 分数
                query_content = self.graph.vs.find(name=query_id)["content"]
                bm25_scores = self._compute_bm25_scores(query_content, cand_contents)

                # 归一化并加权
                norm_cos = self._normalize_array(filtered_cos)
                norm_bm25 = self._normalize_array(bm25_scores)
                weighted = 0.3 * norm_cos + 0.7 * norm_bm25

                # 分离高相似区间 [0.99, 1.0] 和连接区间 [0.85, 0.99)
                high_mask = weighted >= self.high_sim_threshold
                connect_mask = (weighted >= self.threshold) & (weighted < self.high_sim_threshold)

                # 优先处理高相似：打标签（记录相似对象）
                if np.any(high_mask):
                    # 若多个高相似，取加权和最大的那个
                    high_indices = np.where(high_mask)[0]
                    best_high_idx = high_indices[np.argmax(weighted[high_indices])]
                    similar_labels[query_id] = int(filtered_ids[best_high_idx])

                # 其次处理普通连边：只连加权和最大的一个
                if np.any(connect_mask):
                    connect_indices = np.where(connect_mask)[0]
                    best_connect_idx = connect_indices[np.argmax(weighted[connect_indices])]
                    target_id = int(filtered_ids[best_connect_idx])
                    edges_to_add.append((query_id, target_id))
                    edge_weights.append(float(weighted[best_connect_idx]))

        # 批量添加边
        if edges_to_add:
            name_to_index = {v["name"]: v.index for v in self.graph.vs}
            igraph_edges = [(name_to_index[src], name_to_index[dst]) for src, dst in edges_to_add]
            self.graph.add_edges(igraph_edges)
            self.graph.es["weight"] = edge_weights
            logger.info(f"批量添加 {len(edges_to_add)} 条择优边。")

        # 批量写入相似标签
        for node_id, sim_id in similar_labels.items():
            try:
                v = self.graph.vs.find(name=node_id)
                v["metadata"]["similar_to"] = sim_id
                v["metadata"]["has_similar_label"] = True
            except ValueError:
                pass
        logger.info(f"标记 {len(similar_labels)} 个高相似标签。")

        logger.info("初始稠密图谱（混合相似度版）装配完毕！")

    # ------------------------------------------------------------------
    # 单节点入库 + 动态择优连边
    # ------------------------------------------------------------------
    def add_node(self,
                 node_id: int,
                 content: str,
                 node_type: str,
                 embedding: np.ndarray,
                 metadata: Optional[Dict[str, Any]] = None) -> bool:
        if metadata is None:
            metadata = {}

        norm_emb = self._l2_normalize(embedding).reshape(1, -1).astype(np.float32)

        # 图中注册新节点
        self.graph.add_vertex(name=node_id,
                              content=content,
                              type=node_type,
                              metadata=metadata)

        # 若无历史节点，直接加入索引返回
        if self.index.ntotal == 0:
            self.index.add_with_ids(norm_emb, np.array([node_id], dtype=np.int64))
            return True

        # 使用 FAISS 检索 top_k 余弦近邻
        k = min(self.top_k, self.index.ntotal)
        similarities, neighbor_ids = self.index.search(norm_emb, k)
        cos_scores = similarities[0]
        cand_ids = neighbor_ids[0]

        # 过滤自身与无效
        mask = (cand_ids != node_id) & (cand_ids != -1)
        filtered_cos = cos_scores[mask]
        filtered_ids = cand_ids[mask].astype(int)

        if len(filtered_ids) == 0:
            self.index.add_with_ids(norm_emb, np.array([node_id], dtype=np.int64))
            return True

        # 获取候选节点内容
        cand_contents = []
        for cid in filtered_ids:
            try:
                v = self.graph.vs.find(name=cid)
                cand_contents.append(v["content"])
            except ValueError:
                cand_contents.append("")

        # 计算 BM25 并混合加权
        bm25_scores = self._compute_bm25_scores(content, cand_contents)
        norm_cos = self._normalize_array(filtered_cos)
        norm_bm25 = self._normalize_array(bm25_scores)
        weighted = 0.3 * norm_cos + 0.7 * norm_bm25

        # 处理高相似标签
        high_mask = weighted >= self.high_sim_threshold
        if np.any(high_mask):
            high_indices = np.where(high_mask)[0]
            best_high_idx = high_indices[np.argmax(weighted[high_indices])]
            similar_target = int(filtered_ids[best_high_idx])
            v_new = self.graph.vs.find(name=node_id)
            v_new["metadata"]["similar_to"] = similar_target
            v_new["metadata"]["has_similar_label"] = True
            logger.info(f"节点 {node_id} 与 {similar_target} 高度相似（加权和≥{self.high_sim_threshold}），已打标签。")

        # 处理择优连边
        connect_mask = (weighted >= self.threshold) & (weighted < self.high_sim_threshold)
        if np.any(connect_mask):
            connect_indices = np.where(connect_mask)[0]
            best_idx = connect_indices[np.argmax(weighted[connect_indices])]
            target_id = int(filtered_ids[best_idx])
            # 添加边
            name_to_index = {v["name"]: v.index for v in self.graph.vs}
            src_idx = name_to_index[node_id]
            dst_idx = name_to_index[target_id]
            self.graph.add_edge(src_idx, dst_idx, weight=float(weighted[best_idx]))
            logger.info(f"节点 {node_id} 择优连接至 {target_id}（加权和 {weighted[best_idx]:.4f}）。")
        else:
            logger.info(f"节点 {node_id} 无满足连边条件的邻居，保持孤立。")

        # 更新 FAISS 索引
        self.index.add_with_ids(norm_emb, np.array([node_id], dtype=np.int64))

        # 检查是否触发聚类（保持原有逻辑）
        self.check_and_trigger_clustering(node_id)
        return True

    # ------------------------------------------------------------------
    # 以下方法保持不变（仅展示，无修改）
    # ------------------------------------------------------------------
    def generate_summary_node(self,
                              new_summary_id: int,
                              member_ids: List[int],
                              llm_generate_fn: Callable[[List[str]], str],
                              embedding_fn: Callable[[str], np.ndarray],
                              summary_metadata: Optional[Dict[str, Any]] = None) -> bool:
        # ... 原逻辑保持不变
        pass

    def check_and_trigger_clustering(self, target_node_id: int) -> bool:
        # ... 原逻辑保持不变
        pass