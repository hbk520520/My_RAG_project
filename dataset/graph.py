import time
import logging
import numpy as np
import faiss
import igraph as ig
from typing import Dict, Any, Callable, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegalGraphEngine")


class LegalDenseGraphBuilder:
    def __init__(self,
                 embedding_dim: int = 768,
                 similarity_threshold: float = 0.85,
                 dedup_threshold: float = 0.98,
                 top_k_search: int = 100,
                 degree_threshold: int = 15):
        self.dim = embedding_dim
        self.threshold = similarity_threshold
        self.dedup_threshold = dedup_threshold
        self.top_k = top_k_search
        self.degree_threshold = degree_threshold

        # 图引擎切换为 igraph（无向图）
        self.graph = ig.Graph(directed=False)

        # 向量索引：HNSW + IndexIDMap，内积度量要求向量已归一化
        base_index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.index = faiss.IndexIDMap(base_index)

        logger.info(f"Engine initialized: dim={self.dim}, threshold={self.threshold}, "
                    f"dedup={self.dedup_threshold}, degree_threshold={self.degree_threshold}")

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

    # ------------------------------------------------------------------
    # 批量初始化方法（完全使用你提供的逻辑，已嵌入类中）
    # ------------------------------------------------------------------
    def build_initial_graph_batch(self,
                                  nodes_data: List[Dict[str, Any]],
                                  embeddings: np.ndarray,
                                  search_batch_size: int = 10000) -> None:
        """
        将离线生成的结构化节点和向量矩阵，全量批量注入图系统。
        此方法与 prepare_corpus.py 的输出直接对接。
        """
        total_nodes = len(nodes_data)
        if total_nodes != embeddings.shape[0]:
            raise ValueError("节点数量与矩阵维度不匹配！")

        logger.info(f"开始批量构建底层图谱，总节点数: {total_nodes}")

        # 1. 批量写入节点元数据到 igraph
        self.graph.add_vertices(total_nodes)
        self.graph.vs["name"] = [n["id"] for n in nodes_data]
        self.graph.vs["content"] = [n["content"] for n in nodes_data]
        self.graph.vs["type"] = [n["type"] for n in nodes_data]
        self.graph.vs["metadata"] = [n["metadata"] for n in nodes_data]

        # 2. 批量将向量注入 FAISS
        all_ids = np.array([n["id"] for n in nodes_data], dtype=np.int64)
        self.index.add_with_ids(embeddings.astype(np.float32), all_ids)
        logger.info("FAISS 索引全量注入完成。")

        # 3. 分块并行检索，建立稠密边
        all_edges = []
        all_weights = []
        logger.info("开始进行拓扑连边推演...")

        for i in range(0, total_nodes, search_batch_size):
            end_idx = min(i + search_batch_size, total_nodes)
            batch_emb = embeddings[i:end_idx].astype(np.float32)
            batch_ids = all_ids[i:end_idx]

            sims, n_ids = self.index.search(batch_emb, self.top_k)

            for row_idx, query_id in enumerate(batch_ids):
                for sim, target_id in zip(sims[row_idx], n_ids[row_idx]):
                    if (self.threshold <= sim < self.dedup_threshold
                            and target_id != query_id
                            and target_id != -1):
                        all_edges.append((query_id, int(target_id)))
                        all_weights.append(float(sim))

        # 4. 去重并批量添加边
        logger.info(f"推演结束，共计算出 {len(all_edges)} 条合法边，准备批量落图。")
        unique_edges = {}
        for edge, weight in zip(all_edges, all_weights):
            sorted_edge = tuple(sorted(edge))
            if sorted_edge not in unique_edges:
                unique_edges[sorted_edge] = weight

        edges_list = list(unique_edges.keys())
        weights_list = list(unique_edges.values())

        # igraph 要求使用内部顶点索引，构建 name->index 映射
        name_to_index = {v["name"]: v.index for v in self.graph.vs}
        igraph_edges = [(name_to_index[e[0]], name_to_index[e[1]]) for e in edges_list]

        self.graph.add_edges(igraph_edges)
        self.graph.es["weight"] = weights_list

        logger.info("初始稠密图谱装配彻底完毕！")

    # ------------------------------------------------------------------
    # 单节点入库 + 动态连边（已适配 igraph）
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

        # 防线一：极度相似去重
        if self.index.ntotal > 0:
            sims, n_ids = self.index.search(norm_emb, 1)
            if sims[0][0] >= self.dedup_threshold and n_ids[0][0] != -1:
                logger.warning(f"丢弃节点 {node_id}: 与现存节点 {n_ids[0][0]} 相似度 {sims[0][0]:.4f}")
                return False

        # 图中注册节点（igraph 添加顶点）
        self.graph.add_vertex(name=node_id,
                              content=content,
                              type=node_type,
                              metadata=metadata)

        # 防线二：区间动态连边
        if self.index.ntotal > 0:
            k = min(self.top_k, self.index.ntotal)
            similarities, neighbor_ids = self.index.search(norm_emb, k)

            edges_to_add = []
            weights_to_add = []
            for sim, n_id in zip(similarities[0], neighbor_ids[0]):
                if (self.threshold <= sim < self.dedup_threshold
                        and n_id != -1
                        and n_id != node_id):
                    edges_to_add.append((node_id, int(n_id)))
                    weights_to_add.append(float(sim))

            if edges_to_add:
                name_to_index = {v["name"]: v.index for v in self.graph.vs}
                igraph_edges = [(name_to_index[src], name_to_index[dst]) for src, dst in edges_to_add]
                self.graph.add_edges(igraph_edges)
                self.graph.es[-len(igraph_edges):]["weight"] = weights_to_add
                logger.info(f"Node {node_id} connected to {len(edges_to_add)} existing nodes.")

        # 向量索引更新
        self.index.add_with_ids(norm_emb, np.array([node_id], dtype=np.int64))

        # 入库后检查是否触发聚类
        self.check_and_trigger_clustering(node_id)
        return True

    # ------------------------------------------------------------------
    # LLM 摘要平权算子（适配 igraph）
    # ------------------------------------------------------------------
    def generate_summary_node(self,
                              new_summary_id: int,
                              member_ids: List[int],
                              llm_generate_fn: Callable[[List[str]], str],
                              embedding_fn: Callable[[str], np.ndarray],
                              summary_metadata: Optional[Dict[str, Any]] = None) -> bool:
        # 提取成员节点的 content
        texts = []
        for nid in member_ids:
            try:
                v = self.graph.vs.find(name=nid)
                texts.append(v["content"])
            except ValueError:
                continue
        if not texts:
            logger.error("无效的成员 ID，无法生成 Summary")
            return False

        summary_content = llm_generate_fn(texts)
        summary_embedding = embedding_fn(summary_content)

        if summary_metadata is None:
            summary_metadata = {}
        summary_metadata['source_cluster_size'] = len(member_ids)
        summary_metadata['is_auto_generated'] = True

        return self.add_node(
            node_id=new_summary_id,
            content=summary_content,
            node_type="Summary",
            embedding=summary_embedding,
            metadata=summary_metadata
        )

    # ------------------------------------------------------------------
    # 自动聚类触发器（适配 igraph）
    # ------------------------------------------------------------------
    def check_and_trigger_clustering(self, target_node_id: int) -> bool:
        try:
            v = self.graph.vs.find(name=target_node_id)
        except ValueError:
            return False

        degree = v.degree()
        if degree < self.degree_threshold:
            return False

        logger.info(f"节点 {target_node_id} 连通度达到 {degree}，触发局部向上聚类！")

        # 提取一跳邻居（按 name 返回）
        neighbor_names = [self.graph.vs[n]["name"] for n in self.graph.neighbors(v.index)]
        member_ids = [target_node_id] + neighbor_names

        # 修剪原图中该节点与所有邻居的边，防止子图爆炸
        edges_to_del = [(v.index, n) for n in self.graph.neighbors(v.index)]
        self.graph.delete_edges(edges_to_del)

        # 生成新 Summary 节点（需要提前注入外部函数）
        new_id = self._generate_unique_id()
        if hasattr(self, 'llm_generate_fn') and hasattr(self, 'embedding_fn'):
            self.generate_summary_node(
                new_summary_id=new_id,
                member_ids=member_ids,
                llm_generate_fn=self.llm_generate_fn,
                embedding_fn=self.embedding_fn,
                summary_metadata={"trigger_source": target_node_id,
                                  "type": "auto_emerged_summary"}
            )
        else:
            logger.warning("LLM/Embedding 函数未注入，跳过 Summary 生成。")

        logger.info(f"局部子图已被重构并压缩为 Summary 节点 [{new_id}]。")
        return True