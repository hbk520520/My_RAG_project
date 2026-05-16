import time
import logging
import numpy as np
import faiss
import igraph as ig
from typing import Dict, Any, Callable, List, Optional
from FlagEmbedding import BGEM3FlagModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegalGraphEngine")


class LegalDenseGraphBuilder:
    """
    基于 BGE‑M3 的法律知识图谱构建引擎。
    一次前向传播同时获得 dense embedding 和 sparse lexicon，
    混合得分 = alpha_dense * cos_sim + (1-alpha_dense) * sparse_match，
    择优连边，高相似打标签。
    """

    def __init__(self,
                 model_name: str = 'BAAI/bge-m3',
                 use_fp16: bool = True,
                 embedding_dim: int = 1024,          # BGE‑M3 dense 维度
                 alpha_dense: float = 0.3,           # 稠密得分权重（默认 0.3，稀疏 0.7）
                 connect_threshold: float = 0.85,
                 label_threshold: float = 0.99,
                 top_k_search: int = 50,
                 degree_threshold: int = 15):
        self.dim = embedding_dim
        self.alpha = alpha_dense                     # 混合加权中的稠密比例
        self.connect_thresh = connect_threshold      # 连边下限 (含)
        self.label_thresh = label_threshold          # 高相似标签下限
        self.top_k = top_k_search
        self.degree_threshold = degree_threshold

        # 图引擎
        self.graph = ig.Graph(directed=False)

        # FAISS 索引（内积度量，向量必须 L2 归一化）
        base_index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.index = faiss.IndexIDMap(base_index)

        # 加载 BGE‑M3 模型（一次前向传播同时得到 dense 和 sparse）
        logger.info(f"正在加载 BGE‑M3 模型: {model_name} ...")
        self.encoder = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        logger.info("BGE‑M3 模型加载完成。")

    # ------------------------------------------------------------------
    # 核心编码方法：返回归一化 dense 向量 + sparse 词典
    # ------------------------------------------------------------------
    def encode_text(self, text: str) -> Dict[str, Any]:
        """
        对单条文本编码，返回：
        {
            "dense": np.ndarray (dim,)   已 L2 归一化,
            "sparse": dict {token: weight}
        }
        """
        out = self.encoder.encode([text], return_dense=True, return_sparse=True)
        dense = out['dense_vecs'][0]
        dense = dense / np.linalg.norm(dense)          # L2 归一化，用于内积
        sparse = out['lexical_weights'][0]              # {词: 权重}
        return {"dense": dense.astype(np.float32), "sparse": sparse}

    # ------------------------------------------------------------------
    # 混合得分计算（纯 CPU，极速）
    # ------------------------------------------------------------------
    def hybrid_similarity(self,
                          dense_a: np.ndarray,
                          sparse_a: Dict[str, float],
                          dense_b: np.ndarray,
                          sparse_b: Dict[str, float]) -> float:
        """
        计算混合相似度：
        dense 部分 = dense_a · dense_b （等价于余弦，因为已归一化）
        sparse 部分 = Σ_{词 ∈ 交集} sparse_a[词] * sparse_b[词]
        最终得分 = alpha * dense + (1 - alpha) * sparse
        """
        # 稠密内积
        cos_sim = float(np.dot(dense_a, dense_b))

        # 稀疏词汇匹配
        common = set(sparse_a.keys()) & set(sparse_b.keys())
        lex_score = sum(sparse_a[t] * sparse_b[t] for t in common)

        return self.alpha * cos_sim + (1.0 - self.alpha) * lex_score

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    def _generate_unique_id(self) -> int:
        if self.graph.vcount() == 0:
            return 1
        return max(self.graph.vs["name"]) + 1

    # ------------------------------------------------------------------
    # 批量初始化图谱
    # ------------------------------------------------------------------
    def build_initial_graph_batch(self,
                                  nodes_data: List[Dict[str, Any]],
                                  search_batch_size: int = 10000) -> None:
        """
        批量导入节点。nodes_data 中每个节点需包含:
        {
            "id": int,
            "content": str,
            "type": str,
            "metadata": dict (可选)
        }
        所有节点的向量由 BGE‑M3 实时编码生成，无需预先提供 embeddings。
        """
        total_nodes = len(nodes_data)
        logger.info(f"开始批量构建图谱（BGE‑M3），节点数: {total_nodes}")

        # 1. 批量编码所有节点
        texts = [n["content"] for n in nodes_data]
        logger.info("批量编码中...")
        outputs = self.encoder.encode(texts, return_dense=True, return_sparse=True,
                                      batch_size=512, max_length=8192)
        dense_mat = outputs['dense_vecs']     # (total, dim)
        sparse_dicts = outputs['lexical_weights']  # list of dict

        # L2 归一化
        norms = np.linalg.norm(dense_mat, axis=1, keepdims=True)
        dense_mat = dense_mat / np.where(norms == 0, 1, norms)
        dense_mat = dense_mat.astype(np.float32)

        # 2. 注册图节点
        self.graph.add_vertices(total_nodes)
        self.graph.vs["name"] = [n["id"] for n in nodes_data]
        self.graph.vs["content"] = texts
        self.graph.vs["type"] = [n.get("type", "unknown") for n in nodes_data]
        self.graph.vs["metadata"] = [n.get("metadata", {}) for n in nodes_data]
        self.graph.vs["dense"] = list(dense_mat)
        self.graph.vs["sparse"] = sparse_dicts

        # 3. FAISS 注入稠密向量
        all_ids = np.array([n["id"] for n in nodes_data], dtype=np.int64)
        self.index.add_with_ids(dense_mat, all_ids)
        logger.info("FAISS 索引注入完成。")

        # 4. 逐节点择优连边
        edges_to_add = []
        edge_weights = []
        similar_labels = {}  # node_id -> similar_to_id

        logger.info("开始拓扑连边（混合相似度）...")
        for i in range(0, total_nodes, search_batch_size):
            end = min(i + search_batch_size, total_nodes)
            batch_emb = dense_mat[i:end]
            batch_ids = all_ids[i:end]

            sims, n_ids = self.index.search(batch_emb, self.top_k)

            for row, qid in enumerate(batch_ids):
                cos_scores = sims[row]
                cand_ids = n_ids[row]
                # 排除自身和无效 ID
                mask = (cand_ids != qid) & (cand_ids != -1)
                filt_cos = cos_scores[mask]
                filt_ids = cand_ids[mask].astype(int)
                if len(filt_ids) == 0:
                    continue

                # 获取查询节点的 sparse
                q_sparse = self.graph.vs.find(name=qid)["sparse"]

                # 计算混合得分
                scores = []
                for cid in filt_ids:
                    try:
                        v = self.graph.vs.find(name=cid)
                        c_dense = v["dense"]
                        c_sparse = v["sparse"]
                        # 这里 q_dense 已经在 batch_emb 中，可直接用 batch_emb[row]
                        # 但 FAISS 返回的 cos 是内积，正好就是 q_dense · c_dense
                        # 为了复用，我们直接用 cos_scores 作为 dense 部分
                        cos = cos_scores[mask][len(scores)]  # 对应索引
                        lex = sum(q_sparse[t] * c_sparse[t] for t in set(q_sparse) & set(c_sparse))
                        score = self.alpha * cos + (1 - self.alpha) * lex
                        scores.append(score)
                    except ValueError:
                        scores.append(0.0)

                scores = np.array(scores)
                # 高相似标签：>= label_thresh，取最大者
                high_mask = scores >= self.label_thresh
                if np.any(high_mask):
                    best_idx = np.argmax(scores * high_mask)  # 只在 high 中选
                    similar_labels[qid] = int(filt_ids[best_idx])

                # 连边候选：>= connect_thresh 且 < label_thresh，取最大者
                conn_mask = (scores >= self.connect_thresh) & (scores < self.label_thresh)
                if np.any(conn_mask):
                    # 在 conn_mask 中选最大值
                    valid_scores = scores.copy()
                    valid_scores[~conn_mask] = -np.inf
                    best_idx = np.argmax(valid_scores)
                    target_id = int(filt_ids[best_idx])
                    edges_to_add.append((qid, target_id))
                    edge_weights.append(float(scores[best_idx]))

        # 批量落边
        if edges_to_add:
            name2idx = {v["name"]: v.index for v in self.graph.vs}
            ig_edges = [(name2idx[s], name2idx[d]) for s, d in edges_to_add]
            self.graph.add_edges(ig_edges)
            self.graph.es["weight"] = edge_weights
            logger.info(f"添加择优边: {len(edges_to_add)} 条。")

        # 写入高相似标签
        for nid, sim_id in similar_labels.items():
            v = self.graph.vs.find(name=nid)
            v["metadata"]["similar_to"] = sim_id
            v["metadata"]["has_similar_label"] = True
        logger.info(f"高相似标签: {len(similar_labels)} 个。")

        logger.info("初始图谱构建完成！")

    # ------------------------------------------------------------------
    # 单节点增量入库
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
        dense_vec = enc["dense"]      # 已归一化
        sparse_dict = enc["sparse"]

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

        # 检索 top_k 候选
        k = min(self.top_k, self.index.ntotal)
        sims, n_ids = self.index.search(dense_vec.reshape(1, -1), k)
        cos_scores = sims[0]
        cand_ids = n_ids[0]

        mask = (cand_ids != node_id) & (cand_ids != -1)
        filt_cos = cos_scores[mask]
        filt_ids = cand_ids[mask].astype(int)
        if len(filt_ids) == 0:
            self.index.add_with_ids(dense_vec.reshape(1, -1), np.array([node_id], dtype=np.int64))
            return True

        # 计算混合得分
        scores = []
        for cid, cos in zip(filt_ids, filt_cos):
            try:
                v = self.graph.vs.find(name=cid)
                c_sparse = v["sparse"]
                lex = sum(sparse_dict[t] * c_sparse[t] for t in set(sparse_dict) & set(c_sparse))
                scores.append(self.alpha * cos + (1 - self.alpha) * lex)
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
            logger.info(f"节点 {node_id} 高相似于 {similar_target}，已打标签。")

        # 连边
        conn_mask = (scores >= self.connect_thresh) & (scores < self.label_thresh)
        if np.any(conn_mask):
            valid_scores = scores.copy()
            valid_scores[~conn_mask] = -np.inf
            best_idx = np.argmax(valid_scores)
            target_id = int(filt_ids[best_idx])
            src_idx = self.graph.vs.find(name=node_id).index
            dst_idx = self.graph.vs.find(name=target_id).index
            self.graph.add_edge(src_idx, dst_idx, weight=float(scores[best_idx]))
            logger.info(f"节点 {node_id} 择优连边至 {target_id}，混合得分 {scores[best_idx]:.4f}")
        else:
            logger.info(f"节点 {node_id} 无合适连边，孤立。")

        # 更新 FAISS
        self.index.add_with_ids(dense_vec.reshape(1, -1), np.array([node_id], dtype=np.int64))
        # 聚类触发（保留原有逻辑）
        self.check_and_trigger_clustering(node_id)
        return True

    # ------------------------------------------------------------------
    # 以下两个方法沿用原有设计，仅保留接口，实际需要您注入对应的 LLM 和 embedding 函数
    # ------------------------------------------------------------------
    def generate_summary_node(self,
                              new_summary_id: int,
                              member_ids: List[int],
                              llm_generate_fn: Callable[[List[str]], str],
                              embedding_fn: Callable[[str], np.ndarray],
                              summary_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        对一组成员节点生成 Summary 节点。
        注意：此处的 embedding_fn 应返回 BGE‑M3 格式的词典（包含 dense 和 sparse），
        或者我们改用 self.encode_text 进行统一编码。
        为保持接口灵活性，这里暂时保留调用，实际使用时建议直接用 self.add_node 配合编码结果。
        """
        texts = []
        for nid in member_ids:
            try:
                v = self.graph.vs.find(name=nid)
                texts.append(v["content"])
            except ValueError:
                continue
        if not texts:
            return False

        summary_text = llm_generate_fn(texts)
        # 使用引擎自身的编码器，保证一致性
        enc = self.encode_text(summary_text)
        summary_meta = summary_metadata or {}
        summary_meta['source_cluster_size'] = len(member_ids)
        summary_meta['is_auto_generated'] = True

        return self.add_node(
            node_id=new_summary_id,
            content=summary_text,
            node_type="Summary",
            metadata=summary_meta
        )

    def check_and_trigger_clustering(self, target_node_id: int) -> bool:
        try:
            v = self.graph.vs.find(name=target_node_id)
        except ValueError:
            return False
        degree = v.degree()
        if degree < self.degree_threshold:
            return False
        logger.info(f"节点 {target_node_id} 连通度 {degree}，触发局部聚类！")
        neighbor_names = [self.graph.vs[n]["name"] for n in self.graph.neighbors(v.index)]
        member_ids = [target_node_id] + neighbor_names
        # 删去中心节点与邻居的连边
        edges_to_del = [(v.index, n) for n in self.graph.neighbors(v.index)]
        self.graph.delete_edges(edges_to_del)
        # 生成 Summary 节点（需要外部事先注入 llm_generate_fn 和 embedding_fn）
        new_id = self._generate_unique_id()
        if hasattr(self, 'llm_generate_fn') and hasattr(self, 'embedding_fn'):
            self.generate_summary_node(
                new_summary_id=new_id,
                member_ids=member_ids,
                llm_generate_fn=self.llm_generate_fn,
                embedding_fn=self.embedding_fn,
                summary_metadata={"trigger_source": target_node_id, "type": "auto_emerged_summary"}
            )
        else:
            logger.warning("LLM/Embedding 函数未注入，跳过 Summary 生成。")
        return True


# ------------------------------------------------------------------
# 使用示例（演示接入）
# ------------------------------------------------------------------
if __name__ == "__main__":
    # 初始化引擎
    engine = LegalDenseGraphBuilder(
        model_name='BAAI/bge-m3',
        use_fp16=True,
        alpha_dense=0.3,        # 保持与原要求一致的 0.3 余弦 + 0.7 BM25（现在用 sparse 替代）
        connect_threshold=0.85,
        label_threshold=0.99,
        top_k_search=50
    )

    # 准备一批法律 chunk（假设从您之前的 pipeline 来）
    sample_nodes = [
        {"id": 1, "content": "《刑法》第232条：故意杀人的，处死刑、无期徒刑或者十年以上有期徒刑。", "type": "article"},
        {"id": 2, "content": "《刑法》第233条：过失致人死亡的，处三年以上七年以下有期徒刑。", "type": "article"},
        {"id": 3, "content": "《刑法》第23条：已经着手实行犯罪，由于犯罪分子意志以外的原因而未得逞的，是犯罪未遂。", "type": "article"},
        {"id": 4, "content": "我被人持刀威胁，为了防卫反击对方导致重伤，这种情况算正当防卫还是过当？", "type": "user_query"},
    ]

    # 批量构建图谱
    engine.build_initial_graph_batch(sample_nodes)

    # 后续可增量添加
    engine.add_node(
        node_id=5,
        content="故意伤害致人重伤的，处三年以上十年以下有期徒刑。",
        node_type="article"
    )

    # 查看当前图信息
    print(f"节点数: {engine.graph.vcount()}, 边数: {engine.graph.ecount()}")
    for v in engine.graph.vs:
        label_info = v["metadata"].get("similar_to")
        print(f"节点 {v['name']}: {v['content'][:30]}... 相似标签: {label_info}")