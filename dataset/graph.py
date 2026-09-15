"""
法律稠密图引擎 —— 知识的结构化仓库
===============================
用 BGE-M3 把每条法条/案例编成稠密+稀疏双向量，存进 FAISS-HNSW 做毫秒级检索，
用 igraph 管理节点间的语义边。新知识入库自动连边，旧法条可软删除并级联标记脏摘要。

核心思路：
  混合相似度 = α × 余弦 + (1-α) × 稀疏词权重（任一侧无稀疏权重时退化为纯余弦）
  检索走 HNSW 近似近邻，建边走阈值筛选，日常增量维护靠 tombstone 和夜间重算。

阶段 4 变更：
  1. 修好 __init__（阶段 2 插入 from_config 时把它截断了，encoder/tombstone_ids/
     dirty_summary_ids/_lora_loaded 全都没初始化）
  2. 补全三个原空实现：build_initial_graph_batch / generate_summary_node /
     check_and_trigger_clustering
  3. BGE-M3 改为**懒加载**：只做批量注入（传预计算向量）或注入自定义编码器时，
     完全不会去拉 2GB 的基座模型
  4. 维度强校验：编码器输出/外部向量与 self.dim 不一致时立即报错，
     而不是等 FAISS 抛出难以定位的异常
  5. 混合相似度在缺少稀疏权重时退化为纯余弦，避免预计算向量路径下
     分数被整体打折、一条边都连不上

技术栈: FlagEmbedding (BGE-M3) / FAISS (HNSW) / igraph / numpy
"""
import os, sys, time, logging, hashlib
import numpy as np
import faiss
import igraph as ig
from typing import Dict, Any, Callable, List, Optional, Set, Sequence

# 允许从任意 cwd 导入根目录模块（config_loader）
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

try:
    from config_loader import cfg
except Exception:  # 允许脱离 config.yaml 单独使用图引擎
    cfg = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegalGraphEngine")


# ============================================================================
# 顶点名规范化（阶段 10）
# ============================================================================
def vname(node_id) -> str:
    """
    节点 ID → igraph 顶点名。

    igraph 要求顶点名是**字符串**。用整数名会带来两个问题：
      1) 每次 `add_vertex` 都打 `DeprecationWarning`，且未来版本会直接禁止；
      2) 一旦与字符串名混用（语料给的是 str ID），`vs.find(name=...)` 会抛
         `ValueError` —— 而所有调用点都写成 `except ValueError: continue`
         （`retriever_worker` / `mount_to_parent` / `_propagate_dirty` …），
         于是变成**静默漏节点**，检索结果莫名其妙变少。

    因此本模块的约定是：
        **图内顶点名恒为 `str(int)`；内部一切逻辑仍用 int ID。**
    只在 igraph 边界处转换，就是 `vname()` 与 `as_num_id()` 这两个函数。
    """
    return str(node_id)


def as_num_id(value) -> int:
    """
    igraph 顶点名（或语料里的 ID）→ 数值 ID。

    FAISS 只接受 `int64` 主键，`build_initial_graph_batch` 会把 ID 塞进
    `np.array(..., dtype=np.int64)` —— 所以「ID 必须是整数」本来就是这个系统的
    既有前提。这里把它显式化：非整数 ID 会抛出可定位的错误，而不是等到
    numpy 转换时报一句难以理解的 `invalid literal for int()`。
    """
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    try:
        return int(text)
    except ValueError as e:
        raise ValueError(
            f"节点 ID {value!r} 不是整数。FAISS 主键必须是 int64，"
            f"因此 igraph 顶点名必须是 str(int)；"
            f"请在 prepare_corpus 阶段就把 ID 规范成整数。"
        ) from e


class LegalDenseGraphBuilder:
    """BGE-M3 + FAISS-HNSW + igraph 的法律知识图谱引擎"""

    def __init__(self,
                 model_name: str = 'BAAI/bge-m3',
                 use_fp16: bool = True,
                 embedding_dim: int = 1024,
                 alpha_dense: float = 0.3,
                 connect_threshold: float = 0.85,
                 label_threshold: float = 0.99,
                 top_k_search: int = 50,
                 degree_threshold: int = 15,
                 max_cluster_size: int = 50,
                 encoder=None):
        """
        :param encoder:          可选外部编码器。传入后完全跳过 BGE-M3 加载
                                 （离线批量注入 / soul.py 路径）。需支持
                                 encode([text], return_dense=True, return_sparse=True)
        :param max_cluster_size: 单个摘要簇的子节点上限。超过即触发"再聚类"标记，
                                 对应 README 里的"逻辑上必须有最大值截断"。
        """
        self.dim = embedding_dim
        self.alpha = alpha_dense
        self.connect_thresh = connect_threshold
        self.label_thresh = label_threshold
        self.top_k = top_k_search
        self.degree_threshold = degree_threshold
        self.max_cluster_size = max_cluster_size

        # ---- BGE-M3：懒加载 ----
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self._encoder = encoder          # 外部注入时直接用它，永不加载 BGE-M3

        # ---- 图引擎 ----
        self.graph = ig.Graph(directed=False)

        # ---- FAISS 索引（归一化向量 + 内积 == 余弦相似度）----
        base_index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.index = faiss.IndexIDMap(base_index)

        # ---- 法律更新维护 ----
        self.tombstone_ids: Set[int] = set()        # 已软删除的节点 ID
        self.dirty_summary_ids: Set[int] = set()    # 需要重算的摘要节点 ID
        self.pending_cluster_seeds: Set[int] = set()  # 度数过高、等待建簇的节点

        # ---- LoRA ----
        self._lora_loaded = False

        # ---- ID 分配兜底 ----
        self._fallback_id_counter = 0

        logger.info(f"图引擎初始化: dim={self.dim}, alpha={self.alpha}, "
                    f"connect={self.connect_thresh}, label={self.label_thresh}")

    # ------------------------------------------------------------------
    # 构造入口
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls) -> "LegalDenseGraphBuilder":
        """从 config.yaml 的 graph.* 段构造图引擎（推荐入口）"""
        if cfg is None:
            raise RuntimeError("config_loader 不可用，无法从配置构造图引擎")
        return cls(
            model_name=cfg.get("graph", "model_name", default="BAAI/bge-m3"),
            use_fp16=cfg.get("graph", "use_fp16", default=True),
            embedding_dim=cfg.get("graph", "embedding_dim", default=1024),
            alpha_dense=cfg.get("graph", "alpha_dense", default=0.3),
            connect_threshold=cfg.get("graph", "connect_threshold", default=0.85),
            label_threshold=cfg.get("graph", "label_threshold", default=0.99),
            top_k_search=cfg.get("graph", "top_k_search", default=50),
            degree_threshold=cfg.get("graph", "degree_threshold", default=15),
        )

    # ------------------------------------------------------------------
    # 编码器（懒加载）
    # ------------------------------------------------------------------
    @property
    def encoder(self):
        """
        懒加载 BGE-M3。只有真正需要"文本→向量"时才加载，
        这样纯批量注入（传预计算向量）的场景不会白白吃掉几个 GB 显存。
        """
        if self._encoder is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as e:
                raise ImportError(
                    "需要 FlagEmbedding 才能做文本编码：pip install FlagEmbedding。\n"
                    "替代方案：用 set_encoder() 注入自定义编码器，"
                    "或用 build_initial_graph_batch(nodes, embeddings) 直接传预计算向量。"
                ) from e
            logger.info(f"正在加载 BGE‑M3 模型: {self.model_name} ...")
            self._encoder = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)
            logger.info("BGE‑M3 模型加载完成。")
        return self._encoder

    def set_encoder(self, encoder) -> None:
        """注入外部编码器（用于测试桩或已自行加载的模型）"""
        self._encoder = encoder

    def _check_dim(self, vector: np.ndarray, source: str) -> None:
        """维度校验：不一致时立刻报错，避免 FAISS 抛出难以定位的异常"""
        if vector.shape[-1] != self.dim:
            raise ValueError(
                f"{source} 输出维度 {vector.shape[-1]} 与图引擎 dim={self.dim} 不一致。"
                f"请核对 config.yaml 的 embedding.dimension 与 graph.embedding_dim 是否一致。"
            )

    # ------------------------------------------------------------------
    # LoRA 权重加载
    # ------------------------------------------------------------------
    def load_lora_weights(self, lora_path: str) -> bool:
        """
        加载微调后的 LoRA 适配器权重到 BGE-M3 编码器。
        对齐 model/training/train_retriever.py 中 peft 的训练产物。

        :param lora_path: LoRA 权重目录路径或 HuggingFace 模型 ID
        :return: 是否加载成功
        """
        try:
            from peft import PeftModel
            import torch  # noqa: F401

            # BGEM3FlagModel 内部使用 AutoModel，通过 .model 访问底层 Transformer
            base_model = self.encoder.model

            logger.info(f"正在加载 LoRA 权重: {lora_path} ...")
            self.encoder.model = PeftModel.from_pretrained(base_model, lora_path)
            # 合并 LoRA 权重到基座（推理加速）
            self.encoder.model = self.encoder.model.merge_and_unload()
            self._lora_loaded = True
            logger.info("LoRA 权重加载并合并完成，检索器已对齐训练产物。")
            return True

        except ImportError as e:
            logger.warning(f"peft 库不可用，跳过 LoRA 加载: {e}（请安装: pip install peft）")
            return False
        except Exception as e:
            logger.warning(f"LoRA 权重加载失败: {e}；回退使用原始 BGE-M3 权重。")
            return False

    def load_lora_from_sentence_transformers(self, st_model_path: str) -> bool:
        """
        备选方案：从 SentenceTransformers 保存的 LoRA 模型加载。
        适用于 train_retriever.py 中用 model.save() 保存的场景。
        """
        try:
            from sentence_transformers import SentenceTransformer
            import torch

            logger.info(f"从 SentenceTransformer 格式加载: {st_model_path} ...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            st_model = SentenceTransformer(st_model_path, device=device)

            # 提取底层 AutoModel 替换 BGEM3FlagModel 的 model
            for module in st_model.modules():
                if hasattr(module, 'auto_model'):
                    self.encoder.model = module.auto_model
                    self._lora_loaded = True
                    logger.info("SentenceTransformer LoRA 模型加载完成。")
                    return True

            logger.warning("无法从 SentenceTransformer 中提取底层模型")
            return False

        except ImportError:
            logger.warning("sentence_transformers 不可用")
            return False
        except Exception as e:
            logger.error(f"SentenceTransformer LoRA 加载失败: {e}")
            return False

    @property
    def is_lora_loaded(self) -> bool:
        """是否已加载微调权重"""
        return self._lora_loaded

    # ------------------------------------------------------------------
    # 编码与相似度
    # ------------------------------------------------------------------
    def encode_text(self, text: str) -> Dict[str, Any]:
        """单条文本 → {dense: 归一化稠密向量, sparse: 稀疏词权重}"""
        out = self.encoder.encode([text], return_dense=True, return_sparse=True)
        dense = np.asarray(out['dense_vecs'][0], dtype=np.float32)
        self._check_dim(dense, "编码器")
        norm = float(np.linalg.norm(dense))
        dense = dense / norm if norm > 0 else dense
        sparse = out['lexical_weights'][0]
        return {"dense": dense.astype(np.float32), "sparse": sparse}

    @staticmethod
    def _l2_normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector

    def _combined_score(self, cos_sim: float,
                        sparse_a: Optional[Dict[str, float]],
                        sparse_b: Optional[Dict[str, float]]) -> float:
        """
        混合相似度 = α×余弦 + (1-α)×稀疏词权重内积。

        关键点：任一侧没有稀疏权重时**退化为纯余弦**。
        否则传入预计算向量（sparse 为空）的场景会得到 0.3×cos，
        全部低于连边阈值，静默建出一张没有边的图。
        """
        cos_sim = float(cos_sim)
        if not sparse_a or not sparse_b:
            return cos_sim
        common = set(sparse_a) & set(sparse_b)
        if not common:
            return cos_sim
        lex = sum(sparse_a[t] * sparse_b[t] for t in common)
        return self.alpha * cos_sim + (1.0 - self.alpha) * lex

    def hybrid_similarity(self,
                          dense_a: np.ndarray, sparse_a: Dict[str, float],
                          dense_b: np.ndarray, sparse_b: Dict[str, float]) -> float:
        """两个节点之间的混合相似度"""
        return self._combined_score(float(np.dot(dense_a, dense_b)), sparse_a, sparse_b)

    def _generate_unique_id(self) -> int:
        """分配新的节点 ID（int），保证不与已有节点冲突"""
        if self.graph.vcount() == 0:
            return 1
        try:
            numeric = [as_num_id(n) for n in self.graph.vs["name"]]
        except (TypeError, ValueError):
            numeric = []
        if numeric:
            return max(numeric) + 1
        # 名字不是纯数字时退回递增计数器
        self._fallback_id_counter = max(self._fallback_id_counter, self.graph.vcount()) + 1
        return self._fallback_id_counter

    # ------------------------------------------------------------------
    # 批量构建初始图谱（阶段 4：补全原空实现）
    # ------------------------------------------------------------------
    def build_initial_graph_batch(self,
                                  nodes_data: List[Dict[str, Any]],
                                  embeddings: Optional[np.ndarray] = None,
                                  search_batch_size: int = 10000) -> None:
        """
        批量构建初始图谱。

        两种入参：
          - embeddings 传入预计算矩阵 (N × dim)：不加载 BGE-M3，适合离线批量注入
          - embeddings 为 None：用内部编码器逐条编码（会触发 BGE-M3 懒加载）

        建边规则：connect_threshold <= 混合相似度 < label_threshold 才连边。
        等于或高于 label_threshold 视为"几乎完全冗余"，不连边而是打 similar_to 标签。
        """
        total_nodes = len(nodes_data)
        if total_nodes == 0:
            logger.warning("build_initial_graph_batch 收到空节点列表，跳过")
            return

        # ---- 1. 准备向量 ----
        if embeddings is None:
            logger.info(f"未提供预计算向量，使用内部编码器逐条编码 {total_nodes} 条…")
            dense_list, sparse_list = [], []
            for n in nodes_data:
                enc = self.encode_text(n["content"])
                dense_list.append(enc["dense"])
                sparse_list.append(enc["sparse"])
            embeddings = np.vstack(dense_list).astype(np.float32)
        else:
            embeddings = np.asarray(embeddings, dtype=np.float32)
            if embeddings.ndim != 2 or embeddings.shape[0] != total_nodes:
                raise ValueError(
                    f"embeddings 形状 {embeddings.shape} 与节点数 {total_nodes} 不匹配"
                )
            self._check_dim(embeddings, "传入的 embeddings")
            # 预计算路径无法得到稀疏权重 -> 相似度自动退化为纯余弦
            sparse_list = [{} for _ in nodes_data]
            logger.info("使用预计算向量建图；缺少稀疏权重，相似度退化为纯余弦")

        # ---- 2. 写入顶点 ----
        # 阶段 10：顶点名统一为 str(int)。语料里的 id 先规范成 int，
        # 保证「FAISS int64 主键」与「igraph 顶点名」是同一套 ID 的两种表示。
        node_ids = [as_num_id(n["id"]) for n in nodes_data]
        self.graph.add_vertices(total_nodes)
        self.graph.vs["name"] = [vname(i) for i in node_ids]
        self.graph.vs["content"] = [n["content"] for n in nodes_data]
        self.graph.vs["type"] = [n["type"] for n in nodes_data]
        self.graph.vs["dense"] = [embeddings[i] for i in range(total_nodes)]
        self.graph.vs["sparse"] = sparse_list

        meta_list = []
        for n in nodes_data:
            m = dict(n.get("metadata") or {})
            m.setdefault("status", "active")
            m.setdefault("parent_id", None)
            meta_list.append(m)
        self.graph.vs["metadata"] = meta_list

        # ---- 3. 写入 FAISS ----
        all_ids = np.array(node_ids, dtype=np.int64)
        self.index.add_with_ids(embeddings, all_ids)
        logger.info(f"FAISS 批量注入完成，共 {total_nodes} 个节点。")

        # ---- 4. 批量建边 ----
        name_to_index = {as_num_id(v["name"]): v.index for v in self.graph.vs}
        edge_keys = {}
        for i in range(0, total_nodes, search_batch_size):
            end = min(i + search_batch_size, total_nodes)
            batch_emb = embeddings[i:end]
            batch_ids = all_ids[i:end]
            k = min(self.top_k, self.index.ntotal)
            sims, n_ids = self.index.search(batch_emb, k)
            for row, query_id in enumerate(batch_ids):
                q_sparse = sparse_list[i + row]
                for cos, target_id in zip(sims[row], n_ids[row]):
                    if target_id == -1 or target_id == query_id:
                        continue
                    if target_id in self.tombstone_ids:
                        continue
                    try:
                        t_sparse = self.graph.vs.find(name=vname(target_id))["sparse"]
                    except ValueError:
                        continue
                    score = self._combined_score(cos, q_sparse, t_sparse)
                    key = tuple(sorted((int(query_id), int(target_id))))
                    if self.connect_thresh <= score < self.label_thresh:
                        if key not in edge_keys or score > edge_keys[key]:
                            edge_keys[key] = float(score)

        if edge_keys:
            pairs = list(edge_keys.keys())
            igraph_edges = [(name_to_index[a], name_to_index[b]) for a, b in pairs]
            self.graph.add_edges(igraph_edges)
            self.graph.es["weight"] = list(edge_keys.values())

        logger.info(f"初始图谱构建完毕：{self.graph.vcount()} 个节点，"
                    f"{self.graph.ecount()} 条边。")

    # ------------------------------------------------------------------
    # 增量添加节点（挂载 + 脏传播 + tombstone 过滤）
    # ------------------------------------------------------------------
    def add_node(self,
                 node_id: int,
                 content: str,
                 node_type: str,
                 metadata: Optional[Dict[str, Any]] = None,
                 dense: Optional[np.ndarray] = None,
                 sparse: Optional[Dict[str, float]] = None) -> bool:
        """
        :param dense:  预计算的稠密向量 (dim,)。传入后**不再**调用编码器取 dense。
        :param sparse: 预计算的稀疏词权重。传入后**不再**调用编码器取 sparse；
                       传空 dict `{}` 表示「已知没有稀疏权重」，此时混合得分
                       退化为纯余弦（见 `_combined_score`）。

        阶段 9 新增这两个参数：此前 `add_node` 强制自己重新编码一次，于是同一个节点
        在 GMM 侧用的是 `embedding_fn` 的向量、在图引擎侧用的是编码器的向量 ——
        两者不一致时**不报错**，静默分叉成两个不同空间的向量。现在调用方可以把
        自己算好的向量透传进来，保证两边是同一个向量，顺带省掉一次编码。
        """
        if metadata is None:
            metadata = {}

        # 只在确有缺失时才编码（两个都给齐了就完全不碰编码器）
        if dense is None or sparse is None:
            enc = self.encode_text(content)
            if dense is None:
                dense = enc["dense"]
            if sparse is None:
                sparse = enc["sparse"]

        self._check_dim(np.asarray(dense), "add_node.dense")
        # 拷一份：调用方（memory_graph_bridge）会把同一个向量同时交给 GMM 与图引擎，
        # 共享同一个 ndarray 会让任何一侧的原地修改波及另一侧。
        dense_vec = np.array(dense, dtype=np.float32, copy=True)
        sparse_dict = sparse

        # 初始化元数据中的状态与父节点
        metadata.setdefault("status", "active")
        metadata.setdefault("parent_id", None)

        # 注册到图
        self.graph.add_vertex(name=vname(node_id),
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
                v = self.graph.vs.find(name=vname(cid))
                scores.append(self._combined_score(cos, sparse_dict, v["sparse"]))
            except ValueError:
                scores.append(0.0)
        scores = np.array(scores)

        # 高相似标签（几乎完全冗余，只打标签不连边）
        high_mask = scores >= self.label_thresh
        if np.any(high_mask):
            best_idx = np.argmax(scores * high_mask)
            similar_target = int(filt_ids[best_idx])
            v_self = self.graph.vs.find(name=vname(node_id))
            v_self["metadata"]["similar_to"] = similar_target
            v_self["metadata"]["has_similar_label"] = True

        # 择优连边
        conn_mask = (scores >= self.connect_thresh) & (scores < self.label_thresh)
        if np.any(conn_mask):
            valid_scores = scores.copy()
            valid_scores[~conn_mask] = -np.inf
            best_idx = np.argmax(valid_scores)
            target_id = int(filt_ids[best_idx])
            src_idx = self.graph.vs.find(name=vname(node_id)).index
            dst_idx = self.graph.vs.find(name=vname(target_id)).index
            self.graph.add_edge(src_idx, dst_idx, weight=float(scores[best_idx]))
            logger.info(f"节点 {node_id} 连边至 {target_id}, 得分 {scores[best_idx]:.4f}")

        # 更新 FAISS
        self.index.add_with_ids(dense_vec.reshape(1, -1), np.array([node_id], dtype=np.int64))

        # 增量挂载与脏传播（Summary 节点本身不需要父节点）
        if node_type != "Summary":
            self.mount_to_parent(node_id)

        self._propagate_dirty(node_id)
        self.check_and_trigger_clustering(node_id)
        return True

    # ------------------------------------------------------------------
    # 软删除
    # ------------------------------------------------------------------
    def tombstone_node(self, node_id: int) -> bool:
        """将指定节点标记为 tombstone，并向上传播脏标记"""
        try:
            v = self.graph.vs.find(name=vname(node_id))
        except ValueError:
            logger.error(f"节点 {node_id} 不存在，无法软删除")
            return False

        if v["metadata"].get("status") == "tombstone":
            logger.info(f"节点 {node_id} 已经是 tombstone 状态")
            return True

        v["metadata"]["status"] = "tombstone"
        self.tombstone_ids.add(node_id)
        logger.info(f"节点 {node_id} 已软删除（tombstone），对前端隐身")

        self._propagate_dirty(node_id)
        return True

    # ------------------------------------------------------------------
    # 增量挂载
    # ------------------------------------------------------------------
    def mount_to_parent(self, child_id: int):
        """为 child_id 节点寻找最相似的 Summary 类型节点作为父节点"""
        try:
            child_v = self.graph.vs.find(name=vname(child_id))
        except ValueError:
            return

        summary_vertices = [
            v for v in self.graph.vs
            if v["type"] == "Summary" and v["metadata"].get("status") != "tombstone"
        ]
        if not summary_vertices:
            return

        child_dense = child_v["dense"]
        child_sparse = child_v["sparse"]
        best_score = -1.0
        best_parent_id = None
        for sv in summary_vertices:
            score = self.hybrid_similarity(child_dense, child_sparse,
                                           sv["dense"], sv["sparse"])
            if score > best_score:
                best_score = score
                # 阶段 10：顶点名是 str(int)，但 metadata 里统一存**数值 ID**，
                # `_propagate_dirty` 就能一路用 int 走 parent_id 链，不必来回转换。
                best_parent_id = as_num_id(sv["name"])

        if best_parent_id is not None and best_score >= self.connect_thresh:
            child_v["metadata"]["parent_id"] = best_parent_id
            logger.info(f"节点 {child_id} 已挂载到父 Summary {best_parent_id} (得分 {best_score:.4f})")
        else:
            logger.info(f"节点 {child_id} 未找到合适的父 Summary，保持无父节点")

    # ------------------------------------------------------------------
    # 脏标记向上传播
    # ------------------------------------------------------------------
    def _propagate_dirty(self, start_node_id: int):
        """从给定节点开始，沿 parent_id 链向上把摘要节点标记为 dirty"""
        current_id = start_node_id
        visited = set()
        while current_id is not None:
            if current_id in visited:
                break  # 防止环路
            visited.add(current_id)
            try:
                v = self.graph.vs.find(name=vname(current_id))
            except ValueError:
                break
            if v["type"] == "Summary":
                v["metadata"]["dirty"] = True
                self.dirty_summary_ids.add(current_id)
                logger.info(f"摘要节点 {current_id} 标记为 dirty")
            parent_id = v["metadata"].get("parent_id")
            if parent_id and parent_id != current_id:
                current_id = parent_id
            else:
                break

    # ------------------------------------------------------------------
    # 计数辅助
    # ------------------------------------------------------------------
    def _count_active_children(self, summary_id: int) -> int:
        """统计挂在某个 Summary 下、且未被软删除的子节点数"""
        count = 0
        for child in self.graph.vs:
            if as_num_id(child["name"]) == summary_id:
                continue
            if child["metadata"].get("parent_id") != summary_id:
                continue
            if child["metadata"].get("status") == "tombstone":
                continue
            count += 1
        return count

    # ------------------------------------------------------------------
    # 聚类触发（阶段 4：补全原空实现）
    # ------------------------------------------------------------------
    def check_and_trigger_clustering(self, target_node_id: int) -> bool:
        """
        检查 target_node_id 所在局部是否需要建簇 / 再聚类，返回是否触发。

        两类触发条件：
          1. 度数过高：非 Summary 节点的边数 >= degree_threshold，
             说明它已成为枢纽，加入 pending_cluster_seeds 等待建簇
          2. 簇规模超限：所在摘要的有效子节点数 >= max_cluster_size，
             打上 dirty 标记，由 nightly_recalc_summaries() 做"最大截断 + 降序再聚类"

        刻意**不**在这里直接做聚类：README 明确记录"构建图时先不管，
        等构建完之后再做多次降序"，避免插入路径上做重活拖慢写入。
        """
        try:
            v = self.graph.vs.find(name=vname(target_node_id))
        except ValueError:
            return False

        triggered = False

        # ---- 条件1：枢纽节点 ----
        if v["type"] != "Summary" and v.degree() >= self.degree_threshold:
            if target_node_id not in self.pending_cluster_seeds:
                self.pending_cluster_seeds.add(target_node_id)
                logger.info(f"节点 {target_node_id} 度数 {v.degree()} >= "
                            f"{self.degree_threshold}，加入待建簇种子")
            triggered = True

        # ---- 条件2：簇规模超过最大值截断上限 ----
        parent_id = v["metadata"].get("parent_id")
        if parent_id is not None:
            child_count = self._count_active_children(parent_id)
            if child_count >= self.max_cluster_size:
                self.dirty_summary_ids.add(parent_id)
                logger.warning(f"摘要 {parent_id} 子节点数 {child_count} >= "
                               f"max_cluster_size={self.max_cluster_size}，"
                               f"触发再聚类标记")
                triggered = True

        return triggered

    # ------------------------------------------------------------------
    # 摘要节点生成（阶段 4：补全原空实现）
    # ------------------------------------------------------------------
    def generate_summary_node(self,
                              child_texts: Sequence[str],
                              llm_generate_fn: Callable[[List[str]], str],
                              child_ids: Optional[Sequence[int]] = None,
                              summary_id: Optional[int] = None,
                              parent_id: Optional[int] = None) -> Optional[int]:
        """
        为一组子节点生成 Summary 节点并写入图与索引。

        README 要求的"最大值截断"在这里落地：
        子节点数超过 max_cluster_size 时只取前 max_cluster_size 条参与本次摘要生成，
        其余留给后续轮次的降序聚类处理 —— 避免单次 LLM 调用上下文爆炸。

        :param child_texts:     参与摘要生成的文本列表
        :param llm_generate_fn: 输入文本列表，返回摘要文本（通常是 LLM 调用）
        :param child_ids:       这些文本对应的节点 ID；提供时会把它们的 parent_id
                                指向新摘要节点。不提供则只建摘要节点、不做挂载。
        :return: 新摘要节点的 ID；失败返回 None
        """
        if not child_texts:
            logger.warning("generate_summary_node 收到空子节点列表，跳过")
            return None

        # ---- 最大值截断 ----
        used_texts = list(child_texts)
        used_ids = list(child_ids) if child_ids else []
        if len(used_texts) > self.max_cluster_size:
            logger.warning(f"子节点数 {len(used_texts)} 超过 max_cluster_size="
                           f"{self.max_cluster_size}，本次只取前 {self.max_cluster_size} 条，"
                           f"其余留待后续降序聚类")
            used_texts = used_texts[:self.max_cluster_size]
            used_ids = used_ids[:self.max_cluster_size]

        # ---- LLM 生成摘要 ----
        summary_text = llm_generate_fn(used_texts)
        if not summary_text or not str(summary_text).strip():
            logger.warning("llm_generate_fn 返回空摘要，跳过本次生成")
            return None
        summary_text = str(summary_text).strip()

        # ---- 分配 ID ----
        if summary_id is None:
            summary_id = self._generate_unique_id()

        # ---- 写入图与索引 ----
        enc = self.encode_text(summary_text)
        self.graph.add_vertex(
            name=vname(summary_id),
            content=summary_text,
            type="Summary",
            metadata={"status": "active", "parent_id": parent_id, "dirty": False},
            dense=enc["dense"],
            sparse=enc["sparse"],
        )
        self.index.add_with_ids(
            enc["dense"].reshape(1, -1), np.array([summary_id], dtype=np.int64))

        # ---- 把参与的子节点挂到新摘要下 ----
        mounted = 0
        for cid in used_ids:
            try:
                child = self.graph.vs.find(name=vname(cid))
            except ValueError:
                continue
            child["metadata"]["parent_id"] = summary_id
            mounted += 1

        self.dirty_summary_ids.discard(summary_id)
        logger.info(f"摘要节点 {summary_id} 生成完成"
                    f"（参与文本 {len(used_texts)} 条，挂载子节点 {mounted} 个）")
        return int(summary_id)

    # ------------------------------------------------------------------
    # 夜间局部重算
    # ------------------------------------------------------------------
    def nightly_recalc_summaries(self, llm_generate_fn: Callable[[List[str]], str]):
        """重新生成所有被标记为 dirty 的 Summary 节点，并清除 dirty 标记"""
        if not self.dirty_summary_ids:
            logger.info("无脏摘要节点，跳过重算")
            return

        logger.info(f"开始夜间局部重算，共 {len(self.dirty_summary_ids)} 个脏摘要节点")
        recalc_list = list(self.dirty_summary_ids)
        self.dirty_summary_ids.clear()

        for summary_id in recalc_list:
            try:
                v = self.graph.vs.find(name=vname(summary_id))
            except ValueError:
                continue
            if v["type"] != "Summary":
                continue

            # 收集所有直接子节点
            child_texts = []
            for child in self.graph.vs:
                if (child["metadata"].get("parent_id") == summary_id
                        and child["metadata"].get("status") != "tombstone"):
                    child_texts.append(child["content"])
            if not child_texts:
                logger.warning(f"摘要节点 {summary_id} 无有效子节点，跳过重算")
                continue

            # 超过上限时做最大截断（同 generate_summary_node 的口径）
            if len(child_texts) > self.max_cluster_size:
                logger.warning(f"摘要 {summary_id} 子节点 {len(child_texts)} 条超过上限，"
                               f"本次只重算前 {self.max_cluster_size} 条")
                child_texts = child_texts[:self.max_cluster_size]

            new_summary = llm_generate_fn(child_texts)
            if not new_summary:
                logger.warning(f"摘要节点 {summary_id} 重算返回空，保留原内容")
                continue

            enc = self.encode_text(str(new_summary).strip())
            v["content"] = str(new_summary).strip()
            v["dense"] = enc["dense"]
            v["sparse"] = enc["sparse"]
            v["metadata"]["dirty"] = False
            logger.info(f"摘要节点 {summary_id} 重算完成")

            # 说明：FAISS 不支持原地更新向量，这里不重写索引。
            # 摘要节点通常由子节点检索后汇聚，不参与初始近邻检索；
            # 若确有需要，应重建索引（成本高，放到离线任务里做）。


# ------------------------------------------------------------------
# 离线编码器（测试 / demo / 无网环境共用）
# ------------------------------------------------------------------
class StubEncoder:
    """
    可复现的假编码器：**同一文本永远得到同一向量**，不同文本得到不同向量。

    用途（阶段 8 提取为公共类，原先只在 graph.py 的 __main__ 里定义了一份）：
      - 让 demo / 单元测试在**没有网络、不下 BGE-M3（约 2GB）**的情况下跑完整链路
      - 便于断言"同一节点在两个子系统里存的是同一个向量"

    实现要点：向量由 `blake2b(text)` 派生种子后采样，而不是用一个共享的
    RandomState 顺序取样 —— 后者会让「同一文本第二次编码」得到不同向量，
    任何"编码两次结果应一致"的断言都会失败（而且它并不等价于真实编码器）。

    接口与 `BGEM3FlagModel.encode()` 对齐：
        encode(texts, return_dense=True, return_sparse=True)
          -> {"dense_vecs": (n, dim) float32, "lexical_weights": [{token: weight}, ...]}
    """

    def __init__(self, dim: int = 1024, seed: int = 42):
        self.dim = dim
        self.seed = seed

    def _vec(self, text: str) -> np.ndarray:
        digest = hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "big") % (2 ** 32)
        rng = np.random.RandomState((self.seed * 1000003 + h) % (2 ** 32))
        v = rng.randn(self.dim).astype(np.float32)
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v

    def encode(self, texts, return_dense=True, return_sparse=True):
        dense = (np.stack([self._vec(t) for t in texts]) if len(texts)
                 else np.zeros((0, self.dim), dtype=np.float32))
        return {
            "dense_vecs": dense.astype(np.float32),
            "lexical_weights": [{str(t)[:4]: 0.5} for t in texts],
        }


def make_offline_engine():
    """
    构造一个已注入 `StubEncoder` 的图引擎（配置全部来自 config.yaml 的 graph.* 段）。

    等价于：
        engine = LegalDenseGraphBuilder.from_config()
        engine.set_encoder(StubEncoder(engine.dim))

    供 demo / 测试使用，确保永远不会触发 BGE-M3 下载。
    """
    engine = LegalDenseGraphBuilder.from_config()
    engine.set_encoder(StubEncoder(engine.dim))
    return engine


# ------------------------------------------------------------------
# 使用示例（离线可跑：全部使用预计算向量，不加载 BGE-M3）
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys as _sys
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # 用公共 StubEncoder，避免加载 BGE-M3（约 2GB）
    engine = make_offline_engine()

    nodes = [
        {"id": 101, "content": "旧法条：故意杀人，处十年以上有期徒刑。", "type": "article", "metadata": {}},
        {"id": 102, "content": "新法条：故意杀人，处死刑、无期徒刑或十年以上有期徒刑。", "type": "article", "metadata": {}},
        {"id": 200, "content": "Summary 节点：暴力犯罪量刑标准", "type": "Summary", "metadata": {}},
    ]
    embs = np.vstack([engine.encode_text(n["content"])["dense"] for n in nodes])
    engine.build_initial_graph_batch(nodes, embeddings=embs)
    print(f"建图结果: 节点 {engine.graph.vcount()}, 边 {engine.graph.ecount()}")

    # 手动建立父子关系（metadata 里统一存数值 ID）
    engine.graph.vs.find(name=vname(101))["metadata"]["parent_id"] = 200
    engine.graph.vs.find(name=vname(102))["metadata"]["parent_id"] = 200

    # 软删除旧法条 → 摘要 200 应被标记为 dirty
    engine.tombstone_node(101)
    print("脏摘要节点集合:", engine.dirty_summary_ids)

    engine.nightly_recalc_summaries(lambda texts: "根据最新法条，暴力犯罪量刑已更新。")
    print("Summary 200 新内容:", engine.graph.vs.find(name=vname(200))["content"])

    # 演示摘要生成（含最大值截断）
    engine.max_cluster_size = 2
    sid = engine.generate_summary_node(
        ["a", "b", "c", "d"],
        llm_generate_fn=lambda texts: f"合并摘要: {'; '.join(texts)}",
        child_ids=[101, 102],
    )
    print(f"生成摘要节点 ID={sid}")
