"""
语义缓存 —— 一样的问题别查两遍
===========================
用户问过的问题+答案存进向量库，下次相似问题来了直接返回。
原型用内存，生产用 RedisVL（带余弦相似度阈值）。

技术栈: numpy / redis (RedisVL) 可选
"""
import os, sys, logging
import numpy as np
from typing import Optional, Tuple

# 允许从任意 cwd 导入根目录模块（config_loader）
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from config_loader import cfg

logger = logging.getLogger("SemanticCache")

# 可选依赖：pip install redis redisvl
try:
    import redis
    from redisvl.index import SearchIndex
    from redisvl.query import VectorQuery
    from redisvl.query.filter import Tag
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class InMemoryVectorStore:
    """内存向量存储，用于原型验证。生产请切换至 Redis。"""

    def __init__(self, dim: int = None, threshold: float = None):
        # 阶段 2：默认值来自 config.yaml，不再写死 0.98
        self.dim = dim if dim is not None else cfg.get("embedding", "dimension", default=1024)
        self.threshold = threshold if threshold is not None else cfg.get("cache", "threshold", default=0.98)
        self.vectors: list[np.ndarray] = []
        self.answers: list[str] = []

    @classmethod
    def from_config(cls) -> "InMemoryVectorStore":
        """从 config.yaml 构造（阶段 2 新增）"""
        return cls(
            dim=cfg.get("embedding", "dimension", default=1024),
            threshold=cfg.get("cache", "threshold", default=0.98),
        )

    def search(self, query_vec: np.ndarray) -> Optional[str]:
        """返回最匹配答案，若无满足阈值则返回 None"""
        if not self.vectors:
            return None
        # 计算余弦相似度（向量已归一化）
        sims = np.dot(np.array(self.vectors), query_vec)
        best_idx = np.argmax(sims)
        best_sim = sims[best_idx]
        if best_sim >= self.threshold:
            return self.answers[best_idx]
        return None

    def store(self, query_vec: np.ndarray, answer: str):
        """存储一次问答对"""
        self.vectors.append(query_vec)
        self.answers.append(answer)


class RedisVectorStore:
    """基于 RedisVL 的向量存储，需要 redis-stack-server"""

    INDEX_NAME = "legal_semantic_cache"
    PREFIX = "cache:"
    VECTOR_FIELD = "embedding"
    ANSWER_FIELD = "answer"

    def __init__(self, dim: int = None, threshold: float = None,
                 redis_url: str = None, index_name: str = None):
        if not REDIS_AVAILABLE:
            raise ImportError("请先安装 redis 和 redisvl: pip install redis redisvl")
        # 阶段 2：默认值全部来自 config.yaml
        #   cache.threshold / cache.index_name / redis.url / embedding.dimension
        self.dim = dim if dim is not None else cfg.get("embedding", "dimension", default=1024)
        self.threshold = threshold if threshold is not None else cfg.get("cache", "threshold", default=0.98)
        self.index_name = index_name or cfg.get("cache", "index_name", default=self.INDEX_NAME)
        self.redis_url = redis_url or cfg.get("redis", "url", default="redis://localhost:6379")
        self.client = redis.from_url(self.redis_url)
        self._init_index()

    @classmethod
    def from_config(cls) -> "RedisVectorStore":
        """从 config.yaml 构造（阶段 2 新增）"""
        return cls()

    def _init_index(self):
        """创建或加载索引"""
        schema = {
            "index": {
                "name": self.index_name,
                "prefix": self.PREFIX,
            },
            "fields": [
                {"name": "id", "type": "tag"},
                {"name": self.ANSWER_FIELD, "type": "text"},
                {
                    "name": self.VECTOR_FIELD,
                    "type": "vector",
                    "attrs": {
                        "dims": self.dim,
                        "distance_metric": "cosine",
                        "algorithm": "flat",
                    },
                },
            ],
        }
        try:
            self.index = SearchIndex.from_dict(schema)
            self.index.create(overwrite=False)
        except Exception:
            # 顺带修掉原缺陷：原先把裸 host 当作 redis_url 传入
            self.index = SearchIndex.from_existing(name=self.index_name, redis_url=self.redis_url)

    def search(self, query_vec: np.ndarray) -> Optional[str]:
        """向量搜索，返回最匹配答案或 None"""
        vq = VectorQuery(
            vector=query_vec.tolist(),
            vector_field_name=self.VECTOR_FIELD,
            return_fields=[self.ANSWER_FIELD],
            num_results=1,
        )
        results = self.index.query(vq)
        if results and results[0].get("vector_distance", 2.0) <= (1 - self.threshold):
            return results[0][self.ANSWER_FIELD]
        return None

    def store(self, query_vec: np.ndarray, answer: str):
        """将问答对存入缓存"""
        key = f"{self.PREFIX}{np.random.randint(0, int(1e9))}"
        payload = {
            "id": key.split(":")[-1],
            self.ANSWER_FIELD: answer,
            self.VECTOR_FIELD: query_vec.tolist(),
        }
        self.client.json().set(key, "$", payload)
        self.client.expire(key, 3600 * 24 * 30)  # 30 天过期


# ============================================================================
# 统一入口（阶段 2 新增）
# ============================================================================
def build_semantic_cache():
    """
    按 config.yaml 的 cache.engine 选择实现，避免调用方自己拼参数：
        "redis"  -> RedisVectorStore（需要 redis-stack-server）
        "memory" -> InMemoryVectorStore
    redis 库不可用或连接失败时自动降级到内存实现，并打 WARNING。
    """
    engine = (cfg.get("cache", "engine", default="memory") or "memory").lower()

    if engine == "redis":
        if not REDIS_AVAILABLE:
            logger.warning("cache.engine=redis 但未安装 redis/redisvl，降级为内存缓存")
        else:
            try:
                return RedisVectorStore.from_config()
            except Exception as e:
                logger.warning(f"Redis 语义缓存初始化失败({e})，降级为内存缓存")

    return InMemoryVectorStore.from_config()