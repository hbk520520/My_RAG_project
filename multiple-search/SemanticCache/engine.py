import numpy as np
from typing import Optional, Tuple

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

    def __init__(self, dim: int, threshold: float = 0.98):
        self.dim = dim
        self.threshold = threshold
        self.vectors: list[np.ndarray] = []
        self.answers: list[str] = []

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

    def __init__(self, dim: int, threshold: float = 0.98,
                 redis_url: str = "redis://localhost:6379"):
        if not REDIS_AVAILABLE:
            raise ImportError("请先安装 redis 和 redisvl: pip install redis redisvl")
        self.dim = dim
        self.threshold = threshold
        self.client = redis.from_url(redis_url)
        self._init_index()

    def _init_index(self):
        """创建或加载索引"""
        schema = {
            "index": {
                "name": self.INDEX_NAME,
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
            self.index = SearchIndex.from_existing(
                name=self.INDEX_NAME, redis_url=self.client.connection_pool.connection_kwargs["host"]
            )

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