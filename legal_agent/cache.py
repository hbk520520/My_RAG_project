from __future__ import annotations
import logging
from config_loader import cfg
import prompts
logger = logging.getLogger("LegalAgentPlanReplan")
import numpy as np
from typing import Optional, Dict, Any
class SemanticCache:
    """
    语义缓存层，在 Router 之前执行。
    命中则直接返回 answer，未命中返回 None。
    """

    def __init__(self, vector_store):
        self.store = vector_store

    def lookup(self, query_vector: np.ndarray) -> Optional[str]:
        """查找缓存，命中返回答案，否则返回 None"""
        return self.store.search(query_vector)

    def add(self, query_vector: np.ndarray, answer: str):
        """将新问答对加入缓存"""
        self.store.store(query_vector, answer)


class UnifiedQueryRouter_Soul:
    """
    soul.py 专用路由包装器，集成语义缓存。
    委托给 query.py 的 UnifiedQueryRouter_Query 做实际路由，
    本层仅负责缓存拦截和写回。
    """

    def __init__(self, cache: Optional[SemanticCache] = None, query_router=None):
        self.cache = cache
        # 延迟导入避免循环依赖
        if query_router is None:
            from query import UnifiedQueryRouter_Query
            self._router = UnifiedQueryRouter_Query()
        else:
            self._router = query_router

    def process(self, query: str, *, prediction_cutoff=None, causal_targets=None) -> Dict[str, Any]:
        causal_mode = cfg.get("causal_evidence", "enabled", default=False)
        use_cache = not (causal_mode or cfg.get("long_term_memory", "enabled", default=False))
        # 1. 语义缓存拦截
        # 旧缓存没有来源/截止时间/记忆版本，不能用于这两种模式。
        if self.cache and use_cache:
            q_vec = self._router.embedder.encode(query, normalize_embeddings=True)
            cached_answer = self.cache.lookup(q_vec)
            if cached_answer is not None:
                logger.info("⚡ 语义缓存命中，跳过后续流程")
                return {
                    "intent": "CACHED",
                    "status": "cached_response",
                    "response": cached_answer
                }

        # 2. 委托给 query.py 的路由逻辑
        context = {}
        if prediction_cutoff is not None:
            context["prediction_cutoff"] = prediction_cutoff
        if causal_targets is not None:
            context["causal_targets"] = causal_targets
        result = self._router.process(query, **context)

        # 3. 写回缓存：对简单问答和复杂任务的结果进行缓存
        if self.cache and use_cache and result.get("status") in ("simple_rag", "agent_execution"):
            answer = result.get("response") or result.get("agent_report", "")
            if answer:
                self.cache.add(
                    self._router.embedder.encode(query, normalize_embeddings=True),
                    answer
                )

        return result

    def route(self, query: str) -> Dict[str, Any]:
        """透传路由方法"""
        return self._router.route(query)
