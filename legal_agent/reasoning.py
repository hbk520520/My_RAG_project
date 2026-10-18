from __future__ import annotations
import logging
from config_loader import cfg
import prompts
logger = logging.getLogger("LegalAgentPlanReplan")
import numpy as np
from typing import Dict, Any
from dataset.graph import LegalDenseGraphBuilder, vname
from .operators import AgenticNodesOperator
class Reasoner:
    """
    融合检索与推理，并判断信息是否足够回答问题。
    依赖外部的法律图谱、Embedding 函数和 LLM 操作器。
    """
    def __init__(self, legal_graph: LegalDenseGraphBuilder, embedding_fn, agentic_ops: AgenticNodesOperator):
        self.legal_graph = legal_graph
        self.embedding_fn = embedding_fn
        self.agentic_ops = agentic_ops
        self.retrieval_trace = None

    def retrieve(self, query: str) -> str:
        """从法律图谱检索相关文档，返回合并后的文本"""
        return "\n---\n".join(hit["content"] for hit in self._retrieve_hits(query))

    def retrieve_records(self, query: str):
        """保留原文来源和时间元数据；因果图不能从语义相似边推导。"""
        from causal_evidence.pipeline import evidence_from_hit
        return [evidence_from_hit(hit) for hit in self._retrieve_hits(query)]

    def _retrieve_hits(self, query: str):
        from dataset.document_navigation import retrieve
        self.retrieval_trace = None
        if not self.legal_graph:
            return []
        self.legal_graph.refresh_memory()
        result = retrieve(self.legal_graph, query, lambda: self._vector_hits(query),
                          lambda messages: self.agentic_ops._call_messages(messages, require_json=True, temperature=0))
        self.retrieval_trace = result.trace
        return result.hits

    def _vector_hits(self, query: str):
        if not self.legal_graph:
            return []
        graph_encoding = (self.legal_graph._memory_store is not None
                          or cfg.get("causal_evidence", "enabled", default=False))
        if not graph_encoding and not self.embedding_fn:
            return []
        k = min(5, self.legal_graph.index.ntotal)
        if k == 0:
            return []
        emb = self.legal_graph.encode_text(query)["dense"] if graph_encoding else self.embedding_fn(query)
        norm_emb = self.legal_graph._l2_normalize(emb).reshape(1, -1).astype(np.float32)
        sims, ids = self.legal_graph.index.search(norm_emb, k)
        docs = []
        for sim, nid in zip(sims[0], ids[0]):
            if nid != -1 and sim > 0.6:
                try:
                    # 阶段 10：图内顶点名恒为 str(int)，FAISS 给的是 int64
                    node = self.legal_graph.graph.vs.find(name=vname(nid))
                    metadata = node["metadata"]
                    docs.append({"id": int(nid), "content": node["content"], "metadata": metadata,
                                 "score": float(sim)})
                except ValueError:
                    pass
        from dataset.long_term_memory import source_evidence_hits
        return source_evidence_hits(self.legal_graph, docs, 5, norm_emb,
                                    original_only=cfg.get("causal_evidence", "enabled", default=False))

    def answer(self, sub_task: str, original_query: str, *, prediction_cutoff=None,
               causal_targets=None) -> Dict[str, Any]:
        if cfg.get("causal_evidence", "enabled", default=False):
            from causal_evidence.pipeline import analyze_configured
            analysis = analyze_configured(
                self.retrieve_records(sub_task), original_query,
                lambda messages: self.agentic_ops._call_messages(messages, require_json=True, temperature=0),
                prediction_cutoff=prediction_cutoff, targets=causal_targets,
            )
            result = self._answer_from_docs(sub_task, original_query, analysis["context"])
            analysis["retrieval_trace"] = self.retrieval_trace
            result["causal_evidence"] = analysis
            return result
        return self._answer_from_docs(sub_task, original_query, self.retrieve(sub_task))

    def _answer_from_docs(self, sub_task: str, original_query: str, docs: str) -> Dict[str, Any]:
        """
        对子任务进行检索、抽取事实、推理，并判断信息是否充足。
        现优先使用 Grader 结构化输出；若无法获得则回退原有逻辑。
        """
        if not docs:
            return {
                "fact": "未检索到相关文档",
                "reasoning": "无法回答",
                "sufficient": False,
                "suggestion": f"请尝试更宽泛的检索词：{sub_task}",
                "grader_status": "irrelevant"      # 新增字段
            }

        # 优先尝试 Grader 结构化评估
        try:
            grade_res = self.agentic_ops.grade_facts(sub_task, docs)
            status = grade_res.get("status", "irrelevant")
            facts = grade_res.get("extracted_facts", [])
            suggestion = grade_res.get("missing_info", "")
            rationale = grade_res.get("rationale", "")

            if status == "sufficient":
                return {
                    "fact": "\n".join(facts),
                    "reasoning": rationale,
                    "sufficient": True,
                    "suggestion": "",
                    "grader_status": status
                }
            elif status == "partial":
                return {
                    "fact": "\n".join(facts),
                    "reasoning": rationale,
                    "sufficient": False,
                    "suggestion": suggestion if suggestion else f"补充检索: {sub_task}",
                    "grader_status": status
                }
            else:  # irrelevant
                return {
                    "fact": "未提取到相关事实",
                    "reasoning": rationale,
                    "sufficient": False,
                    "suggestion": f"当前文档与任务无关，建议虫洞穿越: {sub_task}",
                    "grader_status": status
                }
        except Exception as e:
            if cfg.get("causal_evidence", "enabled", default=False):
                raise
            logger.warning(f"Grader 调用异常，回退旧逻辑: {e}")

        # 回退原有逻辑 (保持兼容)
        facts = self.agentic_ops.extract_facts(sub_task, docs, original_query=original_query)
        reasoning = self.agentic_ops.reason(sub_task, facts)

        sufficient = True
        suggestion = ""
        if "未找到相关事实" in facts or "无法确定" in reasoning:
            sufficient = False
            suggestion = f"当前文档未覆盖“{sub_task}”，建议检索更精准的法律条文。"
        elif len(docs) < 50:
            sufficient = False
            suggestion = "检索到的文档过于简略，请尝试不同的查询表述。"

        return {
            "fact": facts,
            "reasoning": reasoning,
            "sufficient": sufficient,
            "suggestion": suggestion,
            "grader_status": "irrelevant" if not sufficient else "sufficient"
        }

