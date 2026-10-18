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

    def retrieve(self, query: str, *, engine_kind="GRAPH_TRAVERSAL") -> str:
        """从法律图谱检索相关文档，返回合并后的文本"""
        return "\n---\n".join(hit["content"] for hit in self._retrieve_hits(query, engine_kind))

    def retrieve_records(self, query: str, *, engine_kind="GRAPH_TRAVERSAL"):
        """保留原文来源和时间元数据；因果图不能从语义相似边推导。"""
        from causal_evidence.pipeline import evidence_from_hit
        return [evidence_from_hit(hit) for hit in self._retrieve_hits(query, engine_kind)]

    def _retrieve_hits(self, query: str, engine_kind="GRAPH_TRAVERSAL"):
        from agent_runtime.context import tool_call
        def read():
            hits = self._read_hits(query, engine_kind)
            return {"hits": hits, "trace": self.retrieval_trace}
        result = tool_call("retrieval", {"query": query, "engine": engine_kind}, read)
        self.retrieval_trace = result["trace"]
        return result["hits"]

    def _read_hits(self, query, engine_kind):
        from dataset.document_navigation import retrieve
        self.retrieval_trace = None
        if not self.legal_graph:
            return []
        from agent_runtime.context import current_run
        if current_run.get() is None:
            self.legal_graph.refresh_memory()
        result = retrieve(self.legal_graph, query, lambda: self._vector_hits(query, engine_kind),
                          lambda messages: self.agentic_ops._call_messages(messages, require_json=True, temperature=0))
        self.retrieval_trace = result.trace
        return result.hits

    def _vector_hits(self, query: str, engine_kind="GRAPH_TRAVERSAL"):
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
        if engine_kind == "GRAPH_TRAVERSAL":
            from .retrieval import _graph_expand
            docs.extend(_graph_expand(self.legal_graph, docs, k))
        elif engine_kind != "GLOBAL_DENSE_WORMHOLE":
            raise ValueError(f"Unknown retrieval engine: {engine_kind}")
        from dataset.long_term_memory import source_evidence_hits
        return source_evidence_hits(self.legal_graph, docs, 5, norm_emb,
                                    original_only=cfg.get("causal_evidence", "enabled", default=False))

    def answer(self, sub_task: str, original_query: str, *, prediction_cutoff=None,
               causal_targets=None, engine_kind="GRAPH_TRAVERSAL", user_facts=None) -> Dict[str, Any]:
        supplied = "\nUser-supplied case facts (not legal source evidence):\n" + "\n".join(user_facts) if user_facts else ""
        if cfg.get("causal_evidence", "enabled", default=False):
            from causal_evidence.pipeline import analyze_configured
            analysis = analyze_configured(
                (self.retrieve_records(sub_task) if engine_kind == "GRAPH_TRAVERSAL" else
                 self.retrieve_records(sub_task, engine_kind=engine_kind)), original_query,
                lambda messages: self.agentic_ops._call_messages(messages, require_json=True, temperature=0),
                prediction_cutoff=prediction_cutoff, targets=causal_targets,
            )
            result = self._answer_from_docs(sub_task, original_query, analysis["context"] + supplied)
            analysis["retrieval_trace"] = self.retrieval_trace
            result["causal_evidence"] = analysis
            return result
        docs = self.retrieve(sub_task) if engine_kind == "GRAPH_TRAVERSAL" else self.retrieve(sub_task, engine_kind=engine_kind)
        return self._answer_from_docs(sub_task, original_query, docs + supplied)

    def _answer_from_docs(self, sub_task, original_query, docs):
        if not docs:
            return {"fact": "未检索到相关文档", "reasoning": "无法回答", "sufficient": False,
                    "suggestion": f"请尝试更宽泛的检索词：{sub_task}", "grader_status": "irrelevant"}
        grade = self.agentic_ops.grade_facts(sub_task, docs)
        return {"fact": "\n".join(grade.get("extracted_facts", [])),
                "reasoning": grade.get("rationale", ""), "sufficient": grade["status"] == "sufficient",
                "suggestion": grade.get("missing_info") or f"补充检索: {sub_task}",
                "missing_info_kind": grade.get("missing_info_kind", "evidence"),
                "grader_status": grade["status"]}
