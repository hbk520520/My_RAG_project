import os
import json
import time
import operator
import logging
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from pydantic import BaseModel, ValidationError
from openai import OpenAI
import numpy as np
import faiss
import igraph as ig
from langgraph.graph import StateGraph, END

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegalAgentFullSystem")

# ============================================================================
# 第一部分：法律稠密图引擎（igraph + FAISS）
# ============================================================================
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

        # 图引擎：无向图
        self.graph = ig.Graph(directed=False)

        # 向量引擎：HNSW + ID映射
        base_index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.index = faiss.IndexIDMap(base_index)

        logger.info(f"Graph engine init: dim={self.dim}, M=32, metric=IP")

    # ------------------------------------------------------------------
    def _l2_normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def _generate_unique_id(self) -> int:
        if self.graph.vcount() == 0:
            return 1
        return max(self.graph.vs["name"]) + 1

    # ------------------------------------------------------------------
    def build_initial_graph_batch(self,
                                  nodes_data: List[Dict[str, Any]],
                                  embeddings: np.ndarray,
                                  search_batch_size: int = 10000) -> None:
        total_nodes = len(nodes_data)
        if total_nodes != embeddings.shape[0]:
            raise ValueError("节点数量与矩阵维度不匹配！")

        logger.info(f"开始批量构建图谱，总节点数: {total_nodes}")

        # 写入顶点
        self.graph.add_vertices(total_nodes)
        self.graph.vs["name"] = [n["id"] for n in nodes_data]
        self.graph.vs["content"] = [n["content"] for n in nodes_data]
        self.graph.vs["type"] = [n["type"] for n in nodes_data]
        self.graph.vs["metadata"] = [n["metadata"] for n in nodes_data]

        # 批量注入向量
        all_ids = np.array([n["id"] for n in nodes_data], dtype=np.int64)
        self.index.add_with_ids(embeddings.astype(np.float32), all_ids)
        logger.info("FAISS 索引批量注入完成。")

        # 分块连边
        all_edges, all_weights = [], []
        logger.info("开始拓扑连边推演...")
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

        # 去重后批量添加边
        unique_edges = {}
        for edge, w in zip(all_edges, all_weights):
            sorted_edge = tuple(sorted(edge))
            if sorted_edge not in unique_edges:
                unique_edges[sorted_edge] = w

        edges_list = list(unique_edges.keys())
        weights_list = list(unique_edges.values())

        name_to_index = {v["name"]: v.index for v in self.graph.vs}
        igraph_edges = [(name_to_index[e[0]], name_to_index[e[1]]) for e in edges_list]
        self.graph.add_edges(igraph_edges)
        self.graph.es["weight"] = weights_list

        logger.info(f"初始图谱构建完毕：{self.graph.vcount()} 个节点，{self.graph.ecount()} 条边。")

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

        # 去重防线
        if self.index.ntotal > 0:
            sims, n_ids = self.index.search(norm_emb, 1)
            if sims[0][0] >= self.dedup_threshold and n_ids[0][0] != -1:
                logger.warning(f"丢弃重复节点 {node_id}")
                return False

        self.graph.add_vertex(name=node_id,
                              content=content,
                              type=node_type,
                              metadata=metadata)

        # 动态连边
        if self.index.ntotal > 0:
            k = min(self.top_k, self.index.ntotal)
            similarities, neighbor_ids = self.index.search(norm_emb, k)
            edges_to_add, weights_to_add = [], []
            for sim, n_id in zip(similarities[0], neighbor_ids[0]):
                if (self.threshold <= sim < self.dedup_threshold
                        and n_id != -1 and n_id != node_id):
                    edges_to_add.append((node_id, int(n_id)))
                    weights_to_add.append(float(sim))
            if edges_to_add:
                name_to_idx = {v["name"]: v.index for v in self.graph.vs}
                ig_edges = [(name_to_idx[s], name_to_idx[d]) for s, d in edges_to_add]
                self.graph.add_edges(ig_edges)
                self.graph.es[-len(ig_edges):]["weight"] = weights_to_add

        self.index.add_with_ids(norm_emb, np.array([node_id], dtype=np.int64))
        return True

    # ------------------------------------------------------------------
    def generate_summary_node(self,
                              new_summary_id: int,
                              member_ids: List[int],
                              llm_generate_fn,
                              embedding_fn,
                              summary_metadata: Optional[Dict] = None) -> bool:
        texts = []
        for nid in member_ids:
            try:
                v = self.graph.vs.find(name=nid)
                texts.append(v["content"])
            except ValueError:
                continue
        if not texts:
            return False

        summary = llm_generate_fn(texts)
        emb = embedding_fn(summary)
        if summary_metadata is None:
            summary_metadata = {}
        summary_metadata['source_cluster_size'] = len(member_ids)
        summary_metadata['is_auto_generated'] = True
        return self.add_node(new_summary_id, summary, "Summary", emb, summary_metadata)

    # ------------------------------------------------------------------
    def check_and_trigger_clustering(self, target_node_id: int):
        # 简化版，实际使用需要注入 llm/embedding 函数
        pass


# ============================================================================
# 第二部分：LLM 基座与节点操作器（Meta‑Planner/Extractor/Reasoner/Generator）
# ============================================================================
class LegalLLMBase:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = "deepseek-chat"

    def _call_llm(self,
                  system_prompt: str,
                  user_prompt: str,
                  require_json: bool = False,
                  temperature: float = 0.1) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response_format = {"type": "json_object"} if require_json else None
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                response_format=response_format
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM 调用失败: {e}")
            return "{}" if require_json else f"系统异常: {str(e)}"


class ExecutionPlan(BaseModel):
    strategy_queue: List[str]


class AgenticNodesOperator(LegalLLMBase):
    """所有LLM节点的业务逻辑封装"""

    # --- Meta‑Planner ---
    def generate_plan(self, user_query: str) -> List[str]:
        system_prompt = """你是顶级法律案件拆解专家（Meta-Planner）。面对用户的复杂法律问题，只生成解决步骤的抽象列表，不要回答。严格输出JSON: {"strategy_queue": ["步骤1", "步骤2", ...]}"""
        user_prompt = f"用户问题：{user_query}"
        raw = self._call_llm(system_prompt, user_prompt, require_json=True)
        try:
            data = json.loads(raw)
            plan = ExecutionPlan(**data)
            return plan.strategy_queue
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Meta-Planner输出解析失败: {e}\n原始内容: {raw}")
            return [user_query]   # 降级

    # --- Controller（纯规则路由）---
    @staticmethod
    def controller_route_decision(sub_task: str) -> str:
        math_keywords = ["计算", "核算", "多少钱", "数额", "赔偿金", "违约金", "金额", "倍数", "天数"]
        if any(kw in sub_task for kw in math_keywords):
            return "write_code"
        return "extract_and_reason"

    # --- Extractor ---
    def extract_facts(self,
                      current_sub_task: str,
                      raw_retrieved_docs: str,
                      original_query: Optional[str] = None) -> str:
        system_prompt = """你是法律事实提取器（Extractor）。只从给定文档中抽取与子任务直接相关的原子事实，禁止推理。若无相关信息，回复“未找到相关事实”。"""
        user_prompt = f"子任务：{current_sub_task}\n文档：\n{raw_retrieved_docs}"
        if original_query:
            user_prompt += f"\n[最高指令：确保不偏离原始诉求 -> {original_query}]"
        return self._call_llm(system_prompt, user_prompt)

    # --- Reasoner ---
    def reason(self, current_sub_task: str, extracted_facts: str) -> str:
        system_prompt = """你是法官助理（Reasoner）。严格基于给定事实对子任务进行逻辑推演，不得引入外部知识。"""
        user_prompt = f"子任务：{current_sub_task}\n事实：{extracted_facts}"
        return self._call_llm(system_prompt, user_prompt)

    # --- Generator ---
    def generate_final_report(self, user_query: str, accumulated_context: List[Dict]) -> str:
        system_prompt = """你是资深律师（Generator）。结合已验证的上下文证据链，生成专业、直接回答用户问题的法律意见书。标记计算出的金额。"""
        context_str = ""
        for item in accumulated_context:
            context_str += f"[Hop {item.get('hop')}] {item.get('sub_task', '计算')} -> {item.get('reasoning', item.get('data', ''))}\n\n"
        user_prompt = f"用户问题：{user_query}\n证据链：\n{context_str}"
        return self._call_llm(system_prompt, user_prompt, temperature=0.3)


# ============================================================================
# 第三部分：LangGraph 状态机定义
# ============================================================================
class AgentState(TypedDict):
    user_query: str
    current_hop: int
    execution_plan: List[str]
    current_sub_task: str
    current_context: Annotated[List[Dict[str, Any]], operator.add]
    next_action: str
    generated_code: str
    code_execution_result: str


# 节点实现（依赖注入用闭包）
def make_nodes(legal_graph: LegalDenseGraphBuilder,
               embedding_fn,
               agentic_ops: AgenticNodesOperator):
    """返回一个包含所有节点函数的字典，供 LangGraph 使用"""

    def node_meta_planner(state: AgentState) -> dict:
        logger.info("Meta-Planner 启动")
        plan = agentic_ops.generate_plan(state["user_query"])
        return {"execution_plan": plan, "current_hop": 0, "next_action": "controller"}

    def node_controller(state: AgentState) -> dict:
        plan = state.get("execution_plan", [])
        hop = state.get("current_hop", 0)
        if hop >= 3 or not plan:
            logger.warning(f"熔断或计划结束 (hop={hop})，转向生成")
            return {"next_action": "generate"}
        current_task = plan[0]
        remaining = plan[1:]
        action = AgenticNodesOperator.controller_route_decision(current_task)
        logger.info(f"Controller 分发: {current_task} -> {action}")
        return {
            "execution_plan": remaining,
            "current_sub_task": current_task,
            "next_action": action,
            "current_hop": hop + 1
        }

    def node_extract_and_reason(state: AgentState) -> dict:
        hop = state["current_hop"]
        query = state["user_query"]
        sub_task = state["current_sub_task"]

        # 1. 图谱检索
        retrieved_docs = []
        if legal_graph and embedding_fn:
            sub_emb = embedding_fn(sub_task)
            norm_emb = legal_graph._l2_normalize(sub_emb).reshape(1, -1).astype(np.float32)
            k = min(5, legal_graph.index.ntotal)
            if k > 0:
                sims, ids = legal_graph.index.search(norm_emb, k)
                for sim, nid in zip(sims[0], ids[0]):
                    if nid != -1 and sim > 0.6:
                        try:
                            node = legal_graph.graph.vs.find(name=int(nid))
                            retrieved_docs.append(node["content"])
                        except ValueError:
                            pass
        raw_docs = "\n---\n".join(retrieved_docs) if retrieved_docs else "（无检索结果）"

        # 2. 调用 Extractor（防漂移每2跳注入原问题）
        original = query if hop % 2 == 0 else None
        facts = agentic_ops.extract_facts(sub_task, raw_docs, original_query=original)

        # 3. 调用 Reasoner
        reasoning = agentic_ops.reason(sub_task, facts)

        entry = {
            "hop": hop,
            "sub_task": sub_task,
            "extracted_facts": facts,
            "reasoning": reasoning,
            "sources": retrieved_docs
        }
        return {"current_context": [entry], "next_action": "controller"}

    def node_write_code(state: AgentState) -> dict:
        sub_task = state["current_sub_task"]
        code = f"# 模拟计算代码 for {sub_task}\nresult = 10000 * 2.5"   # Mock
        return {"generated_code": code, "next_action": "execute_code"}

    def node_execute_code(state: AgentState) -> dict:
        code = state.get("generated_code", "")
        logger.info(f"执行代码:\n{code}")
        try:
            local_vars = {}
            exec(code, {}, local_vars)
            result = str(local_vars.get('result', '无返回'))
            context = {"hop": state["current_hop"], "sub_task": "代码执行", "data": result, "status": "success"}
            action = "controller"
        except Exception as e:
            logger.error(f"代码执行错误: {e}")
            context = {"hop": state["current_hop"], "sub_task": "代码执行", "data": str(e), "status": "error"}
            action = "write_code"   # 重试
        return {"current_context": [context], "code_execution_result": context["data"], "next_action": action}

    def node_generate(state: AgentState) -> dict:
        logger.info("最终生成法律意见书")
        report = agentic_ops.generate_final_report(state["user_query"], state["current_context"])
        # 把最终报告存入上下文
        return {"current_context": [{"hop": "final", "sub_task": "总结", "reasoning": report}], "next_action": "end"}

    return {
        "MetaPlanner": node_meta_planner,
        "Controller": node_controller,
        "ExtractAndReason": node_extract_and_reason,
        "WriteCode": node_write_code,
        "ExecuteCode": node_execute_code,
        "Generate": node_generate
    }


# ============================================================================
# 第四部分：组装 LangGraph 状态机
# ============================================================================
def build_legal_agent(legal_graph: LegalDenseGraphBuilder,
                      embedding_fn,
                      agentic_ops: AgenticNodesOperator):
    nodes = make_nodes(legal_graph, embedding_fn, agentic_ops)

    builder = StateGraph(AgentState)

    # 注册节点
    for name, func in nodes.items():
        builder.add_node(name, func)

    # 边与路由
    builder.set_entry_point("MetaPlanner")
    builder.add_edge("MetaPlanner", "Controller")

    def route_controller(state: AgentState) -> str:
        action = state.get("next_action", "Generate")
        return action if action in nodes else "Generate"

    builder.add_conditional_edges("Controller", route_controller)

    # 执行后返回 Controller
    builder.add_edge("ExtractAndReason", "Controller")
    builder.add_edge("ExecuteCode", "Controller")
    builder.add_edge("WriteCode", "ExecuteCode")   # 写完后立即执行

    builder.add_edge("Generate", END)

    return builder.compile()


# ============================================================================
# 第五部分：主程序示例
# ============================================================================
if __name__ == "__main__":
    # 1. 准备图谱（用内存模拟数据演示，实际请用你自己的构建方式）
    graph_engine = LegalDenseGraphBuilder(embedding_dim=768)
    # 模拟几个节点
    dummy_nodes = [
        {"id": 1, "content": "试用期不符合录用条件可解除合同。", "type": "Raw", "metadata": {"source": "劳动法"}},
        {"id": 2, "content": "违法解除劳动合同按经济补偿标准的二倍支付赔偿金。", "type": "Raw", "metadata": {"source": "劳动法"}},
    ]
    # 随机生成向量（实际要用 embedding 模型编码）
    dummy_emb = np.random.randn(len(dummy_nodes), 768).astype(np.float32)
    # 归一化
    dummy_emb = dummy_emb / np.linalg.norm(dummy_emb, axis=1, keepdims=True)
    graph_engine.build_initial_graph_batch(dummy_nodes, dummy_emb)

    # 2. 准备 Embedding 函数（mock，实际请用 SentenceTransformer）
    def mock_embedding(text: str) -> np.ndarray:
        # 返回归一化随机向量，仅示意
        v = np.random.randn(768).astype(np.float32)
        return v / np.linalg.norm(v)

    # 3. 初始化 AgenticNodesOperator（需要替换为你的 API Key）
    # 若没有真实 API，以下调用会失败，因此用模拟 LLM 回退演示
    # agentic_ops = AgenticNodesOperator(api_key="your-api-key")
    # 为了本地运行，我们创建一个 Mock 子类，重写 _call_llm
    class MockAgenticOps(AgenticNodesOperator):
        def _call_llm(self, system_prompt, user_prompt, require_json=False, temperature=0.1):
            logger.info(f"Mock LLM 调用: {system_prompt[:50]}...")
            if require_json:
                return '{"strategy_queue": ["检查劳动关系", "核实辞退理由", "计算赔偿金额"]}'
            # 简单回显
            return f"Mock 回答: 基于事实，结论是赔偿10000元。"

    agentic_ops = MockAgenticOps(api_key="mock")

    # 4. 构建智能体
    legal_agent = build_legal_agent(graph_engine, mock_embedding, agentic_ops)

    # 5. 运行
    initial_state = {
        "user_query": "试用期最后一天被辞退，能拿多少赔偿？",
        "current_hop": 0,
        "execution_plan": [],
        "current_sub_task": "",
        "current_context": [],
        "next_action": "",
        "generated_code": "",
        "code_execution_result": ""
    }

    final_state = legal_agent.invoke(initial_state)

    # 打印最终结果
    print("\n" + "="*50)
    print("最终法律意见书：")
    for entry in final_state["current_context"]:
        if entry.get("hop") == "final":
            print(entry["reasoning"])