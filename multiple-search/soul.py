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
logger = logging.getLogger("LegalAgentPlanReplan")

# ============================================================================
# 第一部分：法律稠密图引擎（igraph + FAISS）—— 保持原样
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

        self.graph = ig.Graph(directed=False)
        base_index = faiss.IndexHNSWFlat(self.dim, 32, faiss.METRIC_INNER_PRODUCT)
        self.index = faiss.IndexIDMap(base_index)
        logger.info(f"Graph engine init: dim={self.dim}, M=32, metric=IP")

    def _l2_normalize(self, vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def build_initial_graph_batch(self,
                                  nodes_data: List[Dict[str, Any]],
                                  embeddings: np.ndarray,
                                  search_batch_size: int = 10000) -> None:
        total_nodes = len(nodes_data)
        if total_nodes != embeddings.shape[0]:
            raise ValueError("节点数量与矩阵维度不匹配！")

        logger.info(f"开始批量构建图谱，总节点数: {total_nodes}")
        self.graph.add_vertices(total_nodes)
        self.graph.vs["name"] = [n["id"] for n in nodes_data]
        self.graph.vs["content"] = [n["content"] for n in nodes_data]
        self.graph.vs["type"] = [n["type"] for n in nodes_data]
        self.graph.vs["metadata"] = [n["metadata"] for n in nodes_data]

        all_ids = np.array([n["id"] for n in nodes_data], dtype=np.int64)
        self.index.add_with_ids(embeddings.astype(np.float32), all_ids)
        logger.info("FAISS 索引批量注入完成。")

        all_edges, all_weights = [], []
        for i in range(0, total_nodes, search_batch_size):
            end_idx = min(i + search_batch_size, total_nodes)
            batch_emb = embeddings[i:end_idx].astype(np.float32)
            batch_ids = all_ids[i:end_idx]
            sims, n_ids = self.index.search(batch_emb, self.top_k)
            for row_idx, query_id in enumerate(batch_ids):
                for sim, target_id in zip(sims[row_idx], n_ids[row_idx]):
                    if (self.threshold <= sim < self.dedup_threshold
                            and target_id != query_id and target_id != -1):
                        all_edges.append((query_id, int(target_id)))
                        all_weights.append(float(sim))

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

    def add_node(self, node_id, content, node_type, embedding, metadata=None):
        # 简化保留接口，此处省略具体实现（与原版一致）
        pass

# ============================================================================
# 第二部分：LLM 基座与节点操作器（Meta‑Planner/Extractor/Reasoner/Generator）
# ============================================================================
class LegalLLMBase:
    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com"):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = "deepseek-chat"

    def _call_llm(self, system_prompt: str, user_prompt: str,
                  require_json: bool = False, temperature: float = 0.1) -> str:
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
    """封装 Extractor/Reasoner/Generator 的 Prompt 逻辑"""

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
            return [user_query]

    def extract_facts(self, current_sub_task: str, raw_retrieved_docs: str,
                      original_query: Optional[str] = None) -> str:
        system_prompt = """你是法律事实提取器（Extractor）。只从给定文档中抽取与子任务直接相关的原子事实，禁止推理。若无相关信息，回复“未找到相关事实”。"""
        user_prompt = f"子任务：{current_sub_task}\n文档：\n{raw_retrieved_docs}"
        if original_query:
            user_prompt += f"\n[最高指令：确保不偏离原始诉求 -> {original_query}]"
        return self._call_llm(system_prompt, user_prompt)

    def reason(self, current_sub_task: str, extracted_facts: str) -> str:
        system_prompt = """你是法官助理（Reasoner）。严格基于给定事实对子任务进行逻辑推演，不得引入外部知识。"""
        user_prompt = f"子任务：{current_sub_task}\n事实：{extracted_facts}"
        return self._call_llm(system_prompt, user_prompt)

    def generate_final_report(self, user_query: str, accumulated_context: List[Dict]) -> str:
        system_prompt = """你是资深律师（Generator）。结合已验证的上下文证据链，生成专业、直接回答用户问题的法律意见书。标记计算出的金额。"""
        context_str = ""
        for item in accumulated_context:
            context_str += f"[Hop {item.get('hop')}] {item.get('sub_task', '计算')} -> {item.get('reasoning', item.get('data', ''))}\n\n"
        user_prompt = f"用户问题：{user_query}\n证据链：\n{context_str}"
        return self._call_llm(system_prompt, user_prompt, temperature=0.3)


# ============================================================================
# 第三部分：Reasoner 模块（负责检索 + 回答 + 信息充分性判断）
# ============================================================================
class Reasoner:
    """
    融合检索与推理，并判断信息是否足够回答问题。
    依赖外部的法律图谱、Embedding 函数和 LLM 操作器。
    """
    def __init__(self, legal_graph: LegalDenseGraphBuilder, embedding_fn, agentic_ops: AgenticNodesOperator):
        self.legal_graph = legal_graph
        self.embedding_fn = embedding_fn
        self.agentic_ops = agentic_ops

    def retrieve(self, query: str) -> str:
        """从法律图谱检索相关文档，返回合并后的文本"""
        if not self.legal_graph or not self.embedding_fn:
            return ""
        emb = self.embedding_fn(query)
        norm_emb = self.legal_graph._l2_normalize(emb).reshape(1, -1).astype(np.float32)
        k = min(5, self.legal_graph.index.ntotal)
        if k == 0:
            return ""
        sims, ids = self.legal_graph.index.search(norm_emb, k)
        docs = []
        for sim, nid in zip(sims[0], ids[0]):
            if nid != -1 and sim > 0.6:
                try:
                    node = self.legal_graph.graph.vs.find(name=int(nid))
                    docs.append(node["content"])
                except ValueError:
                    pass
        return "\n---\n".join(docs)

    def answer(self, sub_task: str, original_query: str) -> Dict[str, Any]:
        """
        对子任务进行检索、抽取事实、推理，并判断信息是否充足。
        返回字典：
          - fact: 抽取的事实
          - reasoning: 推理结论
          - sufficient: 是否充足
          - suggestion: 若不足，给出补充查询建议
        """
        docs = self.retrieve(sub_task)
        if not docs:
            return {
                "fact": "未检索到相关文档",
                "reasoning": "无法回答",
                "sufficient": False,
                "suggestion": f"请尝试更宽泛的检索词：{sub_task}"
            }

        # 调用 agentic_ops 的 extractor 和 reasoner
        facts = self.agentic_ops.extract_facts(sub_task, docs, original_query=original_query)
        reasoning = self.agentic_ops.reason(sub_task, facts)

        # 简单规则判断充足性（可替换为 LLM 判断）
        sufficient = True
        suggestion = ""
        if "未找到相关事实" in facts or "无法确定" in reasoning:
            sufficient = False
            suggestion = f"当前文档未覆盖“{sub_task}”，建议检索更精准的法律条文。"
        elif len(docs) < 50:  # 文档过短也可能不够
            sufficient = False
            suggestion = "检索到的文档过于简略，请尝试不同的查询表述。"

        return {
            "fact": facts,
            "reasoning": reasoning,
            "sufficient": sufficient,
            "suggestion": suggestion
        }


# ============================================================================
# 第四部分：真实 DeepSeek Planner
# ============================================================================
DEEPSEEK_API_KEY = "sk-your-real-api-key"   # 替换为你的 Key
deepseek_client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

def call_deepseek_planner(query: str) -> List[str]:
    system_prompt = """你是一个顶级的中国法律案件拆解专家与智能体规划中枢。
    【核心任务】：不要尝试回答用户的法律问题！你的唯一任务是将用户的案情描述，拆解为按顺序执行的原子查询与计算步骤。
    
    【拆解原则】：
    1. 步步为营：先查明事实前提，再核定法律性质，最后核算具体数额。
    2. 原子化：每一步只能包含一个具体的查询动作。
    
    【严格输出格式】：
    你必须输出合法的 JSON 格式，包含且仅包含一个 `task_queue` 字段，其值为字符串数组。
    示例：{"task_queue": ["核实劳动者的入职时间、离职时间与平均工资", "审查公司单方面解除劳动合同的法定事由及通知程序", "核算违法解除劳动合同的 2N 赔偿金"]}
    """
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"用户案情：{query}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        return parsed.get("task_queue", ["查核基础法律事实"])
    except Exception as e:
        logger.error(f"DeepSeek Planner 调用失败: {e}")
        return ["兜底步骤：人工审查案件材料"]


# ============================================================================
# 第五部分：新的 AgentState 与节点实现
# ============================================================================
class AgentState(TypedDict):
    user_query: str
    task_queue: List[str]                               # 当前任务队列
    past_observations: Annotated[List[str], operator.add]  # 历史观察（追加）
    final_report: Optional[str]
    # 可选的沙箱会话ID
    sandbox_session_id: Optional[str]


def node_planner(state: AgentState) -> dict:
    logger.info("Planner 启动，调用 DeepSeek 生成执行计划")
    queue = call_deepseek_planner(state["user_query"])
    return {"task_queue": queue}


def node_executor(state: AgentState, reasoner: Reasoner) -> dict:
    if not state["task_queue"]:
        return {}
    current_task = state["task_queue"][0]
    logger.info(f"Executor 正在处理: {current_task}")

    # 使用 Reasoner 进行检索、推理、判断充足性
    result = reasoner.answer(current_task, original_query=state["user_query"])

    # 构造观察记录
    observation = (f"【任务：{current_task}】\n"
                   f"事实：{result['fact']}\n"
                   f"推理：{result['reasoning']}")

    new_queue = state["task_queue"][1:]  # 默认移除当前任务

    if not result["sufficient"]:
        # 信息不足：将补充任务插入队列头部（不删除当前任务，而是改成补充查询）
        suggestion = result.get("suggestion", "需要更精准的法律检索")
        # 新建一个补充任务，插入到队首
        new_queue = [f"{current_task}（补充：{suggestion}）"] + new_queue
        observation += f"\n⚠️ 信息不足，已添加补充任务：{suggestion}"

    return {
        "task_queue": new_queue,
        "past_observations": [observation]
    }


def node_replanner(state: AgentState) -> dict:
    obs_text = "\n".join(state.get("past_observations", []))
    queue = state["task_queue"]

    # 如果队列为空，结束
    if not queue:
        logger.info("所有任务完成，准备生成最终报告")
        return {}

    # 动态重规划示例：发现“未签订劳动合同”但计划中没有“双倍工资”
    if "未签订劳动合同" in obs_text and not any("双倍工资" in t for t in queue):
        new_queue = [
            "核查未签劳动合同二倍工资仲裁时效",
            "合并计算二倍工资差额与违法解除赔偿金"
        ]
        logger.info(f"Replanner 推翻原计划，新队列: {new_queue}")
        return {"task_queue": new_queue}

    # 默认保持原队列继续执行
    return {}


def node_generate(state: AgentState, agentic_ops: AgenticNodesOperator) -> dict:
    logger.info("Generator 生成最终法律意见书")
    # 将 past_observations 转换为 agentic_ops 需要的格式
    accumulated = []
    for idx, obs in enumerate(state.get("past_observations", [])):
        accumulated.append({
            "hop": idx + 1,
            "sub_task": state["task_queue"][idx] if idx < len(state["task_queue"]) else "总结",
            "reasoning": obs
        })
    report = agentic_ops.generate_final_report(state["user_query"], accumulated)
    return {"final_report": report}


# ============================================================================
# 第六部分：组装 LangGraph 图
# ============================================================================
def build_plan_replan_agent(reasoner: Reasoner, agentic_ops: AgenticNodesOperator):
    builder = StateGraph(AgentState)

    # 闭包注入依赖
    def planner(state): return node_planner(state)
    def executor(state): return node_executor(state, reasoner)
    def replanner(state): return node_replanner(state)
    def generate(state): return node_generate(state, agentic_ops)

    builder.add_node("Planner", planner)
    builder.add_node("Executor", executor)
    builder.add_node("Replanner", replanner)
    builder.add_node("Generate", generate)

    builder.set_entry_point("Planner")
    builder.add_edge("Planner", "Executor")
    builder.add_edge("Executor", "Replanner")

    # Replanner 后根据队列是否为空决定去向
    def route_after_replan(state: AgentState):
        if len(state.get("task_queue", [])) == 0:
            return "Generate"
        return "Executor"

    builder.add_conditional_edges("Replanner", route_after_replan)
    builder.add_edge("Generate", END)

    return builder.compile()


# ============================================================================
# 第七部分：运行示例（使用 Mock 数据演示，替换真实服务即可运行）
# ============================================================================
if __name__ == "__main__":
    # 1. 模拟图谱（实际应用请加载真实数据）
    graph_engine = LegalDenseGraphBuilder(embedding_dim=768)
    dummy_nodes = [
        {"id": 1, "content": "试用期不符合录用条件可解除合同。", "type": "Raw", "metadata": {"source": "劳动法"}},
        {"id": 2, "content": "违法解除劳动合同按经济补偿标准的二倍支付赔偿金。", "type": "Raw", "metadata": {"source": "劳动法"}},
    ]
    dummy_emb = np.random.randn(len(dummy_nodes), 768).astype(np.float32)
    dummy_emb = dummy_emb / np.linalg.norm(dummy_emb, axis=1, keepdims=True)
    graph_engine.build_initial_graph_batch(dummy_nodes, dummy_emb)

    # 2. 模拟 Embedding 函数
    def mock_embedding(text: str) -> np.ndarray:
        v = np.random.randn(768).astype(np.float32)
        return v / np.linalg.norm(v)

    # 3. 模拟 AgenticNodesOperator（由于无真实 API Key，使用 Mock）
    class MockAgenticOps(AgenticNodesOperator):
        def _call_llm(self, system_prompt, user_prompt, require_json=False, temperature=0.1):
            logger.info(f"Mock LLM called with: {system_prompt[:50]}...")
            if require_json:
                return '{"strategy_queue": ["检查劳动关系", "核实辞退理由", "计算赔偿金额"]}'
            # 简单回显
            if "extract" in system_prompt:
                return "Mock 提取事实：公司口头辞退，属于违法解除。"
            if "reason" in system_prompt:
                return "Mock 推理：根据事实，应支付双倍赔偿金。"
            if "generate" in system_prompt:
                return "Mock 最终报告：您可以获得2N赔偿金。"
            return "Mock 回答"

    mock_ops = MockAgenticOps(api_key="mock")

    # 4. 创建 Reasoner
    reasoner = Reasoner(legal_graph=graph_engine, embedding_fn=mock_embedding, agentic_ops=mock_ops)

    # 5. 构建并运行智能体
    agent = build_plan_replan_agent(reasoner, mock_ops)

    initial_state = {
        "user_query": "试用期最后一天被辞退，能拿多少赔偿？",
        "task_queue": [],
        "past_observations": [],
        "final_report": ""
    }

    final_state = agent.invoke(initial_state)

    print("\n" + "="*50)
    print("最终法律意见书：")
    print(final_state.get("final_report", "无报告生成"))
    print("\n全部观察记录：")
    for obs in final_state["past_observations"]:
        print(obs)