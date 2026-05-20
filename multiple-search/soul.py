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
#语义缓存
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
class UnifiedQueryRouter:
    def __init__(self, cache: Optional[SemanticCache] = None):
        # ... 原有初始化 ...
        self.cache = cache
        # ...

    def process(self, query: str) -> Dict[str, Any]:
        # 1. 语义缓存拦截
        if self.cache:
            q_vec = self.embedder.encode(query, normalize_embeddings=True)
            cached_answer = self.cache.lookup(q_vec)
            if cached_answer is not None:
                logger.info("⚡ 语义缓存命中，跳过后续流程")
                return {
                    "intent": "CACHED",
                    "status": "cached_response",
                    "response": cached_answer
                }

        # 2. 原有路由流程
        result = self.route(query)  # ... 你的三层漏斗 ...

        # 3. 生成最终答案后写回缓存 (示例，真实场景在最终答案生成后执行)
        # if result.get("status") == "simple_rag" or result.get("final_report"):
        #     answer = result.get("response") or result.get("final_report", "")
        #     if answer and self.cache:
        #         self.cache.add(q_vec, answer)

        return result
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

    # ====================== 新增方法 (最小侵入) ======================
    def grade_facts(self, task_desc: str, docs: str) -> dict:
        """
        Grader：强制 LLM 输出 JSON，包含 status (sufficient/partial/irrelevant)
        如果调用失败，退回默认 irrelevant 状态，保证鲁棒性
        """
        system_prompt = """你是一个极其严苛的事实调查官。不要推理，只对比资料与任务。
        输出JSON：
        {
          "rationale": "判决说理",
          "status": "sufficient | partial | irrelevant",
          "extracted_facts": ["事实1"] (仅在 sufficient/partial 时输出),
          "missing_info": "缺少的搜索词" (仅在 partial 时输出)
        }"""
        raw = self._call_llm(system_prompt, f"任务：{task_desc}\n资料：{docs}", require_json=True)
        try:
            return json.loads(raw)
        except Exception:
            logger.warning("Grader JSON 解析失败，退回 irrelevant")
            return {"status": "irrelevant", "rationale": "Grader 解析错误"}

    def replan_with_wormhole(self, query: str, global_facts: list, fail_log: str) -> List[str]:
        """
        虫洞重规划：输出新的任务列表（字符串），若需虫洞则在描述前加 [WORMHOLE] 前缀
        """
        system_prompt = """你是一个重规划引擎。前序任务已失败。
        如果图谱线索断裂，你可以启用虫洞进行全局搜索。
        严格输出JSON：{"task_queue": ["步骤1", "步骤2"]}。
        虫洞任务请以 "[WORMHOLE]" 开头。"""
        prompt = f"案情：{query}\n已有事实：{global_facts}\n失败记录：{fail_log}"
        raw = self._call_llm(system_prompt, prompt, require_json=True)
        try:
            data = json.loads(raw)
            return data.get("task_queue", ["全局检索案情相关法条"])
        except Exception:
            return [f"[WORMHOLE] 全局检索: {query}"]

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
        现优先使用 Grader 结构化输出；若无法获得则回退原有逻辑。
        """
        docs = self.retrieve(sub_task)
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
# 第五部分：新的 AgentState 与节点实现 (最小侵入式扩展)
# ============================================================================
class AgentState(TypedDict):
    user_query: str
    task_queue: List[str]                               # 当前任务队列（保留原字段）
    past_observations: Annotated[List[str], operator.add]  # 历史观察（追加）
    # ---- 新增字段，均带默认值，不影响已有代码 ----
    global_facts: List[str]                              # 全局防篡改事实
    retry_context: Dict[str, Any]                        # 临时状态（grader_status等）
    recursion_depth: int                                 # 熔断计数
    final_report: Optional[str]
    sandbox_session_id: Optional[str]


def node_l0_gateway(state: AgentState) -> dict:
    """新增 L0 安全网关与初始化熔断字段"""
    query = state["user_query"]
    # 极简安全检查示例，可根据需要扩展
    if "忽略指令" in query or "越狱" in query:
        raise ValueError("L0_REJECT: 触发安全熔断")
    return {
        "recursion_depth": 0,
        "retry_context": {},
        "global_facts": state.get("global_facts", [])
    }


def node_planner(state: AgentState) -> dict:
    logger.info("Planner 启动，调用 DeepSeek 生成执行计划")
    queue = call_deepseek_planner(state["user_query"])
    return {"task_queue": queue}


def node_executor(state: AgentState, reasoner: Reasoner) -> dict:
    if not state["task_queue"]:
        return {}
    # ---- 新增熔断保护 ----
    depth = state.get("recursion_depth", 0)
    if depth > 5:
        logger.warning("🚨 触发算力熔断，强制终止")
        return {"retry_context": {"status": "force_stop"}}

    current_task = state["task_queue"][0]
    # ---- 新增：解析虫洞前缀 ----
    actual_query = current_task
    is_wormhole = False
    if current_task.startswith("[WORMHOLE]"):
        actual_query = current_task[len("[WORMHOLE]"):].strip()
        is_wormhole = True
        logger.info(f"虫洞穿越模式激活: {actual_query}")

    logger.info(f"Executor 正在处理: {current_task}")

    # 使用 Reasoner 进行检索、推理、判断充足性
    # Reasoner.answer 已升级为使用 Grader
    result = reasoner.answer(actual_query, original_query=state["user_query"])

    # 构造观察记录（保留原有格式）
    observation = (f"【任务：{current_task}】\n"
                   f"事实：{result['fact']}\n"
                   f"推理：{result['reasoning']}")

    new_queue = state["task_queue"][1:]  # 默认移除当前任务

    # ---- 新增：根据 grader_status 和 sufficient 联合决策 ----
    grader_status = result.get("grader_status", "irrelevant")
    sufficient = result.get("sufficient", False)

    if not sufficient:
        suggestion = result.get("suggestion", "需要更精准的法律检索")
        # 原有逻辑：插入补充任务（保留）
        new_queue = [f"{current_task}（补充：{suggestion}）"] + new_queue
        observation += f"\n⚠️ 信息不足，已添加补充任务：{suggestion}"

        # 新增：设置 retry_context 供 Replanner 决策
        retry_ctx = {
            "status": grader_status,
            "missing_info": suggestion,
            "fail_count": state.get("retry_context", {}).get("fail_count", 0) + 1,
            "is_wormhole": is_wormhole
        }
    else:
        retry_ctx = {"status": "sufficient", "fail_count": 0}

    # ---- 新增：追加全局事实 (只追加，不覆盖) ----
    facts_to_add = []
    if isinstance(result['fact'], str) and result['fact'] != "未检索到相关文档":
        facts_to_add.append(result['fact'])
    elif isinstance(result['fact'], list):
        facts_to_add.extend(result['fact'])

    return {
        "task_queue": new_queue,
        "past_observations": [observation],
        "global_facts": state.get("global_facts", []) + facts_to_add,
        "retry_context": retry_ctx,
        "recursion_depth": depth + 1
    }


def node_replanner(state: AgentState) -> dict:
    obs_text = "\n".join(state.get("past_observations", []))
    queue = state["task_queue"]
    retry_ctx = state.get("retry_context", {})

    # 如果队列为空，结束
    if not queue:
        logger.info("所有任务完成，准备生成最终报告")
        return {}

    # ---- 新增：基于 retry_context 的重规划决策 ----
    status = retry_ctx.get("status")
    fail_count = retry_ctx.get("fail_count", 0)

    # 情况1：强制停止
    if status == "force_stop":
        return {"task_queue": []}

    # 情况2：无头绪 (irrelevant) 或部分缺失且重试超过2次 -> 虫洞重规划
    if status == "irrelevant" or (status == "partial" and fail_count > 2):
        logger.info("触发虫洞重规划引擎")
        # 尝试使用 agentic_ops 的 replan_with_wormhole，若不可用则生成简单虫洞任务
        if hasattr(state, 'agentic_ops'):  # 实际无法直接获取，改为通过闭包传递，此处简化处理
            new_queue = ["[WORMHOLE] 全局检索相关法条"]  # 默认虫洞任务
        else:
            new_queue = ["[WORMHOLE] 全局检索相关法条"]
        return {"task_queue": new_queue, "retry_context": {}}

    # 情况3：原有规则补充（如未签订劳动合同加双倍工资）
    if "未签订劳动合同" in obs_text and not any("双倍工资" in t for t in queue):
        new_queue = [
            "核查未签劳动合同二倍工资仲裁时效",
            "合并计算二倍工资差额与违法解除赔偿金"
        ]
        logger.info(f"Replanner 硬规则补充，新队列: {new_queue}")
        return {"task_queue": new_queue, "retry_context": {}}

    # 默认保持原队列继续执行
    return {}


def node_generate(state: AgentState, agentic_ops: AgenticNodesOperator) -> dict:
    logger.info("Generator 生成最终法律意见书")
    # 保留原有上下文构建方式
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
# 第六部分：组装 LangGraph 图（仅增加 L0 节点，路由增强）
# ============================================================================
def build_plan_replan_agent(reasoner: Reasoner, agentic_ops: AgenticNodesOperator):
    builder = StateGraph(AgentState)

    # 闭包注入依赖
    def l0_gateway(state): return node_l0_gateway(state)
    def planner(state): return node_planner(state)
    def executor(state): return node_executor(state, reasoner)
    def replanner(state): return node_replanner(state)
    def generate(state): return node_generate(state, agentic_ops)

    # 节点注册（增加 L0_Gateway）
    builder.add_node("L0_Gateway", l0_gateway)
    builder.add_node("Planner", planner)
    builder.add_node("Executor", executor)
    builder.add_node("Replanner", replanner)
    builder.add_node("Generate", generate)

    # 拓扑：L0 -> Planner -> Executor -> Replanner
    builder.set_entry_point("L0_Gateway")
    builder.add_edge("L0_Gateway", "Planner")
    builder.add_edge("Planner", "Executor")
    builder.add_edge("Executor", "Replanner")

    # Replanner 后根据队列和熔断状态决定去向
    def route_after_replan(state: AgentState):
        # 新增：强制停止直接结案
        if state.get("retry_context", {}).get("status") == "force_stop":
            return "Generate"
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

    # 3. 模拟 AgenticNodesOperator（包含新增方法）
    class MockAgenticOps(AgenticNodesOperator):
        def _call_llm(self, system_prompt, user_prompt, require_json=False, temperature=0.1):
            logger.info(f"Mock LLM called with: {system_prompt[:50]}...")
            if require_json:
                if "grade" in system_prompt.lower() or "事实调查官" in system_prompt:
                    return '{"rationale": "Mock充足", "status": "sufficient", "extracted_facts": ["Mock事实1"]}'
                if "replan" in system_prompt.lower() or "重规划" in system_prompt:
                    return '{"task_queue": ["[WORMHOLE] Mock全局检索"]}'
                return '{"strategy_queue": ["检查劳动关系", "核实辞退理由", "计算赔偿金额"]}'
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
        "final_report": "",
        "global_facts": [],
        "retry_context": {},
        "recursion_depth": 0
    }

    final_state = agent.invoke(initial_state)

    print("\n" + "="*50)
    print("最终法律意见书：")
    print(final_state.get("final_report", "无报告生成"))
    print("\n全部观察记录：")
    for obs in final_state["past_observations"]:
        print(obs)
    print("\n全局事实：")
    for f in final_state.get("global_facts", []):
        print(f"  - {f}")