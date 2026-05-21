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
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from query import UnifiedQueryRouter_Query
            self._router = UnifiedQueryRouter_Query()
        else:
            self._router = query_router

    def process(self, query: str) -> Dict[str, Any]:
        # 1. 语义缓存拦截
        if self.cache:
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
        result = self._router.process(query)

        # 3. 写回缓存：对简单问答和复杂任务的结果进行缓存
        if self.cache and result.get("status") in ("simple_rag", "agent_execution"):
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

    def replan_with_wormhole(self, query: str, global_facts: list, fail_log: str,
                             fail_count: int = 0) -> List[Dict]:
        """
        虫洞重规划 v2：输出任务列表，每个任务含 task_desc / engine / rationale。
        engine: 'GRAPH_TRAVERSAL' (图游走) 或 'GLOBAL_DENSE_WORMHOLE' (全局向量穿越)
        """
        system_prompt = f"""你是一个经过强化学习训练的顶级重规划引擎 (Replanner)。
【核心任务】：当前系统的法律检索路径已陷入死胡同，你需要基于全局事实账本，推翻或改写下一步的调查计划。

【可用引擎说明】：
1. GRAPH_TRAVERSAL（图谱游走）：系统默认引擎。在当前案件领域内寻找相邻线索（成本极低）。
2. GLOBAL_DENSE_WORMHOLE（虫洞穿越）：当当前图谱已彻底断裂，必须跨法律领域寻找依据时使用（算力消耗极大！连续失败>=3次才建议开启）。

【当前绝境状态】：
- 系统已连续碰壁次数：{fail_count}
- 碰壁原因：{fail_log}

严格按照以下 JSON Schema 输出：
{{"task_queue": [{{"task_desc": "...", "engine": "GRAPH_TRAVERSAL", "rationale": "..."}}]}}
"""
        prompt = f"案情：{query}\n已有事实：{global_facts}\n失败记录：{fail_log}"
        raw = self._call_llm(system_prompt, prompt, require_json=True, temperature=0.4)
        try:
            data = json.loads(raw)
            tasks = data.get("task_queue", [])
            result = []
            for t in tasks:
                if isinstance(t, dict):
                    result.append({
                        "task_desc": t.get("task_desc", t.get("task", str(t))),
                        "engine": t.get("engine", "GRAPH_TRAVERSAL"),
                        "rationale": t.get("rationale", "")
                    })
                else:
                    result.append({"task_desc": str(t), "engine": "GRAPH_TRAVERSAL", "rationale": ""})
            return result if result else [
                {"task_desc": "全局检索案情相关法条", "engine": "GLOBAL_DENSE_WORMHOLE",
                 "rationale": "兜底虫洞穿越"}
            ]
        except Exception:
            return [{"task_desc": f"全局检索: {query[:50]}", "engine": "GLOBAL_DENSE_WORMHOLE",
                     "rationale": "JSON解析降级"}]

    # ====================== 沙箱代码生成 ======================
    def generate_calculation_code(self, user_query: str,
                                  reasoning_chain: List[Dict],
                                  error_context: str = "") -> str:
        """
        生成 Python 计算代码（赔偿金/补偿金/加班费等金额）。
        若 error_context 非空，表示上次代码执行失败，需要修正。
        """
        chain_text = ""
        for item in reasoning_chain:
            chain_text += (f"[{item.get('sub_task', '')}] "
                           f"事实: {item.get('facts', '')} "
                           f"推理: {item.get('reasoning', '')}\n")

        correction_hint = ""
        if error_context:
            correction_hint = (f"\n【上次执行报错，请修正】\n{error_context}\n"
                               "请分析错误原因并生成修正后的代码。")

        system_prompt = f"""你是一名精通中国劳动法的法官助理兼Python程序员。
根据以下案件事实和推理链，编写一段纯 Python 代码来计算最终的赔偿金额。

【代码要求】：
1. 变量命名清晰，使用中文注释说明每步对应的法律依据
2. 最终结果必须赋值给变量 `result`
3. 只输出可执行的 Python 代码，不要包裹在 ```python ``` 中
4. 不要使用任何外部库（如 requests），只用标准库
5. 不要进行任何文件读写或网络操作{correction_hint}"""

        user_prompt = f"案件：{user_query}\n\n推理链：\n{chain_text}"

        return self._call_llm(system_prompt, user_prompt, temperature=0.1)

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
    # ---- 沙箱相关字段 ----
    generated_code: Optional[str]                        # LLM 生成的 Python 计算代码
    calc_result: Optional[str]                           # 沙箱执行的计算结果
    sandbox_retries: int                                 # 沙箱重试次数
    needs_sandbox_calc: bool                             # 是否需要金额计算


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
    # ---- 熔断保护 ----
    depth = state.get("recursion_depth", 0)
    if depth > 5:
        logger.warning("🚨 触发算力熔断，强制终止")
        return {"retry_context": {"status": "force_stop"}}

    current_task = state["task_queue"][0]

    # ---- v2: 解析 Dict 任务格式 ----
    if isinstance(current_task, dict):
        actual_query = current_task.get("task_desc", str(current_task))
        engine = current_task.get("engine", "GRAPH_TRAVERSAL")
        rationale = current_task.get("rationale", "")
        is_wormhole = (engine == "GLOBAL_DENSE_WORMHOLE")
    else:
        # 兼容旧字符串格式
        actual_query = str(current_task)
        engine = "GRAPH_TRAVERSAL"
        rationale = ""
        is_wormhole = actual_query.startswith("[WORMHOLE]")
        if is_wormhole:
            actual_query = actual_query[len("[WORMHOLE]"):].strip()

    if is_wormhole:
        logger.info(f"🌌 虫洞穿越模式激活 (engine={engine}): {actual_query[:60]}")
    logger.info(f"Executor 正在处理: {actual_query[:80]}")

    # 使用 Reasoner 进行检索、推理、判断充足性
    result = reasoner.answer(actual_query, original_query=state["user_query"])

    # 构造观察记录（包含引擎信息）
    task_label = actual_query[:60]
    observation = (f"【任务：{task_label}】【引擎：{engine}】\n"
                   f"事实：{result['fact']}\n"
                   f"推理：{result['reasoning']}")

    new_queue = state["task_queue"][1:]  # 默认移除当前任务

    # ---- 新增：根据 grader_status 和 sufficient 联合决策 ----
    grader_status = result.get("grader_status", "irrelevant")
    sufficient = result.get("sufficient", False)

    if not sufficient:
        suggestion = result.get("suggestion", "需要更精准的法律检索")
        # v2: 补充任务使用 Dict 格式
        supplement_task = {
            "task_desc": f"{actual_query[:40]}（补充：{suggestion}）",
            "engine": engine,  # 保持原引擎
            "rationale": f"信息不足补搜: {suggestion[:60]}"
        }
        new_queue = [supplement_task] + new_queue
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
    """v2: 支持 Pydantic 强类型任务格式 (task_desc + engine + rationale)"""
    obs_text = "\n".join(state.get("past_observations", []))
    queue = state["task_queue"]
    retry_ctx = state.get("retry_context", {})

    # 如果队列为空，结束
    if not queue:
        logger.info("所有任务完成，准备生成最终报告")
        return {}

    status = retry_ctx.get("status") or retry_ctx.get("grader_status")
    fail_count = retry_ctx.get("fail_count", 0)

    # 情况1：强制停止
    if status == "force_stop":
        logger.warning("算力熔断，强制终止")
        return {"task_queue": []}

    # 情况2：无头绪或部分缺失且重试超过阈值 -> LLM 虫洞重规划
    if status == "irrelevant" or (status == "partial" and fail_count > 2):
        logger.info("触发虫洞重规划引擎 (LLM)")
        from openai import OpenAI
        import os as _os
        _sys_path = _os.path.join(_os.path.dirname(__file__), "..")
        import sys as _sys
        _sys.path.insert(0, _sys_path)
        try:
            from config_loader import cfg as _cfg
            client = OpenAI(api_key=_cfg.get("llm", "api_key"), base_url=_cfg.get("llm", "base_url"))
        except Exception:
            client = OpenAI(api_key=os.environ.get("DEEPSEEK_API_KEY", ""), base_url="https://api.deepseek.com")

        class _ReplanOps:
            def __init__(self, c): self.client = c
            def _call_llm(self, sp, up, require_json=False, temperature=0.4):
                kw = {"model": "deepseek-chat", "messages": [{"role":"system","content":sp},{"role":"user","content":up}], "temperature": temperature, "max_tokens": 2048}
                if require_json: kw["response_format"] = {"type":"json_object"}
                return self.client.chat.completions.create(**kw).choices[0].message.content.strip()

        ops = _ReplanOps(client)

        system_prompt = f"""你是一个经过强化学习训练的顶级重规划引擎。
当前图谱检索已陷入死胡同。连续碰壁次数：{fail_count}。碰壁原因：{obs_text[-300:]}。
可用引擎：GRAPH_TRAVERSAL (图游走) / GLOBAL_DENSE_WORMHOLE (虫洞穿越)。
严格输出JSON：{{"task_queue":[{{"task_desc":"...","engine":"GRAPH_TRAVERSAL","rationale":"..."}}]}}"""
        try:
            raw = ops._call_llm(system_prompt, f"案情：{state['user_query']}\n事实：{state.get('global_facts',[])}", require_json=True, temperature=0.4)
            data = json.loads(raw)
            new_queue = []
            for t in data.get("task_queue", []):
                new_queue.append({
                    "task_desc": t.get("task_desc", str(t)),
                    "engine": t.get("engine", "GRAPH_TRAVERSAL"),
                    "rationale": t.get("rationale", "")
                })
            if not new_queue:
                new_queue = [{"task_desc": "全局检索相关法条", "engine": "GLOBAL_DENSE_WORMHOLE", "rationale": "兜底"}]
        except Exception:
            new_queue = [{"task_desc": state["user_query"][:60], "engine": "GLOBAL_DENSE_WORMHOLE", "rationale": "异常降级"}]
    else:
        # 情况3：原有规则补充（兼容字符串和 Dict 格式）
        new_queue = list(queue)  # 保持原队列
        if "未签订劳动合同" in obs_text:
            has_double = any(
                (t.get("task_desc","") if isinstance(t, dict) else str(t)).find("双倍工资") >= 0
                for t in queue
            )
            if not has_double:
                new_queue = [
                    {"task_desc": "核查未签劳动合同二倍工资仲裁时效", "engine": "GRAPH_TRAVERSAL", "rationale": "硬规则"},
                    {"task_desc": "合并计算二倍工资差额与违法解除赔偿金", "engine": "GRAPH_TRAVERSAL", "rationale": "硬规则"},
                ] + new_queue

    return {"task_queue": new_queue, "retry_context": {}}


def node_generate(state: AgentState, agentic_ops: AgenticNodesOperator) -> dict:
    logger.info("Generator 生成最终法律意见书")
    accumulated = []
    for idx, obs in enumerate(state.get("past_observations", [])):
        accumulated.append({
            "hop": idx + 1,
            "sub_task": state["task_queue"][idx] if idx < len(state["task_queue"]) else "总结",
            "reasoning": obs
        })
    report = agentic_ops.generate_final_report(state["user_query"], accumulated)

    # 检测是否需要金额计算 → 触发沙箱流程
    calc_keywords = ["赔偿", "补偿", "加班费", "工资", "双倍", "2N", "N+1", "金额", "元"]
    needs_calc = any(kw in state["user_query"] + report for kw in calc_keywords)

    result = {"final_report": report}
    if needs_calc:
        result["needs_sandbox_calc"] = True
        result["sandbox_retries"] = 0
        logger.info("检测到金额计算需求，将进入沙箱执行流程")
    return result


# ============================================================================
# 沙箱节点：代码生成 → 执行 → 结果注入
# ============================================================================
def node_write_code(state: AgentState, agentic_ops: AgenticNodesOperator) -> dict:
    """生成 Python 计算代码"""
    logger.info("📝 沙箱阶段: 生成计算代码")

    # 从 past_observations 提取推理链
    reasoning_chain = []
    obs_list = state.get("past_observations", [])
    for obs in obs_list:
        if isinstance(obs, dict):
            reasoning_chain.append({
                "sub_task": obs.get("task", ""),
                "facts": str(obs.get("extracted_facts", "")),
                "reasoning": str(obs.get("status", ""))
            })
        elif isinstance(obs, str):
            reasoning_chain.append({"sub_task": "", "facts": obs, "reasoning": ""})

    error_ctx = state.get("retry_context", {}).get("sandbox_error", "")
    code = agentic_ops.generate_calculation_code(
        state["user_query"], reasoning_chain, error_context=error_ctx
    )

    return {
        "generated_code": code,
        "sandbox_retries": state.get("sandbox_retries", 0)
    }


def node_execute_code(state: AgentState) -> dict:
    """在 Docker 沙箱中执行代码"""
    logger.info("🐳 沙箱阶段: 执行计算代码")

    code = state.get("generated_code", "")
    if not code:
        return {"retry_context": {"sandbox_error": "无代码可执行"}}

    # 尝试使用 Docker 沙箱，失败则回退本地执行
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "legal_sandbox"))
        from sandbox_manager import DockerSandboxManager

        session_id = state.get("sandbox_session_id")
        manager = DockerSandboxManager()

        if not session_id:
            try:
                session_id = manager.start_session()
            except Exception as e:
                logger.warning(f"沙箱容器创建失败: {e}，回退本地执行")
                return _execute_locally(state, code)

        result = manager.execute_code(session_id, code)

        if result.get("error"):
            logger.warning(f"沙箱执行报错: {result['error']}")
            retries = state.get("sandbox_retries", 0) + 1
            if retries < 3:
                return {
                    "retry_context": {"sandbox_error": result["error"]},
                    "sandbox_retries": retries,
                    "sandbox_session_id": session_id
                }
            # 超过重试上限，记录错误继续
            return {
                "calc_result": f"沙箱多次执行失败: {result['error']}",
                "sandbox_session_id": session_id
            }

        logger.info(f"沙箱执行成功: {result.get('output', '')[:100]}")
        return {
            "calc_result": result.get("output", "").strip(),
            "sandbox_session_id": session_id
        }

    except ImportError:
        logger.warning("Docker SDK 不可用，回退本地执行")
        return _execute_locally(state, code)


def _execute_locally(state: AgentState, code: str) -> dict:
    """本地安全执行 Python 代码（受限环境回退方案）"""
    import io, sys as _sys, traceback

    old_stdout = _sys.stdout
    redirected = io.StringIO()
    _sys.stdout = redirected

    error_msg = None
    # 受限的全局命名空间
    safe_globals = {
        "__builtins__": {
            "abs": abs, "min": min, "max": max, "sum": sum,
            "round": round, "int": int, "float": float,
            "len": len, "range": range, "list": list,
            "dict": dict, "str": str, "bool": bool,
            "True": True, "False": False, "None": None,
            "print": print, "isinstance": isinstance,
        }
    }

    try:
        exec(code, safe_globals)
        result_value = safe_globals.get("result", "未定义 result 变量")
    except Exception:
        error_msg = traceback.format_exc()
    finally:
        _sys.stdout = old_stdout

    output = redirected.getvalue().strip()
    if error_msg:
        retries = state.get("sandbox_retries", 0) + 1
        if retries < 3:
            return {
                "retry_context": {"sandbox_error": error_msg},
                "sandbox_retries": retries
            }
        return {"calc_result": f"执行失败: {error_msg}"}

    # 提取 result 变量值
    try:
        result_str = str(safe_globals.get("result", output or "计算完成"))
    except Exception:
        result_str = output or "计算完成"

    return {"calc_result": result_str}


def node_inject_calc_result(state: AgentState) -> dict:
    """将沙箱计算结果注入最终报告"""
    calc_result = state.get("calc_result", "")
    final_report = state.get("final_report", "")

    if calc_result:
        enriched_report = (
            f"{final_report}\n\n"
            f"【计算明细】\n"
            f"经代码核算，最终结果为：{calc_result}"
        )
        logger.info("沙箱计算结果已注入最终报告")
        return {"final_report": enriched_report}

    return {}


def node_cleanup_sandbox(state: AgentState) -> dict:
    """清理沙箱资源"""
    session_id = state.get("sandbox_session_id")
    if session_id:
        try:
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "legal_sandbox"))
            from sandbox_manager import DockerSandboxManager
            manager = DockerSandboxManager()
            manager.destroy_session(session_id)
            logger.info(f"沙箱 {session_id} 已销毁")
        except Exception as e:
            logger.warning(f"沙箱清理异常: {e}")
    return {"sandbox_session_id": None}


# ============================================================================
# 第六部分：组装 LangGraph 图（含沙箱节点）
# ============================================================================
def build_plan_replan_agent(reasoner: Reasoner, agentic_ops: AgenticNodesOperator):
    builder = StateGraph(AgentState)

    # 闭包注入依赖
    def l0_gateway(state): return node_l0_gateway(state)
    def planner(state): return node_planner(state)
    def executor(state): return node_executor(state, reasoner)
    def replanner(state): return node_replanner(state)
    def generate(state): return node_generate(state, agentic_ops)
    def write_code(state): return node_write_code(state, agentic_ops)
    def execute_code(state): return node_execute_code(state)
    def inject_result(state): return node_inject_calc_result(state)
    def cleanup(state): return node_cleanup_sandbox(state)

    # 节点注册
    builder.add_node("L0_Gateway", l0_gateway)
    builder.add_node("Planner", planner)
    builder.add_node("Executor", executor)
    builder.add_node("Replanner", replanner)
    builder.add_node("Generate", generate)
    builder.add_node("WriteCode", write_code)
    builder.add_node("ExecuteCode", execute_code)
    builder.add_node("InjectResult", inject_result)
    builder.add_node("Cleanup", cleanup)

    # 拓扑: L0 → Planner → Executor → Replanner
    builder.set_entry_point("L0_Gateway")
    builder.add_edge("L0_Gateway", "Planner")
    builder.add_edge("Planner", "Executor")
    builder.add_edge("Executor", "Replanner")

    # Replanner 后根据队列和熔断状态决定去向
    def route_after_replan(state: AgentState):
        if state.get("retry_context", {}).get("status") == "force_stop":
            return "Generate"
        if len(state.get("task_queue", [])) == 0:
            return "Generate"
        return "Executor"

    builder.add_conditional_edges("Replanner", route_after_replan)

    # Generate 后判断是否需要沙箱计算
    def route_after_generate(state: AgentState):
        if state.get("needs_sandbox_calc"):
            return "WriteCode"
        return "Cleanup"

    builder.add_conditional_edges("Generate", route_after_generate)

    # WriteCode → ExecuteCode
    builder.add_edge("WriteCode", "ExecuteCode")

    # ExecuteCode 后判断是否需要重试
    def route_after_execute(state: AgentState):
        retry_ctx = state.get("retry_context", {})
        sandbox_retries = state.get("sandbox_retries", 0)
        if "sandbox_error" in retry_ctx and sandbox_retries < 3:
            logger.info(f"沙箱重试 {sandbox_retries}/3")
            return "WriteCode"
        return "InjectResult"

    builder.add_conditional_edges("ExecuteCode", route_after_execute)

    # InjectResult → Cleanup → END
    builder.add_edge("InjectResult", "Cleanup")
    builder.add_edge("Cleanup", END)

    return builder.compile()


# ============================================================================
# 第七部分：运行示例
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