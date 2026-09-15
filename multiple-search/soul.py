"""
智能体引擎 —— 整个系统的大脑 (LangGraph 状态机)
============================================
这里不是简单的顺序调用，而是一个有限状态机：Planner 拆任务 → Executor 逐个执行
→ Grader 判断够不够 → Replanner 救场 → Generate 出报告 → 沙箱算钱。

所有节点通过 LangGraph 的 StateGraph 编排，条件路由自动决定下一步走哪。
沙箱闭环的五步（写代码→执行→注入结果→清理）也在这里。

技术栈: LangGraph (StateGraph/END) / igraph / FAISS / Pydantic / DeepSeek API
"""
import os, sys, json, time, operator, logging
from typing import TypedDict, Annotated, List, Dict, Any, Optional
from pydantic import BaseModel, ValidationError
from openai import OpenAI
import numpy as np
import faiss
import igraph as ig
from langgraph.graph import StateGraph, END

# 允许从任意 cwd / 任意入口导入根目录模块（config_loader / double_layer_plan）
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from config_loader import cfg
import prompts  # 阶段 3：Prompt 单一真源（不再内联副本）
# 阶段 5：硬规则表与 Replanner Worker 共用同一份（原先这里只有 1 条规则）
from replanner_rules import apply_hard_rules

# 阶段 5：沙箱执行统一入口（与 reasoner_worker 共用同一份 Docker/降级逻辑）
_SANDBOX_DIR = os.path.join(_ROOT_DIR, "multiple-search", "legal_sandbox")
if _SANDBOX_DIR not in sys.path:
    sys.path.insert(0, _SANDBOX_DIR)
from sandbox_exec import run_code_once, destroy_session as destroy_sandbox_session

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
# 第一部分：法律稠密图引擎
# ============================================================================
# 阶段 4：这里原有一份与 dataset/graph.py **同名**的 LegalDenseGraphBuilder 副本。
# 两份实现行为并不相同（参数名、是否加载 BGE-M3、有无 tombstone/脏传播/摘要机制），
# 调用方根本分不清自己在用哪一个。现在统一复用 dataset/graph.py 的实现。
#
# 新实现只在需要"文本→向量"时才懒加载 BGE-M3；本模块的用法是外部注入
# embedding_fn + 传入预计算向量，因此不会触发模型加载。
#
# 参数名变化：similarity_threshold → connect_threshold，
#            dedup_threshold      → label_threshold（与 config.yaml 的 graph.* 对齐）
from dataset.graph import LegalDenseGraphBuilder, vname  # noqa: F401  (重新导出 + 顶点名转换)

# ============================================================================
# 第二部分：LLM 基座与节点操作器（Meta‑Planner/Extractor/Reasoner/Generator）
# ============================================================================
class LegalLLMBase:
    def __init__(self, api_key: str = None, base_url: str = None):
        # 阶段 2：默认值改为从 config.yaml 取，调用方不传也能正常工作
        self.client = OpenAI(
            api_key=api_key or cfg.get("llm", "api_key"),
            base_url=base_url or cfg.get("llm", "base_url", default="https://api.deepseek.com"),
        )
        self.model_name = cfg.get("llm", "judge_model", default="deepseek-chat")

    def _call_messages(self, messages: List[Dict[str, str]],
                       require_json: bool = False, temperature: float = 0.1) -> str:
        """按 messages 列表调用 LLM（阶段 3 新增，配合 prompts.py 的构造器）"""
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

    def _call_llm(self, system_prompt: str, user_prompt: str,
                  require_json: bool = False, temperature: float = 0.1) -> str:
        """保留原签名，内部转成 messages，兼容既有调用方"""
        return self._call_messages(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            require_json=require_json,
            temperature=temperature,
        )


class ExecutionPlan(BaseModel):
    strategy_queue: List[str]


class AgenticNodesOperator(LegalLLMBase):
    """封装 Extractor/Reasoner/Generator 的 Prompt 逻辑"""

    def generate_plan(self, user_query: str) -> list:
        """
        生成双层蓝图 P_q = {S_q, C_q}，返回展平后的可执行任务队列。
        兼容旧格式：若 LLM 仍返回扁平列表，自动转换。
        """
        # 阶段 3：Prompt 来自 prompts.META_PLANNER_SIMPLE，不再内联副本
        raw = self._call_messages(
            prompts.build_meta_planner_messages(user_query, simple=True),
            require_json=True,
        )
        try:
            data = json.loads(raw)
            # 尝试解析双层蓝图
            import sys, os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
            from double_layer_plan import parse_double_layer_plan
            plan = parse_double_layer_plan(data)
            flat_queue = plan.to_flat_task_queue(respect_deps=True)
            logger.info(f"双层蓝图解析成功: {len(plan.skeleton.nodes)} 节点, 拓扑序展开")
            return flat_queue
        except Exception as e:
            logger.warning(f"双层蓝图解析失败 ({e}), 尝试旧格式")
            try:
                # 兼容旧扁平格式
                data = json.loads(raw)
                if "task_queue" in data:
                    from double_layer_plan import parse_double_layer_plan
                    plan = parse_double_layer_plan(data)
                    return plan.to_flat_task_queue(respect_deps=True)
                if "strategy_queue" in data:
                    return [{"task_desc": t, "engine": "GRAPH_TRAVERSAL", "rationale": ""}
                            for t in data["strategy_queue"]]
            except Exception:
                pass
            logger.error(f"Meta-Planner 输出完全无法解析: {raw[:200]}")
            return [{"task_desc": user_query, "engine": "GRAPH_TRAVERSAL", "rationale": "兜底"}]

    def extract_facts(self, current_sub_task: str, raw_retrieved_docs: str,
                      original_query: Optional[str] = None) -> str:
        # 阶段 3：Prompt 来自 prompts.EXTRACTOR_SYSTEM
        return self._call_messages(
            prompts.build_extractor_messages(
                current_sub_task, raw_retrieved_docs, original_query=original_query)
        )

    def reason(self, current_sub_task: str, extracted_facts: str) -> str:
        # 阶段 3：Prompt 来自 prompts.REASONER_SYSTEM
        return self._call_messages(
            prompts.build_reasoner_messages(current_sub_task, extracted_facts)
        )

    def generate_final_report(self, user_query: str, accumulated_context: List[Dict]) -> str:
        # 阶段 3：Prompt 与证据链拼法都来自 prompts.py
        return self._call_messages(
            prompts.build_generator_messages(user_query, accumulated_context),
            temperature=0.3,
        )

    # ====================== 新增方法 (最小侵入) ======================
    def grade_facts(self, task_desc: str, docs: str) -> dict:
        """
        Grader：强制 LLM 输出 JSON，包含 status (sufficient/partial/irrelevant)
        如果调用失败，退回默认 irrelevant 状态，保证鲁棒性
        """
        # 阶段 3：Prompt 来自 prompts.GRADER_SYSTEM，与 grader_worker 共用同一份
        raw = self._call_messages(
            prompts.build_grader_messages(task_desc, docs), require_json=True
        )
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
        # 阶段 3：Prompt 来自 prompts.REPLANNER_SYSTEM，与 replanner_worker 共用同一份
        raw = self._call_messages(
            prompts.build_replanner_messages(
                original_query=query,
                global_facts=global_facts,
                retry_context={"fail_count": fail_count, "fail_log": fail_log},
                schema_json='{"task_queue": [{"task_desc": "...", "engine": '
                            '"GRAPH_TRAVERSAL", "rationale": "..."}]}',
            ),
            require_json=True,
            temperature=0.4,
        )
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
        # 阶段 3：Prompt 来自 prompts.CODE_GENERATOR_SYSTEM，修正段由构造器拼接
        return self._call_messages(
            prompts.build_code_generator_messages(
                user_query, reasoning_chain, error_context),
            temperature=0.1,
        )

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
                    # 阶段 10：图内顶点名恒为 str(int)，FAISS 给的是 int64
                    node = self.legal_graph.graph.vs.find(name=vname(nid))
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
# 第四部分：真实 Planner
# ============================================================================
# 阶段 2：原先这里是写死的假 Key（"sk-your-real-api-key"）加模块级客户端，
# 结果是 import soul 就会持有一个无效凭据，且真实 Key 无法通过配置注入。
# 改为惰性工厂，统一从 config.yaml 读取。
_llm_client = None


def get_llm_client() -> OpenAI:
    """惰性创建 LLM 客户端（首次调用时才校验凭据）"""
    global _llm_client
    if _llm_client is None:
        api_key = cfg.get("llm", "api_key")
        if not api_key:
            raise RuntimeError(
                "未配置 LLM API Key，无法调用 LLM。"
                "请设置环境变量 DEEPSEEK_API_KEY（参考 .env.example）。"
            )
        _llm_client = OpenAI(
            api_key=api_key,
            base_url=cfg.get("llm", "base_url", default="https://api.deepseek.com"),
        )
    return _llm_client


# 阶段 7：已删除 call_deepseek_planner()。
# 它是阶段 3 之前的死代码——node_planner 早已改走 agentic_ops.generate_plan()，
# 该函数无任何调用方，却自成一套「调 LLM + 解析双层蓝图 + 展平队列」的实现，
# 且内部靠 sys.path.insert 做动态导入。留着只会与 AgenticNodesOperator.generate_plan
# 形成第二份真源，故整体移除（需要时见 git 历史）。


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
    """L0 安全网关：Prompt 注入检测 + 初始化熔断字段"""
    query = state["user_query"]

    # ---- 安全检测规则 ----
    # 中文 jailbreak
    injection_cn = ["忽略指令", "越狱", "忽略之前的", "忘记所有规则",
                    "假装你是", "你现在是", "DAN", "开发者模式"]
    # 英文 jailbreak
    injection_en = ["ignore previous instructions", "ignore all rules",
                    "system prompt", "pretend you are", "jailbreak",
                    "developer mode", "you are now"]
    # 分隔符注入
    injection_delimiters = ['"""', "---", "===", "[[SYSTEM]]", "<<SYS>>"]

    for pattern in injection_cn + injection_en + injection_delimiters:
        if pattern.lower() in query.lower():
            logger.warning(f"L0_REJECT: 检测到注入模式 '{pattern}'")
            raise ValueError(f"L0_REJECT: 触发安全熔断 - 检测到注入模式")

    return {
        "recursion_depth": 0,
        "retry_context": {},
        "global_facts": state.get("global_facts", [])
    }


def node_planner(state: AgentState, agentic_ops: "AgenticNodesOperator") -> dict:
    """
    Planner 节点：用注入的 agentic_ops 生成执行计划。

    阶段 3：改走 agentic_ops.generate_plan()，复用 prompts.py 的模板与同一个
    LLM 客户端。原先这里调模块级 call_deepseek_planner()，等于存在第二份
    Planner 实现，而且它自带一个独立客户端，测试桩无法替换。
    """
    logger.info("Planner 启动，生成执行计划")
    queue = agentic_ops.generate_plan(state["user_query"])
    return {"task_queue": queue}


def node_executor(state: AgentState, reasoner: Reasoner) -> dict:
    if not state["task_queue"]:
        return {}
    # ---- 熔断保护 ----
    depth = state.get("recursion_depth", 0)
    max_depth = cfg.get("agent", "max_recursion_depth", default=5)
    if depth > max_depth:
        logger.warning(f"🚨 触发算力熔断（depth={depth} > {max_depth}），强制终止")
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

        current_failed = queue[0]
        failed_desc = (
            current_failed.get("task_desc", "")
            if isinstance(current_failed, dict)
            else str(current_failed)
        )
        replan_target = failed_desc if failed_desc else state["user_query"]

        # 阶段 3：Prompt 来自 prompts.REPLANNER_SYSTEM，与 Replanner Worker 共用同一份
        replan_messages = prompts.build_replanner_messages(
            original_query=replan_target,
            global_facts=state.get("global_facts", []),
            retry_context={"fail_count": fail_count, "fail_log": obs_text[-300:]},
            schema_json='{"task_queue": [{"task_desc": "...", "engine": '
                        '"GRAPH_TRAVERSAL", "rationale": "..."}]}',
        )
        try:
            # 使用惰性客户端（兼容 soul.py 独立运行时无 agentic_ops 的场景）
            resp = get_llm_client().chat.completions.create(
                model=cfg.get("llm", "judge_model", default="deepseek-chat"),
                messages=replan_messages,
                response_format={"type": "json_object"},
                temperature=0.4, max_tokens=2048
            )
            data = json.loads(resp.choices[0].message.content)
            new_tasks = []
            for t in data.get("task_queue", []):
                new_tasks.append({
                    "task_desc": t.get("task_desc", str(t)),
                    "engine": t.get("engine", "GRAPH_TRAVERSAL"),
                    "rationale": t.get("rationale", "")
                })
            if not new_tasks:
                new_tasks = [{"task_desc": f"全局检索: {replan_target[:40]}", "engine": "GLOBAL_DENSE_WORMHOLE", "rationale": "兜底"}]
        except Exception as e:
            logger.error(f"Replanner LLM 调用异常: {e}")
            new_tasks = [{"task_desc": replan_target[:60], "engine": "GLOBAL_DENSE_WORMHOLE", "rationale": f"异常降级: {str(e)[:50]}"}]

        # 新任务替换队列头，保留队列尾部（其他未执行的子任务）
        new_queue = new_tasks + queue[1:]
    else:
        # 情况3：硬规则补充
        # 规则表来自 replanner_rules.py，与 Replanner Worker 共用同一份。
        # 与 Worker 路径的差别：这里只做"追加"，不做 LLM 重规划。
        new_queue = list(queue)
        hard_tasks = apply_hard_rules(obs_text, queue, logger)
        if hard_tasks:
            logger.info(f"硬规则补充 {len(hard_tasks)} 条任务")
            new_queue = hard_tasks + new_queue

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
    """
    执行沙箱阶段生成的代码。

    阶段 5：Docker 优先 + 本进程降级的逻辑统一收敛到
    legal_sandbox/sandbox_exec.py，本节点只负责"按熔断策略决定要不要重试"。
    """
    logger.info("🐳 沙箱阶段: 执行计算代码")

    code = state.get("generated_code", "")
    if not code:
        return {"retry_context": {"sandbox_error": "无代码可执行"}}

    outcome = run_code_once(code, session_id=state.get("sandbox_session_id"))

    if outcome["error"]:
        logger.warning(f"沙箱执行报错（via={outcome['via']}）: {outcome['error']}")
        retries = state.get("sandbox_retries", 0) + 1
        max_retries = cfg.get("sandbox", "max_retries", default=3)
        if retries < max_retries:
            return {
                "retry_context": {"sandbox_error": outcome["error"]},
                "sandbox_retries": retries,
                "sandbox_session_id": outcome["session_id"],
            }
        # 重试耗尽：必须清掉 sandbox_error 并把计数钉在 max_retries。
        # 否则 route_after_execute 的判定条件（retry_context 里残留 sandbox_error
        # 且 sandbox_retries 未推进）会一直成立，导致 ExecuteCode↔WriteCode 死循环。
        return {
            "calc_result": f"沙箱多次执行失败: {outcome['error']}",
            "sandbox_session_id": outcome["session_id"],
            "retry_context": {},
            "sandbox_retries": max_retries,
        }

    logger.info(f"沙箱执行成功（via={outcome['via']}）: {str(outcome['calc_result'])[:100]}")
    return {
        "calc_result": outcome["calc_result"],
        "sandbox_session_id": outcome["session_id"],
    }


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
    """清理沙箱资源（走 sandbox_exec 的统一入口，幂等）"""
    session_id = state.get("sandbox_session_id")
    if session_id:
        destroy_sandbox_session(session_id)
    return {"sandbox_session_id": None}


# ============================================================================
# 第六部分：组装 LangGraph 图（含沙箱节点）
# ============================================================================
def build_plan_replan_agent(reasoner: Reasoner, agentic_ops: AgenticNodesOperator):
    builder = StateGraph(AgentState)

    # 闭包注入依赖
    def l0_gateway(state): return node_l0_gateway(state)
    def planner(state): return node_planner(state, agentic_ops)
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
        # 已有结果（执行成功、或重试已耗尽并放弃）-> 直接注入。
        # 这是终止条件的权威判据，避免 retry_context 残留导致死循环。
        if state.get("calc_result"):
            return "InjectResult"
        retry_ctx = state.get("retry_context", {})
        sandbox_retries = state.get("sandbox_retries", 0)
        max_retries = cfg.get("sandbox", "max_retries", default=3)
        if "sandbox_error" in retry_ctx and sandbox_retries < max_retries:
            logger.info(f"沙箱重试 {sandbox_retries}/{max_retries}")
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
    # Windows 控制台默认 GBK，而观察记录里含 emoji（如 ⚠️/🐳），
    # 直接 print 会抛 UnicodeEncodeError。这里把 stdout 切到 UTF-8。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # 1. 模拟图谱（实际应用请加载真实数据）
    # 使用 from_config() 保证维度与 config.yaml 的 embedding.dimension 一致
    graph_engine = LegalDenseGraphBuilder.from_config()
    dummy_nodes = [
        {"id": 1, "content": "试用期不符合录用条件可解除合同。", "type": "Raw", "metadata": {"source": "劳动法"}},
        {"id": 2, "content": "违法解除劳动合同按经济补偿标准的二倍支付赔偿金。", "type": "Raw", "metadata": {"source": "劳动法"}},
    ]
    dummy_emb = np.random.randn(len(dummy_nodes), graph_engine.dim).astype(np.float32)
    dummy_emb = dummy_emb / np.linalg.norm(dummy_emb, axis=1, keepdims=True)
    graph_engine.build_initial_graph_batch(dummy_nodes, dummy_emb)

    # 2. 模拟 Embedding 函数
    def mock_embedding(text: str) -> np.ndarray:
        v = np.random.randn(graph_engine.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    # 3. 模拟 AgenticNodesOperator（包含新增方法）
    class MockAgenticOps(AgenticNodesOperator):
        """阶段 3：改为覆写 _call_messages（Prompt 已统一走 prompts.py 构造器）"""

        def _call_messages(self, messages, require_json=False, temperature=0.1):
            system_prompt = messages[0]["content"] if messages else ""
            logger.info(f"Mock LLM called with: {system_prompt[:50]}...")
            if require_json:
                if "事实调查官" in system_prompt:
                    return '{"rationale": "Mock充足", "status": "sufficient", "extracted_facts": ["Mock事实1"]}'
                if "重规划引擎" in system_prompt:
                    return ('{"task_queue": [{"task_desc": "Mock全局检索", '
                            '"engine": "GLOBAL_DENSE_WORMHOLE", "rationale": "Mock"}]}')
                # Meta-Planner：返回双层蓝图
                return ('{"skeleton": {"nodes": [{"id": "1", "abstract": "核实劳动关系", "deps": []}, '
                        '{"id": "2", "abstract": "核实解除合法性", "deps": ["1"]}, '
                        '{"id": "3", "abstract": "计算赔偿金额", "deps": ["2"]}]}, '
                        '"concretion": {"concretions": {"1": "核实劳动关系", '
                        '"2": "核实辞退是否合法", "3": "计算赔偿金额"}}}')
            if "Extractor" in system_prompt:
                return "Mock 提取事实：公司口头辞退，属于违法解除。"
            if "Reasoner" in system_prompt:
                return "Mock 推理：根据事实，应支付双倍赔偿金。"
            if "Generator" in system_prompt:
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