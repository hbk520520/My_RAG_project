from __future__ import annotations
import logging
from config_loader import cfg
import prompts
logger = logging.getLogger("LegalAgentPlanReplan")
import json
from typing import List, Dict, Optional
from pydantic import BaseModel, ValidationError
from openai import OpenAI
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
            if cfg.get("causal_evidence", "enabled", default=False):
                raise
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
            if result:
                return result
            # 解析成功但一个任务都没有 —— 必须喊一声，否则"虫洞重规划"会静默变成
            # 一条通用兜底任务，看起来像正常重规划，实际是 LLM 没按格式输出。
            logger.warning(
                f"虫洞重规划返回空任务列表，改用兜底任务。原始输出: {raw[:200]}")
            return [{"task_desc": "全局检索案情相关法条", "engine": "GLOBAL_DENSE_WORMHOLE",
                     "rationale": "兜底虫洞穿越"}]
        except Exception as e:
            logger.error(f"虫洞重规划输出无法解析（{type(e).__name__}: {e}），"
                         f"改用兜底任务。原始输出: {raw[:200]}")
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

