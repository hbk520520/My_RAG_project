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
    def __init__(self, api_key=None, base_url=None):
        from .llm import create_client
        self.client = create_client(api_key, base_url)
        self.model_name = cfg.get("llm", "judge_model", default="deepseek-chat")

    def _call_messages(self, messages, require_json=False, temperature=0.1):
        from .llm import complete
        return complete(self.client, self.model_name, messages, require_json=require_json, temperature=temperature)

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

    def generate_calculation_plan(self, query, observations):
        from agent_tools.calculator import CalculationPlan
        messages = [{"role": "system", "content":
            "Return a JSON arithmetic plan using only supplied case facts and supported legal formulas. "
            "Never invent missing numeric inputs. State formula applicability assumptions. "
            + json.dumps(CalculationPlan.model_json_schema(), ensure_ascii=False)},
            {"role": "user", "content": json.dumps({"query": query, "observations": observations}, ensure_ascii=False)}]
        return CalculationPlan.model_validate_json(self._call_messages(messages, require_json=True)).model_dump(mode="json")

    def generate_plan(self, user_query):
        from double_layer_plan import parse_double_layer_plan
        raw = self._call_messages(prompts.build_meta_planner_messages(user_query, simple=True), require_json=True)
        data = json.loads(raw)
        if "strategy_queue" in data:
            data = {"task_queue": data["strategy_queue"]}
        plan = parse_double_layer_plan(data)
        queue = []
        by_id = {node.id: node for node in plan.skeleton.nodes}
        for task_id in plan.skeleton.topological_order():
            node = by_id[task_id]
            desc = plan.concretion.concretions.get(task_id, node.abstract)
            queue.append({"task_id": task_id, "task_desc": desc, "deps": node.deps,
                          "engine": "GRAPH_TRAVERSAL", "rationale": node.abstract})
        if not queue:
            raise ValueError("Planner returned an empty plan")
        return queue

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
    def grade_facts(self, task_desc, docs):
        from agent_runtime.contracts import Grade
        messages = prompts.build_grader_messages(task_desc, docs)
        messages[0]["content"] += ('\nIf missing information can only be supplied by the user, set '
                                   'missing_info_kind="user" and ask a concrete question in missing_info. '
                                   'Otherwise use missing_info_kind="evidence". Do not guess user facts.')
        raw = self._call_messages(messages, require_json=True)
        return Grade.model_validate_json(raw).model_dump()

    def replan_with_wormhole(self, query, global_facts, fail_log, fail_count=0):
        from agent_runtime.contracts import TaskSpec
        raw = self._call_messages(prompts.build_replanner_messages(
            original_query=query, global_facts=global_facts,
            retry_context={"fail_count": fail_count, "fail_log": fail_log},
            schema_json='{"task_queue": [{"task_desc": "...", "engine": "GLOBAL_DENSE_WORMHOLE", "rationale": "..."}]}'),
            require_json=True, temperature=0.4)
        tasks = json.loads(raw)["task_queue"]
        if not tasks:
            raise ValueError("Replanner returned an empty task queue")
        return [TaskSpec.model_validate({"task_id": f"replan-{i}", **task}).model_dump()
                for i, task in enumerate(tasks)]

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
