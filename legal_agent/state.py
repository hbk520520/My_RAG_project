from __future__ import annotations
import operator
from typing import TypedDict, Annotated, List, Dict, Any, Optional
class AgentState(TypedDict):
    intent: str
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
    prediction_cutoff: Optional[str]                     # 可选：预测截止日（不使用系统当前时间猜测）
    causal_targets: Optional[List[str]]                  # 可选：图中明确的目标实体
    causal_evidence: Annotated[List[dict], operator.add]  # 可选：每任务因果证据审计
    citations: List[dict]                               # 可选：从已读原文校验的引用
    completed_tasks: List[dict]
    user_facts: List[str]
    pending_question: Optional[str]
    execution_status: str
    calculation_ok: bool
    calculation_mode: str
    calculation_plan: dict
    task_retry_contexts: Dict[str, dict]
    task_results: Dict[str, dict]
    pending_task_id: Optional[str]
    source_evidence: Annotated[List[dict], operator.add]
    task_user_facts: Dict[str, List[str]]
