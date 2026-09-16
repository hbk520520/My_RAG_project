"""
因果评测 —— 数据契约层
======================
把「评测什么、怎么作答、怎么判」固化成 Pydantic Schema，作为唯一真源。

设计原则（继承本项目已确立的纪律）：
  1. **能被 Python 确定性判断的，绝不交给 LLM** —— 见 gates.py；
     Judge 只负责语义类判断（因果链命中率 / 混杂因子是否隔离）。
  2. **ground truth 由 SCM 推导，不由人写、不由 LLM 说** —— 见 scm.py。
     L1/L2/L3 三层标准答案同源，天然自洽。
  3. **格式违规要响** —— 解析失败返回明确的 error，不静默当 0 分放过去。

术语：
  · 因果之梯（Pearl）：L1 观测(Seeing) / L2 干预(Doing) / L3 反事实(Counterfactual)
"""
from __future__ import annotations

import json
import re
from enum import IntEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# 金额比较容差（元）。法律赔偿金是精确值，不需要模糊匹配。
AMOUNT_TOLERANCE = 0.01


class CausalLevel(IntEnum):
    """Pearl 因果之梯三层"""

    SEEING = 1          # 观测：给定案情，推结论
    DOING = 2           # 干预：do(X=x) 后，推结论
    COUNTERFACTUAL = 3  # 反事实：已发生的事实若被改变，结论会怎样


class LawRef(BaseModel):
    """一条法条引用。硬门禁用它查「是否存在」与「在案件发生日是否有效」。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    law_name: str = Field(..., description="法律名称，如「劳动合同法」")
    article_no: str = Field(..., description="条号，如「第八十七条」")

    @property
    def key(self) -> str:
        return f"{self.law_name}{self.article_no}"

    def __str__(self) -> str:  # pragma: no cover - 便于日志
        return self.key


class LegalCausalTestCase(BaseModel):
    """
    一道因果评测题。

    三层题**共享** `source_case_id` 与 `scm_assignments`，只是 `causal_level` 与
    `prompt` 不同 —— 这正是「因果一致性惩罚」能在 Case 级聚合的前提。
    """

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(..., description="全局唯一题目 ID")
    source_case_id: str = Field(..., description="同一 SCM 案例的分组 ID（L1/L2/L3 共享）")
    causal_level: CausalLevel
    domain: str = Field(..., description="SCM 名称，如 labor_unlawful_dismissal")

    case_date: str = Field(..., description="案件事实发生日期 YYYY-MM-DD（决定适用法版本）")
    prompt: str = Field(..., min_length=1)

    # ---- ground truth：全部由 SCM 推导 ----
    ground_truth_outcome: str
    ground_truth_amount: Optional[float] = Field(
        default=None, description="需精确匹配的金额（元）；无金额类结论时为 None"
    )
    ground_truth_terms: List[str] = Field(
        default_factory=list,
        description="必须出现在结论里的关键要素（如「违法解除」「二倍」），用于**确定性**结论判定",
    )
    key_reasoning_points: List[str] = Field(
        default_factory=list, description="必须命中的因果推理节点（SCM 变量实例化链）"
    )
    applicable_laws: List[LawRef] = Field(
        default_factory=list, description="本题应适用的法条（来自因果边的锚点）"
    )

    # ---- 审计与复算 ----
    scm_assignments: Dict[str, Any] = Field(
        default_factory=dict, description="本题的变量取值，可用 SCM 复算出 ground truth"
    )
    intervention: Dict[str, Any] = Field(
        default_factory=dict, description="L2 的 do(·) 目标；L1/L3 为空"
    )
    counterfactual_flip: Dict[str, Any] = Field(
        default_factory=dict, description="L3 被翻转的变量；L1/L2 为空"
    )
    has_confounder: bool = Field(
        default=False,
        description="本题的 treatment→outcome 之间是否真存在混杂因子（由 SCM 判定）。"
                    "为 False 时 L3 的「隔离混杂因子」一项应判为**不适用**，而不是算扣分。",
    )
    perturbation: Optional[str] = Field(
        default=None, description="抗扰动题的注入内容（无关细节）；未注入为 None"
    )

    @field_validator("case_date")
    @classmethod
    def _check_date(cls, v: str) -> str:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v or ""):
            raise ValueError(f"case_date 需形如 YYYY-MM-DD，实际是 {v!r}")
        return v

    @field_validator("key_reasoning_points")
    @classmethod
    def _check_points(cls, v: List[str]) -> List[str]:
        if any(not p.strip() for p in v):
            raise ValueError("key_reasoning_points 不能含空字符串")
        return v


class ModelResponse(BaseModel):
    """
    被测 Agent 的作答规范（强制结构化输出）。

    注意：被测对象是**整个 Agent**，所以这些字段应从 Agent 的终态/trace 里映射过来，
    而不是要求模型一次性吐 JSON —— 由调用方（runner）负责适配。
    """

    model_config = ConfigDict(extra="ignore")  # 对模型宽容：多余字段忽略

    conclusion: str = Field(..., description="结论")
    causal_chain: List[str] = Field(default_factory=list, description="因果推理步骤（CoT）")
    citations: List[str] = Field(
        default_factory=list, description="引用的法条，如「劳动合同法第八十七条」"
    )
    amount: Optional[float] = Field(default=None, description="金额类结论的数值（元）")

    @staticmethod
    def parse_llm_json(raw: str) -> "ModelResponse":
        """
        容错解析模型输出。

        真实模型经常把 JSON 包在 ```json 围栏里，或在前后带解释文字。
        这里先剥围栏、再尝试提取第一个平衡的 JSON 对象。
        """
        if raw is None:
            raise ValueError("模型输出为空")
        text = raw.strip()

        fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = _extract_first_json_object(text)
            if data is None:
                raise ValueError(f"无法从模型输出中提取 JSON：{raw[:200]!r}")
        return ModelResponse.model_validate(data)


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    """扫描出第一个括号平衡的 JSON 对象（跳过字符串内的括号）"""
    depth, start, in_str, escape = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


class GateResult(BaseModel):
    """
    硬门禁结果（**纯 Python 判定**，不看 LLM 脸色）。

    任一门禁不过 → 本题直接 0 分，连裁判都不用请。
    """

    model_config = ConfigDict(extra="forbid")

    format_ok: bool = False
    citations_exist: bool = Field(default=False, description="引用的法条是否**存在**")
    citations_effective: bool = Field(
        default=False, description="引用的法条在案件发生日是否**有效**（法不溯及既往）"
    )
    conclusion_match: bool = Field(default=False, description="结论与 SCM 标准结论是否一致")
    amount_match: Optional[bool] = Field(
        default=None, description="金额是否精确匹配；非金额类题为 None"
    )
    errors: List[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        if not (self.format_ok and self.citations_exist and self.citations_effective
                and self.conclusion_match):
            return False
        return self.amount_match is not False


class JudgeVerdict(BaseModel):
    """
    软评分（LLM 裁判只负责这两项**语义**判断）。

    设计上刻意不让裁判给"总分"：总分由 Python 按权重合成，
    延续本项目「不让 LLM 做确定性计算」的原则。
    """

    model_config = ConfigDict(extra="ignore")

    reasoning_hit_rate: float = Field(..., ge=0.0, le=1.0, description="因果链命中率")
    confounder_isolated: Optional[int] = Field(
        default=None, ge=0, le=1,
        description="L3 专用：是否成功隔离混杂因子（1/0）；非 L3 为 None",
    )
    reason: str = Field(default="", description="裁判给出的简短理由")


class CaseScore(BaseModel):
    """一道题最终得分（硬门禁 + 软评分合成）"""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    source_case_id: str
    causal_level: CausalLevel
    hard_gate: GateResult
    judge: Optional[JudgeVerdict] = None
    final_score: float = Field(..., ge=0.0, le=1.0)
    final_reason: str = ""
