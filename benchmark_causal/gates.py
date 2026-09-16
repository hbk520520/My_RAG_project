"""
硬门禁与评分合成
================
**评分的第一原则：能用 Python 确定性判断的，绝不交给 LLM。**

本模块实现三道确定性门禁，任一不过 → 该题直接 0 分，连裁判都不用请：

  1. 格式合规   —— 响应能否被解析成 `ModelResponse`，结论/因果链是否非空
  2. 引证可信   —— 引用的每条法条**是否存在**，且在**案件发生日是否有效**
                    （复用 `legal_corpus`，含「法不溯及既往」判定）
  3. 结论一致   —— 与 SCM 推导出的标准答案比对：
                    · 金额类：数值精确匹配（容差 1e-2）
                    · 文本类：`ground_truth_terms` 全部出现 + 标准答案里的数字全部出现
                  这三项都是字符串/数值运算，不需要任何模型。

只有**因果链命中率**与**混杂因子是否隔离**是语义判断，才交给 `judge`。
且即便没有裁判（无 API key / 离线测试），也提供一个**确定性词法匹配器**兜底 ——
它比 LLM 裁判弱，但完全可复现，足以支撑回归门禁与单元测试。

得分合成：
    硬门禁不过                     → 0.0
    硬门禁过、无裁判（词法兜底）    → w_reason + w_conf * confounder   （按加权）
    硬门禁过、有裁判                → 同公式，用裁判给的 reasoning_hit_rate
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Sequence

from .legal_corpus import LegalCorpus, parse_citation
from .schemas import (
    AMOUNT_TOLERANCE,
    CaseScore,
    GateResult,
    JudgeVerdict,
    LegalCausalTestCase,
    ModelResponse,
)

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


# ===========================================================================
# 基础工具
# ===========================================================================
def normalize_text(text: str) -> str:
    """归一化：去空白、统一全半角标点，便于确定性比对"""
    t = (text or "").strip()
    t = t.replace("，", ",").replace("。", ".").replace("；", ";").replace("：", ":")
    t = t.replace("（", "(").replace("）", ")").replace("、", ",")
    t = re.sub(r"[\s\u3000]+", "", t)
    return t


def extract_numbers(text: str) -> List[float]:
    """提取文本中的全部数字（用于结论的确定性比对）"""
    return [float(m) for m in _NUM_RE.findall(text or "")]


def _numbers_covered(expected: Sequence[float], actual: Sequence[float],
                      tol: float = AMOUNT_TOLERANCE) -> bool:
    """expected 中每个数字都要在 actual 中找到（容差内）"""
    for e in expected:
        if not any(abs(e - a) <= tol for a in actual):
            return False
    return True


# ===========================================================================
# 因果链匹配（语义判断的接口 + 确定性兜底实现）
# ===========================================================================
class ReasoningMatcher(Protocol):
    """因果链质量匹配器。LLM 裁判与词法匹配器都实现这个协议。"""

    def match(self, case: LegalCausalTestCase, response: ModelResponse) -> JudgeVerdict:
        ...


class LexicalMatcher:
    """
    **确定性**兜底匹配器：把每个关键推理节点拆成「特征词 + 数字」，
    检查它们是否都出现在被测因果链里。

    注意它的定位：比 LLM 裁判弱（没有语义泛化能力），但
      · 完全可复现，可进回归门禁
      · 离线可跑（无 API key 也能评）
    因此它同时用作"无裁判时的兜底"与"裁判的对照基线"。
    """

    def __init__(self, hit_threshold: float = 0.6):
        self.hit_threshold = hit_threshold

    def _point_hit(self, point: str, chain_text: str, chain_numbers: List[float]) -> bool:
        nums = extract_numbers(point)
        if nums:
            # 有数字的节点：以数字是否出现为准（金额是硬信号）
            return _numbers_covered(nums, chain_numbers)
        # 纯文本节点：取长度 >=2 的中文片段做包含判断
        frags = [f for f in re.split(r"[，,;；:：/()（）\s]+", point) if len(f) >= 2]
        return any(normalize_text(f) in chain_text for f in frags) if frags else False

    def match(self, case: LegalCausalTestCase, response: ModelResponse) -> JudgeVerdict:
        chain_text = normalize_text(" ".join(response.causal_chain))
        chain_numbers = extract_numbers(" ".join(response.causal_chain))
        points = case.key_reasoning_points
        if not points:
            return JudgeVerdict(reasoning_hit_rate=1.0,
                                reason="该题未声明关键推理节点，不扣分")
        hits = [p for p in points if self._point_hit(p, chain_text, chain_numbers)]
        rate = len(hits) / len(points)
        return JudgeVerdict(
            reasoning_hit_rate=rate,
            reason=f"词法匹配命中 {len(hits)}/{len(points)} 个推理节点",
        )


# ===========================================================================
# 硬门禁
# ===========================================================================
def run_hard_gates(case: LegalCausalTestCase,
                   response: Optional[ModelResponse],
                   corpus: LegalCorpus,
                   parse_error: str = "") -> GateResult:
    """
    三道确定性门禁。`response=None` 表示解析失败（由调用方把原因放进 parse_error）。
    """
    errors: List[str] = []
    gate = GateResult()

    # ---- 门禁 1：格式 ----
    if response is None:
        errors.append(parse_error or "响应无法解析为 ModelResponse")
        gate.format_ok = False
    else:
        if not response.conclusion.strip():
            errors.append("conclusion 为空")
        if not response.causal_chain:
            errors.append("causal_chain 为空（本题要求给出因果推理链）")
        gate.format_ok = not errors

    if response is None:
        gate.errors = errors
        return gate

    # ---- 门禁 2：引证可信（存在性 + 时效性）----
    if not response.citations:
        errors.append("未引用任何法条（本项目要求结论必须可溯源到法条）")
        gate.citations_exist = False
        gate.citations_effective = False
    else:
        exist_ok, effect_ok = True, True
        for raw in response.citations:
            parsed = parse_citation(raw)
            if parsed is None:
                exist_ok = False
                errors.append(f"无法解析的法条引用：{raw!r}")
                continue
            law_name, idx = parsed
            from .schemas import LawRef
            ref = LawRef(law_name=law_name,
                         article_no=f"第{idx}条" if idx < 1000 else str(idx))
            check = corpus.check(ref, case.case_date)
            if not check.exists:
                exist_ok = False
                errors.append(f"引用了不存在的法条：{raw}")
            elif not check.effective:
                effect_ok = False
                errors.append(f"{raw} 在案件发生日 {case.case_date} 已失效 —— {check.reason}")
        gate.citations_exist = exist_ok
        gate.citations_effective = effect_ok

    # ---- 门禁 3：结论与标准答案一致 ----
    conclusion = normalize_text(response.conclusion)
    gt_outcome = normalize_text(case.ground_truth_outcome)

    term_ok = all(normalize_text(t) in conclusion for t in case.ground_truth_terms) \
        if case.ground_truth_terms else True
    gt_numbers = extract_numbers(gt_outcome)
    num_ok = _numbers_covered(gt_numbers, extract_numbers(response.conclusion)) \
        if gt_numbers else True
    gate.conclusion_match = bool(term_ok and num_ok)
    if not gate.conclusion_match:
        missing_terms = [t for t in case.ground_truth_terms
                         if normalize_text(t) not in conclusion]
        missing_nums = [n for n in gt_numbers
                        if not any(abs(n - a) <= AMOUNT_TOLERANCE
                                   for a in extract_numbers(response.conclusion))]
        if missing_terms:
            errors.append(f"结论缺少关键要素：{missing_terms}")
        if missing_nums:
            errors.append(f"结论缺少关键数值：{missing_nums}")

    # ---- 门禁 3b：金额精确匹配（金额类题）----
    if case.ground_truth_amount is None:
        gate.amount_match = None
    else:
        got = response.amount
        if got is None:
            nums = extract_numbers(response.conclusion)
            got = nums[0] if len(nums) == 1 else (
                min(nums, key=lambda x: abs(x - case.ground_truth_amount)) if nums else None)
        if got is None:
            gate.amount_match = False
            errors.append("金额类题未给出金额")
        else:
            gate.amount_match = abs(got - case.ground_truth_amount) <= AMOUNT_TOLERANCE
            if not gate.amount_match:
                errors.append(
                    f"金额不匹配：应为 {case.ground_truth_amount:g} 元，实得 {got:g} 元"
                )

    gate.errors = errors
    return gate


# ===========================================================================
# 评分合成
# ===========================================================================
@dataclass(frozen=True)
class ScoringWeights:
    """软评分权重（硬门禁是 0/1 门禁，不参与加权）"""

    reasoning: float = 0.6
    confounder: float = 0.4

    def __post_init__(self):
        if self.reasoning < 0 or self.confounder < 0:
            raise ValueError("权重不能为负")
        if abs(self.reasoning + self.confounder - 1.0) > 1e-9:
            raise ValueError("权重之和必须为 1.0")


DEFAULT_WEIGHTS = ScoringWeights()


def score_case(case: LegalCausalTestCase,
               response: Optional[ModelResponse],
               corpus: LegalCorpus,
               matcher: Optional[ReasoningMatcher] = None,
               parse_error: str = "",
               weights: ScoringWeights = DEFAULT_WEIGHTS) -> CaseScore:
    """
    给一道题打分。

    :param matcher: 因果链匹配器；None 时使用确定性的 `LexicalMatcher`
    """
    gate = run_hard_gates(case, response, corpus, parse_error=parse_error)

    if not gate.passed:
        return CaseScore(
            case_id=case.case_id,
            source_case_id=case.source_case_id,
            causal_level=case.causal_level,
            hard_gate=gate,
            judge=None,
            final_score=0.0,
            final_reason="硬门禁未通过：" + "；".join(gate.errors),
        )

    matcher = matcher or LexicalMatcher()
    verdict = matcher.match(case, response)

    # 混杂因子项：只有 L3 且该题**确实存在**混杂因子时才计分
    has_confounder = case.has_confounder
    if case.causal_level == 3 and has_confounder and verdict.confounder_isolated is None:
        # 裁判没给（如词法兜底）→ 该项不计入分母，避免把不存在的信息算成扣分
        score = verdict.reasoning_hit_rate
        reason = f"无裁判提供混杂因子判定，仅按因果链命中率计分：{verdict.reason}"
    elif verdict.confounder_isolated is None:
        score = verdict.reasoning_hit_rate
        suffix = ("该题不涉及混杂因子隔离，仅按因果链命中率计分"
                  if not has_confounder
                  else "裁判未提供混杂因子判定，仅按因果链命中率计分")
        reason = f"{verdict.reason}；{suffix}"
    else:
        score = (weights.reasoning * verdict.reasoning_hit_rate
                 + weights.confounder * float(verdict.confounder_isolated))
        reason = verdict.reason

    return CaseScore(
        case_id=case.case_id,
        source_case_id=case.source_case_id,
        causal_level=case.causal_level,
        hard_gate=gate,
        judge=verdict,
        final_score=round(min(max(score, 0.0), 1.0), 6),
        final_reason=reason,
    )


# ===========================================================================
# Case 级聚合：因果一致性惩罚
# ===========================================================================
def aggregate_by_case(scores: Iterable[CaseScore]) -> dict:
    """
    **因果一致性率（Causal Consistency Rate）**

    同一 `source_case_id` 下的 L1/L2/L3 必须**全部通过**，该 Case 才记 1；
    任一层不过则该 Case 记 0 —— 因为「L1 对、L3 错」说明是瞎猫碰上死耗子。

    通过判定：`final_score >= threshold`（默认 0.8）。

    返回：
      {
        "cases": {source_case_id: {...逐层得分...}},
        "consistency_rate": 0.0~1.0,
        "levels": {1: 平均分, 2: ..., 3: ...},
        "level_pass_rate": {1: ..., 2: ..., 3: ...},
      }
    """
    by_case: dict = {}
    for s in scores:
        by_case.setdefault(s.source_case_id, {})[int(s.causal_level)] = s

    threshold = 0.8
    consistent = 0
    detail: dict = {}
    level_scores: dict = {1: [], 2: [], 3: []}

    for cid, levels in by_case.items():
        all_pass = bool(levels) and all(s.final_score >= threshold for s in levels.values())
        consistent += int(all_pass)
        for lvl, s in levels.items():
            level_scores.setdefault(lvl, []).append(s.final_score)
        detail[cid] = {
            "levels": {lvl: round(s.final_score, 4) for lvl, s in sorted(levels.items())},
            "all_pass": all_pass,
        }

    def _avg(xs: Sequence[float]) -> float:
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    def _pass_rate(xs: Sequence[float]) -> float:
        return round(sum(1 for x in xs if x >= threshold) / len(xs), 4) if xs else 0.0

    return {
        "cases": detail,
        "case_count": len(by_case),
        "consistency_rate": round(consistent / len(by_case), 4) if by_case else 0.0,
        "levels": {lvl: _avg(v) for lvl, v in level_scores.items()},
        "level_pass_rate": {lvl: _pass_rate(v) for lvl, v in level_scores.items()},
        "threshold": threshold,
    }
