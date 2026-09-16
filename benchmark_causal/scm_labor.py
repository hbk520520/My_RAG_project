"""
劳动争议领域的结构因果模型
==========================
第一批两个 SCM，刻意覆盖两种拓扑：

  A. `labor_overtime_claim`  —— **含混杂因子**
     赶工强度 Z 同时影响「加班时长 X」与「夜班津贴 M2」，而两者都进总额 Y。
     Z 是 X→Y 的混杂因子，因此能做真正的 L3「隔离混杂因子」题。

  B. `labor_unlawful_dismissal` —— **不含混杂因子**
     处理变量是根节点，没有共同祖先。用来验证评测器能正确把
     `confounder_isolated` 标为 **不适用（None）**，而不是硬考一个不存在的东西。

每条因果边都标注了性质：
  · `KIND_LEGAL`   —— 法条规定的后果/计算（**必须**带法条锚点）
  · `KIND_FACTUAL` —— 案情内的事实关联（无法条依据，给出说明即可）
这个区分是为了不做"伪溯源"：把"赶工导致加班多"硬挂一条法条是假的可审计性。
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping

from .schemas import LawRef
from .scm import (
    KIND_FACTUAL,
    KIND_LEGAL,
    ROLE_CONFOUNDER,
    ROLE_EXOGENOUS,
    ROLE_MEDIATOR,
    ROLE_OUTCOME,
    ROLE_TREATMENT,
    Edge,
    Equation,
    StructuralCausalModel,
    Variable,
)

# ---------------------------------------------------------------------------
# 法条锚点（名称与条号必须与 benchmark_causal/data/legal_corpus.jsonl 一致）
# ---------------------------------------------------------------------------
LABOR_LAW_44 = LawRef(law_name="劳动法", article_no="第四十四条")          # 加班费 150%/200%/300%
LCL_10 = LawRef(law_name="劳动合同法", article_no="第十条")                # 应订立书面合同
LCL_39 = LawRef(law_name="劳动合同法", article_no="第三十九条")            # 过失性解除
LCL_47 = LawRef(law_name="劳动合同法", article_no="第四十七条")            # 经济补偿 N
LCL_82 = LawRef(law_name="劳动合同法", article_no="第八十二条")            # 未签合同二倍工资
LCL_87 = LawRef(law_name="劳动合同法", article_no="第八十七条")            # 违法解除 2N


def _severance_months(dismissal_ground: str, years: float) -> float:
    """
    经济补偿月数 N（《劳动合同法》第四十七条），违法解除按二倍（第八十七条）。

      满一年 → 1 个月/年；剩余满 6 个月 → 计 1 年；剩余不满 6 个月 → 计 0.5 个月
      过失性解除（第三十九条）→ 无经济补偿
    """
    if dismissal_ground == "employee_fault":
        return 0.0
    whole = float(int(years))
    rest = years - whole
    if rest >= 0.5:
        months = whole + 1
    elif rest > 0:
        months = whole + 0.5
    else:
        months = whole
    return months * (2.0 if dismissal_ground == "unlawful" else 1.0)


def _double_wage(months_without_contract: float, monthly_wage: float) -> float:
    """
    未订立书面劳动合同的二倍工资**差额**（《劳动合同法》第八十二条）。

    条文：自用工之日起**超过一个月不满一年**未订立书面合同的，每月支付二倍工资。
    故差额 = 超出 1 个月的月数 × 月工资，且最多 11 个月。
    """
    billable = min(max(months_without_contract - 1.0, 0.0), 11.0)
    return billable * monthly_wage


# ===========================================================================
# A. 加班费诉求（含混杂因子）
# ===========================================================================
def build_overtime_scm() -> StructuralCausalModel:
    variables = [
        Variable("project_crunch", "项目赶工强度", ROLE_CONFOUNDER,
                 domain=("normal", "crunch"),
                 description="同时影响加班时长与夜班津贴 —— 是 X→Y 的混杂因子"),
        Variable("overtime_hours", "工作日加班时长", ROLE_TREATMENT,
                 unit="小时", description="处理变量 X"),
        Variable("hourly_wage", "折算小时工资", ROLE_EXOGENOUS, unit="元/小时"),
        Variable("overtime_pay", "加班费", ROLE_MEDIATOR, unit="元"),
        Variable("night_shift_allowance", "夜班津贴", ROLE_MEDIATOR, unit="元"),
        Variable("total_claim", "可主张总额", ROLE_OUTCOME, unit="元",
                 description="结果变量 Y"),
    ]

    edges = [
        Edge("project_crunch", "overtime_hours",
             "赶工期间排班延长，加班时长随之上升", kind=KIND_FACTUAL),
        Edge("project_crunch", "night_shift_allowance",
             "赶工期间大量排夜班，产生夜班津贴", kind=KIND_FACTUAL),
        Edge("overtime_hours", "overtime_pay",
             "工作日延长工作时间，按不低于工资的百分之一百五十支付",
             article_refs=(LABOR_LAW_44,), kind=KIND_LEGAL),
        Edge("hourly_wage", "overtime_pay",
             "加班费以折算小时工资为基数", article_refs=(LABOR_LAW_44,), kind=KIND_LEGAL),
        Edge("overtime_pay", "total_claim",
             "加班费是可主张金额的组成部分", kind=KIND_FACTUAL),
        Edge("night_shift_allowance", "total_claim",
             "夜班津贴是可主张金额的组成部分", kind=KIND_FACTUAL),
    ]

    equations = [
        Equation(
            target="overtime_hours",
            fn=lambda v: 60.0 if v["project_crunch"] == "crunch" else 10.0,
            formula="赶工=60 小时，否则=10 小时",
            describe=lambda v, val: (
                f"项目为「{'赶工' if v['project_crunch'] == 'crunch' else '正常'}」，"
                f"故工作日加班时长为 {val:g} 小时"
            ),
            kind=KIND_FACTUAL,
        ),
        Equation(
            target="overtime_pay",
            fn=lambda v: round(v["overtime_hours"] * v["hourly_wage"] * 1.5, 2),
            formula="加班费 = 加班时长 × 小时工资 × 150%",
            article_refs=(LABOR_LAW_44,),
            describe=lambda v, val: (
                f"加班费 = {v['overtime_hours']:g} 小时 × {v['hourly_wage']:g} 元/小时 × 150% = {val:g} 元"
            ),
            kind=KIND_LEGAL,
        ),
        Equation(
            target="night_shift_allowance",
            fn=lambda v: 300.0 if v["project_crunch"] == "crunch" else 0.0,
            formula="赶工=300 元，正常=0 元",
            describe=lambda v, val: (
                f"夜班津贴 {val:g} 元（{'赶工期间排夜班' if val else '无夜班'}）"
            ),
            kind=KIND_FACTUAL,
        ),
        Equation(
            target="total_claim",
            fn=lambda v: round(v["overtime_pay"] + v["night_shift_allowance"], 2),
            formula="可主张总额 = 加班费 + 夜班津贴",
            describe=lambda v, val: (
                f"可主张总额 = 加班费 {v['overtime_pay']:g} + 夜班津贴 "
                f"{v['night_shift_allowance']:g} = {val:g} 元"
            ),
            kind=KIND_FACTUAL,
        ),
    ]

    return StructuralCausalModel(
        name="labor_overtime_claim",
        title="劳动争议 · 加班费与夜班津贴诉求（含混杂因子）",
        variables=variables,
        edges=edges,
        equations=equations,
        outcome="total_claim",
        treatment="overtime_hours",
        outcome_renderer=lambda v: f"劳动者可主张总额为 {v['total_claim']:g} 元",
        notes="混杂因子 project_crunch 同时影响 overtime_hours 与 night_shift_allowance，"
              "因此 do(project_crunch=normal) 只切断它对两条路径的输入，"
              "而不会改变已被固定的 overtime_hours —— 这是 L3 隔离题的核心。",
    )


# ===========================================================================
# B. 违法解除赔偿金（不含混杂因子）
# ===========================================================================
def build_unlawful_dismissal_scm() -> StructuralCausalModel:
    variables = [
        Variable("dismissal_ground", "解除事由", ROLE_TREATMENT,
                 domain=("employee_fault", "lawful", "unlawful", "agreed"),
                 description="处理变量 X"),
        Variable("years_of_service", "在本单位工作年限", ROLE_EXOGENOUS, unit="年"),
        Variable("monthly_wage", "解除前十二个月平均月工资", ROLE_EXOGENOUS, unit="元"),
        Variable("months_without_contract", "未订立书面合同的月数", ROLE_EXOGENOUS, unit="个月"),
        Variable("compensation_months", "经济补偿月数", ROLE_MEDIATOR, unit="个月"),
        Variable("severance_amount", "赔偿金/经济补偿", ROLE_MEDIATOR, unit="元"),
        Variable("double_wage_amount", "未签书面合同二倍工资差额", ROLE_MEDIATOR, unit="元"),
        Variable("total_amount", "应支付总额", ROLE_OUTCOME, unit="元"),
    ]

    edges = [
        Edge("dismissal_ground", "compensation_months",
             "过失性解除无经济补偿；违法解除按经济补偿标准的二倍",
             article_refs=(LCL_39, LCL_87), kind=KIND_LEGAL),
        Edge("years_of_service", "compensation_months",
             "经济补偿按在本单位工作年限计算",
             article_refs=(LCL_47,), kind=KIND_LEGAL),
        Edge("compensation_months", "severance_amount",
             "经济补偿以月数 × 月工资为基数",
             article_refs=(LCL_47,), kind=KIND_LEGAL),
        Edge("monthly_wage", "severance_amount",
             "以解除前十二个月平均月工资为基数",
             article_refs=(LCL_47,), kind=KIND_LEGAL),
        Edge("months_without_contract", "double_wage_amount",
             "超过一个月不满一年未订立书面合同的，每月支付二倍工资",
             article_refs=(LCL_10, LCL_82), kind=KIND_LEGAL),
        Edge("monthly_wage", "double_wage_amount",
             "二倍工资差额以月工资为基数",
             article_refs=(LCL_82,), kind=KIND_LEGAL),
        Edge("severance_amount", "total_amount",
             "应支付总额包含赔偿金/经济补偿", kind=KIND_FACTUAL),
        Edge("double_wage_amount", "total_amount",
             "应支付总额包含二倍工资差额", kind=KIND_FACTUAL),
    ]

    _label = {"employee_fault": "劳动者过失性解除", "lawful": "合法解除",
              "unlawful": "违法解除", "agreed": "协商一致解除"}

    equations = [
        Equation(
            target="compensation_months",
            fn=lambda v: _severance_months(v["dismissal_ground"], v["years_of_service"]),
            formula="N = f(解除事由, 工作年限)；违法解除取 2N；过失性解除取 0",
            article_refs=(LCL_39, LCL_47, LCL_87),
            describe=lambda v, val: (
                f"解除事由为「{_label.get(v['dismissal_ground'], v['dismissal_ground'])}」、"
                f"工作年限 {v['years_of_service']:g} 年 → 经济补偿月数 {val:g} 个月"
            ),
            kind=KIND_LEGAL,
        ),
        Equation(
            target="severance_amount",
            fn=lambda v: round(v["compensation_months"] * v["monthly_wage"], 2),
            formula="赔偿金 = 经济补偿月数 × 月工资",
            article_refs=(LCL_47, LCL_87),
            describe=lambda v, val: (
                f"赔偿金 = {v['compensation_months']:g} 个月 × {v['monthly_wage']:g} 元 = {val:g} 元"
            ),
            kind=KIND_LEGAL,
        ),
        Equation(
            target="double_wage_amount",
            fn=lambda v: round(_double_wage(v["months_without_contract"], v["monthly_wage"]), 2),
            formula="二倍工资差额 = min(max(未签月数 − 1, 0), 11) × 月工资",
            article_refs=(LCL_82,),
            describe=lambda v, val: (
                f"未订立书面合同 {v['months_without_contract']:g} 个月 → 二倍工资差额 {val:g} 元"
            ),
            kind=KIND_LEGAL,
        ),
        Equation(
            target="total_amount",
            fn=lambda v: round(v["severance_amount"] + v["double_wage_amount"], 2),
            formula="应支付总额 = 赔偿金 + 二倍工资差额",
            describe=lambda v, val: (
                f"应支付总额 = {v['severance_amount']:g} + {v['double_wage_amount']:g} = {val:g} 元"
            ),
            kind=KIND_FACTUAL,
        ),
    ]

    return StructuralCausalModel(
        name="labor_unlawful_dismissal",
        title="劳动争议 · 违法解除赔偿金与未签合同二倍工资",
        variables=variables,
        edges=edges,
        equations=equations,
        outcome="total_amount",
        treatment="dismissal_ground",
        outcome_renderer=lambda v: f"用人单位应支付总额为 {v['total_amount']:g} 元",
        notes="处理变量 dismissal_ground 是根节点，不存在共同祖先 —— "
              "因此本 SCM 的 confounder_isolated 应为「不适用」。",
    )


# ===========================================================================
# 注册表
# ===========================================================================
LABOR_SCMS: Dict[str, StructuralCausalModel] = {
    "labor_overtime_claim": build_overtime_scm(),
    "labor_unlawful_dismissal": build_unlawful_dismissal_scm(),
}


def get_scm(name: str) -> StructuralCausalModel:
    if name not in LABOR_SCMS:
        raise KeyError(f"未注册的 SCM: {name}；可用: {sorted(LABOR_SCMS)}")
    return LABOR_SCMS[name]


def all_article_refs() -> List[LawRef]:
    """本项目 SCM 用到的全部法条（用于评测第 0 层的一次性语料体检）"""
    out: List[LawRef] = []
    seen = set()
    for scm in LABOR_SCMS.values():
        for ref in scm.article_refs():
            if ref.key not in seen:
                seen.add(ref.key)
                out.append(ref)
    return out
