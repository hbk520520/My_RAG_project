"""
内置测试场景
============
每个场景把「案情事实、外生变量取值、L2 干预、L3 反事实」**全部显式写出来**，
不做自动魔改 —— 这样每一道题都能被人工复核、被 git 审计。

两个场景刻意覆盖两种拓扑：

  `OVERTIME_SCENARIO`  —— **含混杂因子**
     赶工强度 Z 同时影响加班时长 X 与夜班津贴 M2。
     L3 设问：*"若非赶工，但加班时长仍为 60 小时，总额多少？"*
     对应 `do(Z=normal, X=60)`（**双变量干预**）。
     正确解：加班费**不变**（3600），夜班津贴归零 → 总额 3600。
     不懂 do 语义的模型会答"不赶工→加班少→加班费少 → 600"，从而被区分出来。

  `DISMISSAL_SCENARIO` —— **不含混杂因子**
     处理变量是根节点，没有共同祖先。
     用来验证评测器会把「隔离混杂因子」标为**不适用**，而不是硬考一个不存在的东西。
"""
from __future__ import annotations

from typing import Dict, List

from .generator import CausalScenario

# ===========================================================================
# 场景一：加班费与夜班津贴（含混杂因子）
# ===========================================================================
OVERTIME_SCENARIO = CausalScenario(
    source_case_id="labor-ot-001",
    domain="labor_overtime_claim",
    case_date="2024-06-15",          # 《劳动法》第四十四条 2018-12-29 起有效
    story=(
        "某互联网公司于 2024 年 3 月启动重点版本上线，全体研发进入「赶工」状态："
        "工作日普遍延长工作时间，且大量安排夜班。技术员张某在该期间累计工作日延长工作时间 "
        "60 小时，其折算小时工资为 40 元/小时。公司因赶工统一安排了夜班，"
        "但截至离职未支付任何加班费，也未支付夜班津贴。"
    ),
    assignments={"project_crunch": "crunch", "hourly_wage": 40.0},
    question="公司应向张某支付的加班费与夜班津贴合计金额",

    # L2：单变量干预 —— 切掉混尡因子，加班时长随之回到正常水平
    intervention={"project_crunch": "normal"},
    intervention_text="该公司从未进入赶工状态（项目赶工强度为「normal」），因此未安排延长工作时间，也未安排夜班",

    # L3：双变量干预 —— 改掉赶工强度，同时把加班时长钉在观测值上（隔离混杂因子）
    counterfactual={"project_crunch": "normal", "overtime_hours": 60.0},
    counterfactual_text="该公司从未进入赶工状态（项目赶工强度为「normal」）",
    hold_clause="张某的工作日延长工作时间仍为 60 小时，折算小时工资仍为 40 元/小时",

    perturbation_detail="另查明，张某在赶工期间几乎每天穿同一件红色卫衣上班。",
)

# ===========================================================================
# 场景二：违法解除赔偿金（不含混杂因子）
# ===========================================================================
DISMISSAL_SCENARIO = CausalScenario(
    source_case_id="labor-dis-001",
    domain="labor_unlawful_dismissal",
    case_date="2024-06-15",          # 《劳动合同法》2013-07-01 起有效
    story=(
        "李某于 2021 年 3 月 1 日入职某公司，离职前十二个月平均工资 8000 元，"
        "双方自入职即签订书面劳动合同（未出现未签书面合同的情形）。"
        "2024 年 6 月 15 日，公司以「部门取消」为由口头通知李某次日无需上班，"
        "既未提前三十日书面通知，也未与李某协商变更劳动合同，且未支付任何补偿。"
    ),
    assignments={
        "dismissal_ground": "unlawful",
        "years_of_service": 3.0,
        "monthly_wage": 8000.0,
        "months_without_contract": 0.0,
    },
    question="公司应向李某支付的赔偿金总额",

    # L2：干预解除事由
    intervention={"dismissal_ground": "lawful"},
    intervention_text="该次解除被认定为合法解除（解除事由为「lawful」）",

    # L3：反事实 —— 若当时是合法解除
    counterfactual={"dismissal_ground": "lawful"},
    counterfactual_text="该公司当时的解除行为被认定为合法解除（解除事由为「lawful」）",
    hold_clause="工作年限仍为 3 年，离职前十二个月平均工资仍为 8000 元",

    perturbation_detail="另查明，李某在离职当天穿了一双白色运动鞋。",
)


ALL_SCENARIOS: List[CausalScenario] = [OVERTIME_SCENARIO, DISMISSAL_SCENARIO]


def scenarios_for(domain: str) -> List[CausalScenario]:
    return [s for s in ALL_SCENARIOS if s.domain == domain]


def scenario_by_id(source_case_id: str) -> CausalScenario:
    for s in ALL_SCENARIOS:
        if s.source_case_id == source_case_id:
            return s
    raise KeyError(f"未找到场景 {source_case_id}；可用: {[s.source_case_id for s in ALL_SCENARIOS]}")
