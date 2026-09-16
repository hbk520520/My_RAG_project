"""
因果评测题生成器 —— 从 SCM 派生 L1 / L2 / L3 三层题
=================================================
三层题**全部**由同一个 SCM 派生，所以标准答案天然自洽：

  L1 观测（Seeing）        给定完整事实 → 问结论
  L2 干预（Doing）         对变量施加 $do(\\cdot)$ → 问结论
  L3 反事实（Counterfactual） 翻转已发生变量 → 问结论
  L3′ 抗扰动               与 L3 等价，但题干里塞入**无关细节**（如"当事人穿了红衣服"）
                           标准答案与 L3 完全相同 —— 用来测模型的因果推断是否被无关信息带偏

**L3 的关键设计：混杂因子隔离题**
  以加班费 SCM 为例，混杂因子 `project_crunch` 同时影响「加班时长」与「夜班津贴」。
  反事实设问是：*"若项目并非赶工，但加班时长仍为 60 小时，总额是多少？"*
  对应的干预是 `do(project_crunch=normal, overtime_hours=60)` —— **双变量干预**：
    · 改掉混杂因子（切断 Z→X 与 Z→M2）
    · 同时把 X 钉在观测值上（相当于"在保持加班不变的前提下，换掉赶工环境"）
  正确解只有「夜班津贴」变化，加班费**不变**。
  一个不懂 do 语义的模型会答"不赶工了→加班少了→加班费少了"，于是答错。
  这正是"隔离混杂因子"要测的东西。

Prompt 模板放在本模块内（`*_PROMPT`）而不是根目录 `prompts.py`：
  `prompts.py` 是**Agent 运行时**提示词的单一真源（planner/grader/reasoner…）；
  评测题的题面是**基准工件**，与 Agent 提示词生命周期不同，分开放更内聚。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .legal_corpus import LegalCorpus
from .schemas import CausalLevel, LegalCausalTestCase, LawRef, ModelResponse
from .scm import StructuralCausalModel

# ===========================================================================
# 题面模板
# ===========================================================================
L1_PROMPT = """【案情事实】
{story}

【问题】
请基于上述事实，判断{question}。要求：
1. 逐步写出你的因果推演链（每一步说明"因为…所以…"）；
2. 明确指出你依据的法条（写成「法律名称第X条」的形式）；
3. 最后给出结论，并给出确定的金额（单位为元）。"""

L2_PROMPT = """【案情事实】
{story}

【假设情形】
现在假设：{intervention}（其余事实均不变）。

【问题】
在该假设情形下，判断{question}。要求：
1. 逐步写出你的因果推演链；
2. 明确指出你依据的法条（写成「法律名称第X条」的形式）；
3. 最后给出结论与确定的金额（单位为元）。"""

L3_PROMPT = """【案情事实】
{story}

【反事实假设】
现假设：{counterfactual}
**注意**：{hold_clause}

【问题】
在该反事实假设下，判断{question}。要求：
1. 逐步写出你的因果推演链，并**明确说明哪些数额发生了变化、哪些没有变化**；
2. 明确指出你依据的法条（写成「法律名称第X条」的形式）；
3. 最后给出结论与确定的金额（单位为元）。"""

PERTURBATION_SUFFIX = """
【补充细节】（与法律分析无关）
{detail}"""


# ===========================================================================
# 场景定义
# ===========================================================================
@dataclass
class CausalScenario:
    """
    一个 SCM 案例的完整设定。所有干预/反事实**显式声明**，
    不做自动魔改 —— 保证每个测试用例都是可审计、可人工复核的。
    """

    source_case_id: str
    domain: str                      # SCM 名称
    case_date: str                   # 案件事实发生日（决定适用法版本）
    story: str                       # 中文事实际述（题干主体）
    assignments: Dict[str, Any]      # 外生变量取值
    question: str = "劳动者依法可以主张的金额"   # 题干里的问句宾语

    # L2：单变量或多变量干预
    intervention: Dict[str, Any] = field(default_factory=dict)
    intervention_text: str = ""

    # L3：反事实干预（通常 = 改混杂因子 + 钉住处理变量）
    counterfactual: Dict[str, Any] = field(default_factory=dict)
    counterfactual_text: str = ""
    hold_clause: str = "其余已认定的事实保持不变"

    # L3′：无关细节注入
    perturbation_detail: Optional[str] = None


# ===========================================================================
# 生成器
# ===========================================================================
class CausalCaseGenerator:
    """把 `CausalScenario` 展开成 L1 / L2 / L3 / L3′ 四道题"""

    def __init__(self, scm: StructuralCausalModel, corpus: Optional[LegalCorpus] = None):
        self.scm = scm
        self.corpus = corpus

    # ------------------------------------------------------------------ 工具
    def _outcome_var_label(self) -> str:
        return self.scm.variables[self.scm.outcome].label

    def _law_refs_text(self) -> str:
        return "、".join(f"《{r.law_name}》{r.article_no}" for r in self.scm.article_refs())

    def _build_case(self, scenario: CausalScenario, level: CausalLevel,
                    do: Mapping[str, Any], prompt: str,
                    perturbation: Optional[str] = None) -> LegalCausalTestCase:
        values = self.scm.compute(scenario.assignments, do=do)
        outcome = self.scm.render_outcome(values)
        amount = values.get(self.scm.outcome)
        return LegalCausalTestCase(
            case_id=f"{scenario.source_case_id}-L{int(level)}"
                    + ("-perturb" if perturbation else ""),
            source_case_id=scenario.source_case_id,
            causal_level=level,
            domain=self.scm.name,
            case_date=scenario.case_date,
            prompt=prompt,
            ground_truth_outcome=outcome,
            ground_truth_amount=float(amount) if isinstance(amount, (int, float)) else None,
            # 关键要素留空：本批用例的结论判别以**金额精确匹配**为主（见 gates.py），
            # 若某题无金额，需在此显式填 terms，否则结论判定退化为"非空即过"。
            ground_truth_terms=[],
            key_reasoning_points=self.scm.chain(values),
            applicable_laws=list(self.scm.article_refs()),
            scm_assignments=dict(scenario.assignments),
            intervention=dict(do) if level != CausalLevel.SEEING else {},
            counterfactual_flip=dict(do) if level == CausalLevel.COUNTERFACTUAL else {},
            # 由 SCM 判定该题是否存在混杂因子 —— 决定「隔离混杂因子」一项是否适用
            has_confounder=(level == CausalLevel.COUNTERFACTUAL and self.scm.has_confounder()),
            perturbation=perturbation,
        )

    # ------------------------------------------------------------------ 主入口
    def generate(self, scenario: CausalScenario,
                 include_perturbation: bool = True) -> List[LegalCausalTestCase]:
        if scenario.domain != self.scm.name:
            raise ValueError(f"场景 domain={scenario.domain} 与 SCM {self.scm.name} 不匹配")

        # 先算一遍观测值，供 L3 的"钉住处理变量"使用
        observed = self.scm.compute(scenario.assignments)

        cases: List[LegalCausalTestCase] = []

        # ---- L1 观测 ----
        cases.append(self._build_case(
            scenario, CausalLevel.SEEING, {},
            L1_PROMPT.format(story=scenario.story, question=scenario.question),
        ))

        # ---- L2 干预 ----
        if scenario.intervention:
            cases.append(self._build_case(
                scenario, CausalLevel.DOING, scenario.intervention,
                L2_PROMPT.format(story=scenario.story,
                                 intervention=scenario.intervention_text
                                 or self._describe_do(scenario.intervention),
                                 question=scenario.question),
            ))

        # ---- L3 反事实 ----
        if scenario.counterfactual:
            cf = dict(scenario.counterfactual)
            # 若把处理变量钉在观测值上，题干里要显式说明"保持不变"
            hold_bits = []
            for name, val in cf.items():
                if name in observed and observed[name] == val:
                    var = self.scm.variables[name]
                    hold_bits.append(f"{var.label}仍为 {self.scm.fmt(val, var)}")
            hold_clause = "；".join(hold_bits) if hold_bits else scenario.hold_clause

            cf_prompt = L3_PROMPT.format(
                story=scenario.story,
                counterfactual=scenario.counterfactual_text
                or self._describe_do(cf),
                hold_clause=hold_clause,
                question=scenario.question,
            )
            cases.append(self._build_case(
                scenario, CausalLevel.COUNTERFACTUAL, cf, cf_prompt))

            # ---- L3′ 抗扰动（题干加无关细节，标准答案与 L3 完全一致）----
            if include_perturbation and scenario.perturbation_detail:
                cases.append(self._build_case(
                    scenario, CausalLevel.COUNTERFACTUAL, cf,
                    cf_prompt + PERTURBATION_SUFFIX.format(
                        detail=scenario.perturbation_detail),
                    perturbation=scenario.perturbation_detail,
                ))

        return cases

    # ------------------------------------------------------------------ 描述
    def _describe_do(self, do: Mapping[str, Any]) -> str:
        parts = []
        for name, val in do.items():
            var = self.scm.variables[name]
            parts.append(f"{var.label}为「{val}」")
        return "，且".join(parts)

    def describe(self, scenario: CausalScenario) -> str:
        """打印该场景的三层标准答案，便于人工核对（出题自检用）"""
        lines = [f"=== {scenario.source_case_id} [{self.scm.name}] ===",
                 f"  案情: {scenario.story}",
                 f"  外生取值: {scenario.assignments}"]
        for label, do in (("L1 观测", {}),
                          ("L2 干预", scenario.intervention),
                          ("L3 反事实", scenario.counterfactual)):
            if label.startswith("L2") and not do:
                continue
            if label.startswith("L3") and not do:
                continue
            v = self.scm.compute(scenario.assignments, do=do)
            lines.append(f"  {label}: do={do or '{}'}")
            lines.append(f"      → {self.scm.render_outcome(v)}")
            lines.append(f"      → 推理链 {len(self.scm.chain(v))} 步")
        return "\n".join(lines)
