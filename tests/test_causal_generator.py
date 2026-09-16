"""
因果评测 · 出题器测试
====================
验收三条核心不变量：

  1. **三层答案必须互不相同** —— 否则 L3 没有区分度（L1=L3 的话，
     "瞎猫碰上死耗子"的模型也会被判对）
  2. **抗扰动题的标准答案必须与 L3 完全一致** —— 才能用它测"因果推断是否被无关信息带偏"
  3. **第 0 层：所有 SCM 锚定法条必须真实存在且在当时有效** ——
     这是"先检查法律条文是否存在"的落地，也是防止出题本身出错
"""
import pytest

from benchmark_causal.generator import (
    L1_PROMPT,
    CausalCaseGenerator,
    CausalScenario,
)
from benchmark_causal.legal_corpus import LegalCorpus
from benchmark_causal.scenarios import (
    ALL_SCENARIOS,
    DISMISSAL_SCENARIO,
    OVERTIME_SCENARIO,
    scenario_by_id,
    scenarios_for,
)
from benchmark_causal.schemas import CausalLevel
from benchmark_causal.scm_labor import LABOR_SCMS, all_article_refs


@pytest.fixture(scope="module")
def corpus() -> LegalCorpus:
    return LegalCorpus.load()


def _generate(scenario, corpus):
    scm = LABOR_SCMS[scenario.domain]
    return CausalCaseGenerator(scm, corpus).generate(scenario)


def _by_level(cases):
    return {int(c.causal_level): c for c in cases if not c.perturbation}


# ===========================================================================
# 第 0 层：出题前的法条体检
# ===========================================================================
def test_every_scm_anchor_exists_and_is_effective(corpus):
    """
    SCM 的每条法条锚点都必须：**存在** + 在案件发生日**有效**。
    若这条断言挂了，说明题目本身错了 —— 模型"答错"其实是题错。
    """
    for scenario in ALL_SCENARIOS:
        bad = []
        for ref in all_article_refs():
            check = corpus.check(ref, scenario.case_date)
            if not check.ok:
                bad.append(f"{ref.key}: {check.reason}")
        assert not bad, f"[{scenario.source_case_id}] 法条锚点不合格：{bad}"


def test_scenario_story_does_not_cite_expired_law(corpus):
    """案情里引用的法条必须当时有效（这里是防止题面写错）"""
    for scenario in ALL_SCENARIOS:
        for ref in all_article_refs():
            assert corpus.check(ref, scenario.case_date).effective is True


# ===========================================================================
# 三层题生成
# ===========================================================================
def test_generates_four_cases_per_scenario(corpus):
    """L1 + L2 + L3 + L3′抗扰动 = 4 道"""
    for scenario in ALL_SCENARIOS:
        cases = _generate(scenario, corpus)
        assert len(cases) == 4
        levels = sorted(int(c.causal_level) for c in cases)
        assert levels == [1, 2, 3, 3]


def test_case_ids_are_unique_and_traceable(corpus):
    for scenario in ALL_SCENARIOS:
        cases = _generate(scenario, corpus)
        ids = [c.case_id for c in cases]
        assert len(set(ids)) == len(ids), "case_id 不能重复"
        assert all(c.source_case_id == scenario.source_case_id for c in cases)
        assert all(c.domain == scenario.domain for c in cases)


def test_all_levels_share_same_case_date_and_assignments(corpus):
    """三层题必须同源 —— 这是 Case 级"因果一致性"聚合的前提"""
    for scenario in ALL_SCENARIOS:
        cases = _generate(scenario, corpus)
        assert len({c.case_date for c in cases}) == 1
        assert len({tuple(sorted(c.scm_assignments.items())) for c in cases}) == 1


# ===========================================================================
# 核心不变量 1：三层答案互不相同
# ===========================================================================
def test_three_levels_have_distinct_answers(corpus):
    for scenario in ALL_SCENARIOS:
        lv = _by_level(_generate(scenario, corpus))
        amounts = [lv[i].ground_truth_amount for i in (1, 2, 3)]
        assert amounts[0] != amounts[1], \
            f"[{scenario.source_case_id}] L1 与 L2 答案相同，干预没起作用"
        assert amounts[0] != amounts[2], \
            f"[{scenario.source_case_id}] L1 与 L3 答案相同，反事实没有区分度"


def test_l3_amount_differs_from_naive_reanswer(corpus):
    """
    L3 的正确答案必须**不同于**"不赶工所以加班少"的天真答案（600）。
    这是这个用例唯一的价值所在 —— 如果两者相同，题目就废了。
    """
    cases = _by_level(_generate(OVERTIME_SCENARIO, corpus))
    assert cases[3].ground_truth_amount == pytest.approx(3600.0)
    assert cases[2].ground_truth_amount == pytest.approx(600.0)


# ===========================================================================
# 核心不变量 2：抗扰动题答案与 L3 一致
# ===========================================================================
def test_perturbation_case_matches_l3_ground_truth(corpus):
    """
    L3′ 只多了一句无关细节，标准答案必须与 L3 **完全相同** ——
    否则"抗扰动率"这个指标就没有基准了。
    """
    for scenario in ALL_SCENARIOS:
        cases = _generate(scenario, corpus)
        l3 = next(c for c in cases
                  if int(c.causal_level) == 3 and c.perturbation is None)
        pert = next(c for c in cases if c.perturbation)
        assert pert.ground_truth_amount == l3.ground_truth_amount
        assert pert.ground_truth_outcome == l3.ground_truth_outcome
        assert pert.key_reasoning_points == l3.key_reasoning_points


def test_perturbation_prompt_contains_irrelevant_detail(corpus):
    cases = _generate(OVERTIME_SCENARIO, corpus)
    pert = next(c for c in cases if c.perturbation)
    assert OVERTIME_SCENARIO.perturbation_detail in pert.prompt
    assert "红色卫衣" in pert.prompt


def test_perturbation_can_be_disabled(corpus):
    scm = LABOR_SCMS[OVERTIME_SCENARIO.domain]
    cases = CausalCaseGenerator(scm, corpus).generate(OVERTIME_SCENARIO,
                                                      include_perturbation=False)
    assert len(cases) == 3


# ===========================================================================
# 题面与元数据
# ===========================================================================
def test_prompts_contain_story_and_stay_structured(corpus):
    for scenario in ALL_SCENARIOS:
        for case in _generate(scenario, corpus):
            assert scenario.story in case.prompt
            assert scenario.question in case.prompt
            assert "法条" in case.prompt, "题面必须要求模型给出法条依据"
            assert "金额" in case.prompt


def test_each_case_carries_its_applicable_laws(corpus):
    for scenario in ALL_SCENARIOS:
        for case in _generate(scenario, corpus):
            assert case.applicable_laws, "每题都必须带可适用的法条清单"
            scm = LABOR_SCMS[scenario.domain]
            assert {r.key for r in case.applicable_laws} <= {r.key for r in scm.article_refs()}


def test_confounder_flag_matches_scm(corpus):
    """只有"存在混杂因子 + L3"的题才该带 has_confounder=True"""
    ot = _by_level(_generate(OVERTIME_SCENARIO, corpus))
    dis = _by_level(_generate(DISMISSAL_SCENARIO, corpus))
    assert [ot[i].has_confounder for i in (1, 2, 3)] == [False, False, True]
    assert [dis[i].has_confounder for i in (1, 2, 3)] == [False, False, False]


def test_intervention_and_flip_are_recorded(corpus):
    ot = _by_level(_generate(OVERTIME_SCENARIO, corpus))
    assert ot[1].intervention == {}
    assert ot[2].intervention == {"project_crunch": "normal"}
    assert ot[3].counterfactual_flip == {"project_crunch": "normal",
                                        "overtime_hours": 60.0}


def test_reasoning_chain_matches_scm_variable_count(corpus):
    for scenario in ALL_SCENARIOS:
        scm = LABOR_SCMS[scenario.domain]
        for case in _generate(scenario, corpus):
            assert len(case.key_reasoning_points) == len(scm.variables)


def test_domain_mismatch_is_rejected(corpus):
    scm = LABOR_SCMS[OVERTIME_SCENARIO.domain]
    wrong = CausalScenario(
        source_case_id="x", domain="labor_unlawful_dismissal",
        case_date="2024-06-15", story="x", assignments={})
    with pytest.raises(ValueError, match="不匹配"):
        CausalCaseGenerator(scm, corpus).generate(wrong)


def test_scenario_registry_helpers():
    assert scenario_by_id("labor-ot-001") is OVERTIME_SCENARIO
    assert scenarios_for("labor_unlawful_dismissal") == [DISMISSAL_SCENARIO]
    with pytest.raises(KeyError):
        scenario_by_id("不存在")
