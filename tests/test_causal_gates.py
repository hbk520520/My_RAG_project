"""
因果评测 · 硬门禁与打分测试
==========================
本文件是"**能用 Python 判的绝不交给 LLM**"这条原则的验收：

  硬门禁（不过 → 直接 0 分，连裁判都不用请）
    · 格式：结论/因果链非空
    · 引证：法条**存在** + 在**案件发生日有效**（编造条文、引用废止条文都要被拦）
    · 结论：关键要素与关键数值必须命中
    · 金额：精确匹配（容差 1e-2）

  软评分（才用裁判/词法匹配）
    · 因果链命中率、混杂因子是否隔离 → 再按权重由 **Python** 合成总分
"""
import pytest

from benchmark_causal.gates import (
    DEFAULT_WEIGHTS,
    LexicalMatcher,
    ScoringWeights,
    aggregate_by_case,
    extract_numbers,
    normalize_text,
    run_hard_gates,
    score_case,
)
from benchmark_causal.generator import CausalCaseGenerator
from benchmark_causal.legal_corpus import LegalCorpus
from benchmark_causal.scenarios import DISMISSAL_SCENARIO, OVERTIME_SCENARIO
from benchmark_causal.schemas import CausalLevel, LegalCausalTestCase, ModelResponse
from benchmark_causal.scm import ROLE_OUTCOME
from benchmark_causal.scm_labor import LABOR_SCMS


@pytest.fixture(scope="module")
def corpus() -> LegalCorpus:
    return LegalCorpus.load()


@pytest.fixture(scope="module")
def overtime_cases(corpus) -> dict:
    scm = LABOR_SCMS[OVERTIME_SCENARIO.domain]
    cases = CausalCaseGenerator(scm, corpus).generate(OVERTIME_SCENARIO)
    return {int(c.causal_level): c for c in cases if not c.perturbation}


@pytest.fixture(scope="module")
def dismissal_cases(corpus) -> dict:
    scm = LABOR_SCMS[DISMISSAL_SCENARIO.domain]
    cases = CausalCaseGenerator(scm, corpus).generate(DISMISSAL_SCENARIO)
    return {int(c.causal_level): c for c in cases if not c.perturbation}


def _good_response(case: LegalCausalTestCase) -> ModelResponse:
    """构造一个"完全正确"的作答：引用适用法条 + 命中全部关键数值"""
    return ModelResponse(
        conclusion=case.ground_truth_outcome,
        causal_chain=list(case.key_reasoning_points),
        citations=[f"{r.law_name}{r.article_no}" for r in case.applicable_laws],
        amount=case.ground_truth_amount,
    )


# ===========================================================================
# 工具函数
# ===========================================================================
def test_extract_numbers():
    assert extract_numbers("加班费 3600 元，津贴 300 元") == [3600.0, 300.0]
    assert extract_numbers("没有数字") == []
    assert extract_numbers("负数 -5 与小数 1.25") == [-5.0, 1.25]


def test_normalize_text_unifies_punctuation():
    assert normalize_text("加班费，3600 元。") == "加班费,3600元."


def test_scoring_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="之和必须为 1.0"):
        ScoringWeights(reasoning=0.5, confounder=0.2)


# ===========================================================================
# 硬门禁 1：格式
# ===========================================================================
def test_format_violation_fails_gate(corpus, overtime_cases):
    case = overtime_cases[1]
    gate = run_hard_gates(case, None, corpus, parse_error="无法解析 JSON")
    assert gate.format_ok is False
    assert gate.passed is False
    assert "无法解析" in " ".join(gate.errors)


def test_empty_conclusion_fails_gate(corpus, overtime_cases):
    case = overtime_cases[1]
    resp = _good_response(case)
    resp.conclusion = "   "
    gate = run_hard_gates(case, resp, corpus)
    assert gate.format_ok is False
    assert any("conclusion" in e for e in gate.errors)


def test_empty_causal_chain_fails_gate(corpus, overtime_cases):
    case = overtime_cases[1]
    resp = _good_response(case)
    resp.causal_chain = []
    gate = run_hard_gates(case, resp, corpus)
    assert gate.format_ok is False
    assert any("causal_chain" in e for e in gate.errors)


def test_missing_citations_fails_gate(corpus, overtime_cases):
    """本项目要求结论必须可溯源到法条 —— 不引用直接不过"""
    case = overtime_cases[1]
    resp = _good_response(case)
    resp.citations = []
    gate = run_hard_gates(case, resp, corpus)
    assert gate.citations_exist is False
    assert gate.passed is False


# ===========================================================================
# 硬门禁 2：引证可信（存在性 + 时效性）
# ===========================================================================
def test_fabricated_article_is_caught(corpus, overtime_cases):
    """编造一条不存在的法条 —— 这是比 LLM 裁判更硬的信号"""
    case = overtime_cases[1]
    resp = _good_response(case)
    resp.citations = ["劳动法第九千九百九十九条"]
    gate = run_hard_gates(case, resp, corpus)
    assert gate.citations_exist is False
    assert gate.passed is False
    assert any("不存在" in e for e in gate.errors)


def test_expired_article_is_caught(corpus, overtime_cases):
    """
    引用**已废止**的《合同法》—— 而且案件发生在 2024 年。
    这正是"法不溯及既往"硬门禁要拦的：法条存在，但当时已失效。
    """
    case = overtime_cases[1]                       # case_date = 2024-06-15
    resp = _good_response(case)
    resp.citations = ["合同法第一百零七条"]
    gate = run_hard_gates(case, resp, corpus)
    assert gate.citations_exist is True, "该法条确实存在"
    assert gate.citations_effective is False, "但 2024 年已失效"
    assert gate.passed is False


def test_unparsable_citation_is_caught(corpus, overtime_cases):
    case = overtime_cases[1]
    resp = _good_response(case)
    resp.citations = ["我也不知道该引哪条"]
    gate = run_hard_gates(case, resp, corpus)
    assert gate.citations_exist is False
    assert any("无法解析" in e for e in gate.errors)


def test_valid_citation_passes(corpus, overtime_cases):
    case = overtime_cases[1]
    gate = run_hard_gates(case, _good_response(case), corpus)
    assert gate.citations_exist is True
    assert gate.citations_effective is True


# ===========================================================================
# 硬门禁 3：结论与金额
# ===========================================================================
def test_wrong_amount_fails_gate(corpus, overtime_cases):
    case = overtime_cases[1]
    resp = _good_response(case)
    resp.amount = case.ground_truth_amount + 100.0
    resp.conclusion = case.ground_truth_outcome.replace(
        f"{case.ground_truth_amount:g}", f"{case.ground_truth_amount + 100:g}")
    gate = run_hard_gates(case, resp, corpus)
    assert gate.amount_match is False
    assert gate.conclusion_match is False
    assert gate.passed is False


def test_amount_tolerance_is_one_cent(corpus, overtime_cases):
    case = overtime_cases[1]
    resp = _good_response(case)
    resp.amount = case.ground_truth_amount + 0.005      # 容差内
    assert run_hard_gates(case, resp, corpus).amount_match is True

    resp.amount = case.ground_truth_amount + 0.02       # 容差外
    assert run_hard_gates(case, resp, corpus).amount_match is False


def test_naive_wrong_answer_is_caught(corpus, overtime_cases):
    """
    关键区分度验证：L3 的"天真答法"必须被判错。
    天真答法 = 以为"不赶工→加班少→加班费少"，于是答 600（正确是 3600）。
    """
    case = overtime_cases[3]
    assert case.ground_truth_amount == pytest.approx(3600.0)

    naive = ModelResponse(
        conclusion="项目不赶工，按正常排班加班 10 小时，应支付 600 元",
        causal_chain=["项目赶工强度=normal", "工作日加班时长=10小时",
                      "加班费=600元"],
        citations=["劳动法第四十四条"],
        amount=600.0,
    )
    score = score_case(case, naive, corpus)
    assert score.final_score == 0.0
    assert score.hard_gate.amount_match is False


# ===========================================================================
# 软评分：因果链命中率（词法兜底，确定性）
# ===========================================================================
def test_lexical_matcher_full_hit(corpus, overtime_cases):
    case = overtime_cases[1]
    verdict = LexicalMatcher().match(case, _good_response(case))
    assert verdict.reasoning_hit_rate == pytest.approx(1.0)


def test_lexical_matcher_partial_hit(corpus, overtime_cases):
    """因果链只写到一半（缺夜班津贴与总额）→ 命中率必须低于 1"""
    case = overtime_cases[1]
    resp = _good_response(case)
    resp.causal_chain = resp.causal_chain[:2]
    verdict = LexicalMatcher().match(case, resp)
    assert 0.0 < verdict.reasoning_hit_rate < 1.0


def test_lexical_matcher_is_deterministic(corpus, overtime_cases):
    """同一输入必须给出同一结果（这是它能进回归门禁的前提）"""
    case = overtime_cases[1]
    resp = _good_response(case)
    a = LexicalMatcher().match(case, resp)
    b = LexicalMatcher().match(case, resp)
    assert a.reasoning_hit_rate == b.reasoning_hit_rate


# ===========================================================================
# 总分合成
# ===========================================================================
def test_score_is_zero_when_gate_fails(corpus, overtime_cases):
    case = overtime_cases[1]
    resp = _good_response(case)
    resp.citations = ["劳动合同法第九千条"]      # 编造
    score = score_case(case, resp, corpus)
    assert score.final_score == 0.0
    assert score.judge is None, "硬门禁不过就不该请裁判"
    assert "硬门禁未通过" in score.final_reason


def test_score_full_marks_for_correct_answer(corpus, overtime_cases):
    case = overtime_cases[1]
    score = score_case(case, _good_response(case), corpus)
    assert score.hard_gate.passed is True
    assert score.final_score == pytest.approx(1.0)


def test_confounder_item_is_not_applicable_without_confounder(corpus, dismissal_cases):
    """
    无混杂因子的 L3（违法解除场景）：`has_confounder=False`
    → 「隔离混杂因子」一项**不适用**，不能因为裁判没给就把分扣掉。
    """
    case = dismissal_cases[3]
    assert case.has_confounder is False
    score = score_case(case, _good_response(case), corpus)
    assert score.final_score == pytest.approx(1.0)
    assert "不涉及混杂因子" in score.final_reason or "仅按因果链" in score.final_reason


def test_confounder_item_applies_when_confounder_exists(corpus, overtime_cases):
    case = overtime_cases[3]
    assert case.has_confounder is True
    assert case.causal_level == CausalLevel.COUNTERFACTUAL


def test_custom_judge_can_override(corpus, overtime_cases):
    """注入 LLM 裁判时，其 verdict 应当被采信（权重合成仍在 Python 里做）"""
    from benchmark_causal.schemas import JudgeVerdict

    class FakeJudge:
        def match(self, case, response):
            return JudgeVerdict(reasoning_hit_rate=0.5, confounder_isolated=1,
                                reason="假裁判")

    case = overtime_cases[3]           # L3 且存在混杂因子
    score = score_case(case, _good_response(case), corpus, matcher=FakeJudge())
    expected = DEFAULT_WEIGHTS.reasoning * 0.5 + DEFAULT_WEIGHTS.confounder * 1.0
    assert score.final_score == pytest.approx(expected)
    assert score.judge.reason == "假裁判"


# ===========================================================================
# Case 级聚合：因果一致性惩罚
# ===========================================================================
def test_consistency_requires_all_three_levels(corpus, overtime_cases):
    """
    L1、L2 都对但 L3 错 → 该 Case **整体记 0**
    （"L1 对、L3 错"说明是瞎猫碰上死耗子）
    """
    responses = {1: _good_response(overtime_cases[1]),
                 2: _good_response(overtime_cases[2]),
                 3: ModelResponse(conclusion="不赶工所以加班少，一共 600 元",
                                  causal_chain=["加班费=600元"],
                                  citations=["劳动法第四十四条"], amount=600.0)}
    scores = [score_case(overtime_cases[lv], responses[lv], corpus)
              for lv in (1, 2, 3)]

    report = aggregate_by_case(scores)
    assert report["case_count"] == 1
    assert report["consistency_rate"] == 0.0
    assert report["cases"][OVERTIME_SCENARIO.source_case_id]["all_pass"] is False
    # 但逐层得分仍能看到 L1/L2 是过的 —— 便于诊断
    assert report["levels"][1] == pytest.approx(1.0)
    assert report["levels"][3] == pytest.approx(0.0)


def test_consistency_full_marks_when_all_levels_correct(corpus, overtime_cases):
    scores = [score_case(overtime_cases[lv], _good_response(overtime_cases[lv]), corpus)
              for lv in (1, 2, 3)]
    report = aggregate_by_case(scores)
    assert report["consistency_rate"] == pytest.approx(1.0)
    assert report["level_pass_rate"][3] == pytest.approx(1.0)
