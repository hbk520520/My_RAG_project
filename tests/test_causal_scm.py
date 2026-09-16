"""
因果评测 · SCM 执行器测试
========================
重点验证 `do(·)` 是**真的图手术**，而不是"改改题干"：

  · 正常推演：所有内生变量按结构方程求值
  · `do(X=x)`：X 固定为 x，其结构方程**被跳过**，下游按剩余方程重算
  · 反事实：翻转已发生变量后重跑；确定性 SCM 下溯因平凡
  · 混杂因子：能被自动识别（决定 L3「隔离」一项是否适用）

另外验证定义校验会在**构建期**就拦住不合法模型（缺法条锚点、成环、outcome 无方程）。
"""
import pytest

from benchmark_causal.scm import (
    KIND_FACTUAL,
    KIND_LEGAL,
    ROLE_CONFOUNDER,
    ROLE_EXOGENOUS,
    ROLE_MEDIATOR,
    ROLE_OUTCOME,
    ROLE_TREATMENT,
    SCMValidationError,
    Edge,
    Equation,
    StructuralCausalModel,
    Variable,
)
from benchmark_causal.scm_labor import (
    LCL_47,
    LCL_87,
    LABOR_LAW_44,
    LABOR_SCMS,
    all_article_refs,
    build_overtime_scm,
    build_unlawful_dismissal_scm,
)


@pytest.fixture
def overtime() -> StructuralCausalModel:
    return build_overtime_scm()


# ===========================================================================
# 定义校验
# ===========================================================================
def _minimal(variables, edges, equations, outcome, treatment="", renderer=None):
    return StructuralCausalModel("t", "测试模型", variables, edges, equations,
                                 outcome, treatment=treatment, outcome_renderer=renderer)


def test_legal_edge_without_article_anchor_is_rejected():
    """规范性因果边必须能溯源到法条 —— 否则拒绝构建（可审计性要求）"""
    variables = [Variable("x", "因"), Variable("y", "果", ROLE_OUTCOME)]
    edges = [Edge("x", "y", "因为没有锚点", article_refs=(), kind=KIND_LEGAL)]
    equations = [Equation("y", lambda v: v["x"], "y = x", article_refs=(LABOR_LAW_44,))]
    with pytest.raises(SCMValidationError, match="缺少法条锚点"):
        _minimal(variables, edges, equations, "y")


def test_factual_edge_only_needs_rationale():
    """事实性因果边（非法律规则）不要求法条锚点，但必须有依据说明"""
    variables = [Variable("x", "因"), Variable("y", "果", ROLE_OUTCOME)]
    ok = _minimal(
        variables,
        [Edge("x", "y", "案情内的事实关联", kind=KIND_FACTUAL)],
        [Equation("y", lambda v: v["x"], "y = x", article_refs=(LABOR_LAW_44,))],
        "y")
    assert ok.outcome == "y"

    with pytest.raises(SCMValidationError, match="rationale"):
        _minimal(variables,
                 [Edge("x", "y", "   ", kind=KIND_FACTUAL)],
                 [Equation("y", lambda v: v["x"], "y = x", article_refs=(LABOR_LAW_44,))],
                 "y")


def test_cycle_is_rejected():
    # 注意：两个变量都必须是**非** exogenous 角色，否则会先撞上
    # "标为外生却又有方程"的语义矛盾检查，测不到环路检查。
    variables = [Variable("a", "A", ROLE_MEDIATOR), Variable("b", "B", ROLE_OUTCOME)]
    edges = [Edge("a", "b", "r", article_refs=(LABOR_LAW_44,)),
             Edge("b", "a", "r", article_refs=(LABOR_LAW_44,))]
    equations = [Equation("a", lambda v: v["b"], "a=b", article_refs=(LABOR_LAW_44,)),
                 Equation("b", lambda v: v["a"], "b=a", article_refs=(LABOR_LAW_44,))]
    with pytest.raises(SCMValidationError, match="环路"):
        _minimal(variables, edges, equations, "b")


def test_outcome_must_have_equation():
    variables = [Variable("x", "因"), Variable("y", "果", ROLE_OUTCOME)]
    with pytest.raises(SCMValidationError, match="必须有结构方程"):
        _minimal(variables, [], [], "y")


def test_exogenous_role_cannot_have_equation():
    """标为外生却又有方程 = 语义矛盾，必须在构建期拦住"""
    variables = [Variable("x", "因", ROLE_EXOGENOUS), Variable("y", "果", ROLE_OUTCOME)]
    with pytest.raises(SCMValidationError, match="语义矛盾"):
        _minimal(variables, [],
                 [Equation("x", lambda v: 1, "x=1", article_refs=(LABOR_LAW_44,)),
                  Equation("y", lambda v: v["x"], "y=x", article_refs=(LABOR_LAW_44,))],
                 "y")


def test_equation_without_anchor_is_rejected_for_legal_kind():
    variables = [Variable("x", "因"), Variable("y", "果", ROLE_OUTCOME)]
    with pytest.raises(SCMValidationError, match="缺少法条锚点"):
        _minimal(variables,
                 [Edge("x", "y", "r", article_refs=(LABOR_LAW_44,))],
                 [Equation("y", lambda v: v["x"], "y=x")],   # 未给锚点且默认 legal
                 "y")


def test_unknown_variable_in_edge_is_rejected():
    variables = [Variable("x", "因")]
    with pytest.raises(SCMValidationError, match="未声明"):
        _minimal(variables,
                 [Edge("x", "zzz", "r", article_refs=(LABOR_LAW_44,))],
                 [], "x")


# ===========================================================================
# 观测推演
# ===========================================================================
def test_observation_computes_all_endogenous(overtime):
    values = overtime.compute({"project_crunch": "crunch", "hourly_wage": 40.0})
    assert values["overtime_hours"] == 60.0          # 赶工 → 60 小时
    assert values["overtime_pay"] == pytest.approx(3600.0)   # 60 × 40 × 1.5
    assert values["night_shift_allowance"] == pytest.approx(300.0)
    assert values["total_claim"] == pytest.approx(3900.0)


def test_missing_root_variable_is_rejected(overtime):
    with pytest.raises(SCMValidationError, match="缺少根变量"):
        overtime.compute({"project_crunch": "crunch"})    # 缺 hourly_wage


def test_out_of_domain_value_is_rejected(overtime):
    with pytest.raises(SCMValidationError, match="不在允许域"):
        overtime.compute({"project_crunch": "被外星人抓走了", "hourly_wage": 40.0})


# ===========================================================================
# do 算子 = 图手术
# ===========================================================================
def test_do_cuts_incoming_edges(overtime):
    """
    do(project_crunch=normal) 后，赶工强度是**外生固定**的，
    其下游（加班时长、夜班津贴）按剩余方程重算：
      · 加班时长方程仍看 project_crunch → 变成 10
      · 夜班津贴方程仍看 project_crunch → 变成 0
    """
    values = overtime.compute({"project_crunch": "crunch", "hourly_wage": 40.0},
                              do={"project_crunch": "normal"})
    assert values["overtime_hours"] == 10.0
    assert values["night_shift_allowance"] == pytest.approx(0.0)
    assert values["overtime_pay"] == pytest.approx(600.0)      # 10 × 40 × 1.5
    assert values["total_claim"] == pytest.approx(600.0)


def test_do_on_downstream_variable_does_not_touch_its_ancestors(overtime):
    """
    do(overtime_hours=60) 时，project_crunch 保持观测值 crunch：
    这是"图手术"与"改题干"的本质区别 —— 干预 X 不会倒过来改 Z。
    """
    values = overtime.compute({"project_crunch": "crunch", "hourly_wage": 40.0},
                              do={"overtime_hours": 60.0})
    assert values["project_crunch"] == "crunch"          # 未被干预
    assert values["night_shift_allowance"] == pytest.approx(300.0)   # Z 仍为 crunch
    assert values["overtime_pay"] == pytest.approx(3600.0)


def test_do_out_of_domain_is_rejected(overtime):
    with pytest.raises(SCMValidationError, match="不在允许域"):
        overtime.compute({"project_crunch": "crunch", "hourly_wage": 40.0},
                         do={"project_crunch": "impossible"})


# ===========================================================================
# 反事实：隔离混杂因子
# ===========================================================================
def test_counterfactual_isolates_confounder(overtime):
    """
    L3 的核心：`do(Z=normal, X=60)` —— 换掉赶工环境，但把加班时长钉死。
      · 加班费**不变**（3600）：因为 X 被固定，Z→X 的路径已被切断
      · 夜班津贴归零：Z→M2 的路径变了
      · 总额 = 3600（而不是 600 —— 600 是"不赶工所以加班少"的天真答法）
    """
    observed = overtime.compute({"project_crunch": "crunch", "hourly_wage": 40.0})
    cf = overtime.counterfactual(
        {"project_crunch": "crunch", "hourly_wage": 40.0},
        flip={"project_crunch": "normal", "overtime_hours": observed["overtime_hours"]},
    )
    assert cf["overtime_pay"] == pytest.approx(observed["overtime_pay"])
    assert cf["night_shift_allowance"] == pytest.approx(0.0)
    assert cf["total_claim"] == pytest.approx(3600.0)
    assert cf["total_claim"] != pytest.approx(600.0)


def test_counterfactual_diff_reports_what_changed(overtime):
    base = {"project_crunch": "crunch", "hourly_wage": 40.0}
    cf = {"project_crunch": "normal", "overtime_hours": 60.0}
    before = overtime.compute(base)
    after = overtime.compute(base, do=cf)
    changed = overtime.diff(before, after)

    assert "overtime_pay" not in changed, "被钉住的 X 决定了加班费不该变"
    assert "night_shift_allowance" in changed
    assert "total_claim" in changed


# ===========================================================================
# 混杂因子识别
# ===========================================================================
def test_confounder_is_detected(overtime):
    assert overtime.treatment == "overtime_hours"
    assert overtime.confounders("overtime_hours", "total_claim") == {"project_crunch"}
    assert overtime.has_confounder() is True


def test_no_confounder_when_treatment_is_root():
    """处理变量是根节点 → 没有共同祖先 → has_confounder 为 False（L3 隔离项不适用）"""
    scm = build_unlawful_dismissal_scm()
    assert scm.confounders("dismissal_ground", "total_amount") == set()
    assert scm.has_confounder() is False


# ===========================================================================
# 因果链与渲染
# ===========================================================================
def test_chain_is_topological_and_cites_articles(overtime):
    values = overtime.compute({"project_crunch": "crunch", "hourly_wage": 40.0})
    chain = overtime.chain(values)
    assert len(chain) == len(overtime.variables)

    joined = "\n".join(chain)
    assert "《劳动法》第四十四条" in joined
    labels = [v.label for v in overtime.variables.values()]
    for label in labels:
        assert any(label in item for item in chain), f"链中缺少变量 {label}"


def test_render_outcome_contains_amount(overtime):
    values = overtime.compute({"project_crunch": "crunch", "hourly_wage": 40.0})
    assert "3900" in overtime.render_outcome(values)


# ===========================================================================
# 领域 SCM 整体一致性
# ===========================================================================
def test_all_labor_scms_are_valid():
    for name, scm in LABOR_SCMS.items():
        assert scm.name == name
        assert scm.article_refs(), f"{name} 没有任何法条锚点"
        assert len(scm.topological_order()) == len(scm.variables)
        assert scm.outcome in scm.equations


def test_severance_months_follows_article_47():
    """《劳动合同法》第四十七条：满一年 1 个月；剩余满 6 个月计 1 年；不满 6 个月计 0.5"""
    scm = build_unlawful_dismissal_scm()
    base = {"monthly_wage": 8000.0, "months_without_contract": 0.0}

    def months(ground, years):
        return scm.compute({**base, "dismissal_ground": ground,
                            "years_of_service": years})["compensation_months"]

    assert months("lawful", 0.4) == pytest.approx(0.5)
    assert months("lawful", 0.5) == pytest.approx(1.0)
    assert months("lawful", 3.0) == pytest.approx(3.0)
    assert months("lawful", 3.5) == pytest.approx(4.0)
    assert months("lawful", 3.4) == pytest.approx(3.5)
    # 违法解除 = 2N（第八十七条）
    assert months("unlawful", 3.0) == pytest.approx(6.0)
    # 过失性解除无补偿（第三十九条）
    assert months("employee_fault", 3.0) == pytest.approx(0.0)


def test_double_wage_is_capped_at_11_months():
    """《劳动合同法》第八十二条：超过一个月不满一年未签 → 最多 11 个月差额"""
    scm = build_unlawful_dismissal_scm()
    base = {"dismissal_ground": "lawful", "years_of_service": 1.0, "monthly_wage": 8000.0}
    for months_in, expected in ((0.0, 0.0), (1.0, 0.0), (2.0, 8000.0),
                                (12.0, 88000.0), (24.0, 88000.0)):
        v = scm.compute({**base, "months_without_contract": months_in})
        assert v["double_wage_amount"] == pytest.approx(expected), f"{months_in} 个月"
