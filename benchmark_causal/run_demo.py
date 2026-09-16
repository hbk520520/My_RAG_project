"""
因果评测离线演示
================
不联网、不调 API，把整条评测链路跑一遍：

  第 0 层  法条体检      —— SCM 的每条法条锚点都必须存在且在案件发生日有效
  SCM     因果图打印    —— 每条边标出「规范性(法) / 事实性(事)」与法条依据
  出题    L1/L2/L3/扰动 —— 三层答案由同一套结构方程推导
  评分    三种假作答    —— 正确 / 天真 / 编造法条，展示硬门禁与打分的区分度

运行：
    python benchmark_causal/run_demo.py
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from benchmark_causal.gates import LexicalMatcher, aggregate_by_case, score_case
from benchmark_causal.generator import CausalCaseGenerator
from benchmark_causal.legal_corpus import LegalCorpus
from benchmark_causal.scenarios import ALL_SCENARIOS, OVERTIME_SCENARIO
from benchmark_causal.schemas import ModelResponse
from benchmark_causal.scm_labor import LABOR_SCMS, all_article_refs


def _rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    corpus = LegalCorpus.load()
    _rule("0. 法条语料")
    stats = corpus.stats()
    print(f"  条目 {stats['articles']} · 法律 {stats['laws']} 部 · "
          f"其中已废止 {stats['expired_articles']} 条")
    print(f"  来源 {stats['source']}")

    _rule("1. 第 0 层：审计 SCM 的法条锚点（存在性 + 时效性）")
    bad = 0
    for scenario in ALL_SCENARIOS:
        print(f"\n  [{scenario.source_case_id}] 案件发生日 {scenario.case_date}")
        for ref in all_article_refs():
            check = corpus.check(ref, scenario.case_date)
            mark = "OK" if check.ok else "!!"
            bad += int(not check.ok)
            print(f"    {mark}  {ref.key:<32} {check.reason}")
    print(f"\n  不合格锚点 = {bad}（应为 0：法条不是真实有效的，题就不合法）")

    _rule("2. 因果图（法=规范性边，有此锚点；事=案情内事实关联）")
    for name, scm in LABOR_SCMS.items():
        print()
        print(scm.describe_graph())
        print(f"  → treatment={scm.treatment}  outcome={scm.outcome}  "
              f"has_confounder={scm.has_confounder()}")
        if scm.has_confounder():
            conf = scm.confounders(scm.treatment, scm.outcome)
            print(f"  → 混杂因子 {sorted(conf)}："
                  f"do(混杂因子) 会切断它对 {scm.treatment} 的输入")

    _rule("3. 出题：L1 观测 / L2 干预 / L3 反事实 / L3′ 抗扰动")
    all_cases = []
    for scenario in ALL_SCENARIOS:
        scm = LABOR_SCMS[scenario.domain]
        gen = CausalCaseGenerator(scm, corpus)
        print()
        print(gen.describe(scenario))
        cases = gen.generate(scenario)
        all_cases.extend(cases)
        print()
        for c in cases:
            tag = "抗扰动" if c.perturbation else f"L{int(c.causal_level)}"
            print(f"    {tag:<6} {c.case_id:<26} 标准答案 = {c.ground_truth_amount:>10,.2f} 元"
                  f"   混杂因子项适用={c.has_confounder}")

    _rule("4. 评分：三种假作答的区分度")

    ot_scm = LABOR_SCMS[OVERTIME_SCENARIO.domain]
    ot_gen = CausalCaseGenerator(ot_scm, corpus)
    ot_cases = {int(c.causal_level): c
                for c in ot_gen.generate(OVERTIME_SCENARIO) if not c.perturbation}

    def good(case):
        return ModelResponse(
            conclusion=case.ground_truth_outcome,
            causal_chain=list(case.key_reasoning_points),
            citations=[f"{r.law_name}{r.article_no}" for r in case.applicable_laws],
            amount=case.ground_truth_amount,
        )

    l1, l2, l3 = ot_cases[1], ot_cases[2], ot_cases[3]
    # Pydantic 模型不可哈希，用 case_id 作键
    K1, K2, K3 = l1.case_id, l2.case_id, l3.case_id

    scenarios_of_answers = {
        "① 正确作答": {
            K1: good(l1), K2: good(l2), K3: good(l3),
        },
        "② 天真作答（以为不赶工→加班少→加班费少）": {
            K1: good(l1), K2: good(l2),
            K3: ModelResponse(
                conclusion="项目不赶工，按正常排班加班 10 小时，应支付 600 元",
                causal_chain=["项目赶工强度=normal", "工作日加班时长为 10 小时",
                              "加班费 = 10 × 40 × 150% = 600 元", "可主张总额 = 600 元"],
                citations=["劳动法第四十四条"], amount=600.0),
        },
        "③ 编造法条作答（内容对但引了不存在的条文）": {
            K1: ModelResponse(
                conclusion=l1.ground_truth_outcome,
                causal_chain=list(l1.key_reasoning_points),
                citations=["劳动法第九千九百九十九条"],
                amount=l1.ground_truth_amount),
            K2: good(l2), K3: good(l3),
        },
        "④ 引用已废止法条（2024 年案子引《合同法》）": {
            K1: ModelResponse(
                conclusion=l1.ground_truth_outcome,
                causal_chain=list(l1.key_reasoning_points),
                citations=["合同法第一百零七条"],
                amount=l1.ground_truth_amount),
            K2: good(l2), K3: good(l3),
        },
        "⑤ 无引证作答": {
            K1: ModelResponse(
                conclusion=l1.ground_truth_outcome,
                causal_chain=list(l1.key_reasoning_points),
                citations=[], amount=l1.ground_truth_amount),
            K2: good(l2), K3: good(l3),
        },
    }

    matcher = LexicalMatcher()
    for label, answers in scenarios_of_answers.items():
        print(f"\n  {label}")
        scores = []
        for case in (l1, l2, l3):
            resp = answers[case.case_id]
            s = score_case(case, resp, corpus, matcher=matcher)
            scores.append(s)
            print(f"     L{int(case.causal_level)}  得分 {s.final_score:<6.4f} "
                  f"硬门禁={'过' if s.hard_gate.passed else '不过'}"
                  f"{'' if s.hard_gate.passed else '  ← ' + '；'.join(s.hard_gate.errors)[:70]}")
        report = aggregate_by_case(scores)
        print(f"     → 因果一致性率 = {report['consistency_rate']} "
              f"（L1/L2/L3 全对才计 1；逐层平均分 {report['levels']}）")

    _rule("5. 结论")
    print("  · 评分的第一原则：能用 Python 判的绝不交给 LLM")
    print("      — 法条存在性、时效性、格式、金额精确匹配 都是确定性硬门禁")
    print("      — 只有「因果链命中率」「混杂因子是否隔离」才需要裁判")
    print("  · 三层答案由同一套结构方程推导 → 天然自洽，不会把模型冤判成错")
    print("  · L3 的答案是 3600 而不是 600：必须理解 do(Z) 已切断 Z→X 的路径")
    print(f"  共生成 {len(all_cases)} 道题（含抗扰动题）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
