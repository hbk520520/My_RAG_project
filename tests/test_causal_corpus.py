"""
因果评测 · 法条库测试
====================
重点验证两件**确定性**的事（它们是硬门禁的基础）：
  1. 引用解析要能容忍各种写法（「第87条」「第八十七条」「《X法》第…条」）
  2. **法不溯及既往**：同一条引用在不同案件发生日结论不同
     （2019 年的案子引《合同法》有效；2024 年的案子引《合同法》已废止）
"""
import pytest

from benchmark_causal.legal_corpus import (
    OPEN_END,
    LegalCorpus,
    cn_to_int,
    normalize_law_name,
    parse_citation,
)
from benchmark_causal.schemas import LawRef


@pytest.fixture(scope="module")
def corpus() -> LegalCorpus:
    return LegalCorpus.load()


# ===========================================================================
# 中文数字与引用解析
# ===========================================================================
@pytest.mark.parametrize("text,expected", [
    ("八十七", 87),
    ("一百零三", 103),
    ("一千二百", 1200),
    ("十", 10),
    ("一", 1),
    ("87", 87),
    ("一十九", 19),
])
def test_cn_to_int(text, expected):
    assert cn_to_int(text) == expected


def test_cn_to_int_rejects_non_numeric():
    assert cn_to_int("第") is None
    assert cn_to_int("abc") is None
    assert cn_to_int("") is None


@pytest.mark.parametrize("raw", [
    "劳动合同法第八十七条",
    "《劳动合同法》第八十七条",
    "中华人民共和国劳动合同法第87条",
    "劳动合同法第87条",
    " 劳动合同法 第八十七条 ",
])
def test_parse_citation_tolerates_common_forms(raw):
    assert parse_citation(raw) == ("劳动合同法", 87)


@pytest.mark.parametrize("raw", ["", "随便写点什么", "劳动合同法", "第八十七条", None])
def test_parse_citation_rejects_garbage(raw):
    """解析不出来必须返回 None，交由调用方按「格式不合规」硬门禁处理"""
    assert parse_citation(raw) is None


def test_normalize_law_name_strips_prefix_and_spaces():
    assert normalize_law_name("《中华人民共和国劳动合同法》") == "劳动合同法"
    assert normalize_law_name("劳动 合同 法") == "劳动合同法"


# ===========================================================================
# 语料加载
# ===========================================================================
def test_corpus_loads_with_expected_laws(corpus):
    stats = corpus.stats()
    assert stats["articles"] > 600, "语料条目过少，可能构建不全"
    assert stats["laws"] >= 6
    assert stats["expired_articles"] > 0, "应包含已废止法条（用于失效引用测试）"


def test_corpus_contains_key_labor_articles(corpus):
    for ref in (LawRef(law_name="劳动合同法", article_no="第八十七条"),
                LawRef(law_name="劳动合同法", article_no="第四十七条"),
                LawRef(law_name="劳动法", article_no="第四十四条")):
        assert corpus.exists(ref.law_name, ref.article_no), f"缺少 {ref.key}"


def test_article_text_is_non_empty(corpus):
    art = corpus.get("劳动合同法", "第八十七条")
    assert art is not None
    assert "二倍" in art.text and "赔偿金" in art.text
    assert art.valid_to >= OPEN_END, "劳动合同法应为现行有效"


# ===========================================================================
# 存在性与时效性
# ===========================================================================
def test_nonexistent_article_is_reported(corpus):
    check = corpus.check(
        LawRef(law_name="劳动合同法", article_no="第九千九百九十九条"), "2024-06-15")
    assert check.exists is False
    assert check.effective is False
    assert check.ok is False
    assert "不存在" in check.reason


def test_effective_article_passes(corpus):
    check = corpus.check(
        LawRef(law_name="劳动合同法", article_no="第八十七条"), "2024-06-15")
    assert check.ok is True
    assert check.matched


def test_not_yet_effective_before_valid_from(corpus):
    """《劳动合同法》2013-07-01 起施行 —— 2010 年的案子不能引用它"""
    check = corpus.check(
        LawRef(law_name="劳动合同法", article_no="第八十七条"), "2010-01-01")
    assert check.exists is True
    assert check.effective is False
    assert "无效" in check.reason


def test_law_not_applied_retroactively(corpus):
    """
    **法不溯及既往**（本题是本模块最重要的确定性判定）：
    《合同法》valid_to = 2019-12-31
      · 2019-06-01 的案子引用它 → 有效
      · 2024-06-15 的案子引用它 → 已废止
    """
    ref = LawRef(law_name="合同法", article_no="第一百零七条")

    before = corpus.check(ref, "2019-06-01")
    after = corpus.check(ref, "2024-06-15")

    assert before.exists is True and after.exists is True
    assert before.effective is True, "2019 年的案子，《合同法》当时仍有效"
    assert after.effective is False, "2024 年的案子，《合同法》已废止"
    assert after.ok is False
    assert "已废止" in after.reason or "无效" in after.reason


def test_check_many_returns_one_result_per_ref(corpus):
    refs = [LawRef(law_name="劳动合同法", article_no="第八十七条"),
            LawRef(law_name="劳动关系法", article_no="第一条")]
    out = corpus.check_many(refs, "2024-06-15")
    assert len(out) == 2
    assert out[0].ok is True
    assert out[1].exists is False


def test_cross_check_source_can_flag_dispute(corpus):
    """第二来源结论不一致时必须标记 disputed，而不是悄悄采信单源"""
    strict = LegalCorpus(
        [corpus.get("劳动合同法", "第八十七条")],
        cross_check=lambda law, idx: False,
    )
    check = strict.check(LawRef(law_name="劳动合同法", article_no="第八十七条"),
                         "2024-06-15")
    assert check.exists is True
    assert check.disputed is True
    assert "第二来源" in check.reason


def test_versions_of_returns_history(corpus):
    versions = corpus.versions_of("劳动合同法", "第八十七条")
    assert len(versions) >= 1
    assert all(v.article_index == 87 for v in versions)
