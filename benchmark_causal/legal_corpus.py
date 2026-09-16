"""
法条库 —— 「法条是否存在 / 在案件发生日是否有效」的确定性判定
============================================================
这是因果评测**第 0 层**：出题前先验证题目引用的法条是真实且当时有效的，
不合格的题不进测试集；作答后再用同一套逻辑做**硬门禁**。

两个关键设计：

1. **按数值条号比对，不按字符串**
   模型可能写「劳动合同法第87条」「中华人民共和国劳动合同法第八十七条」
   「《劳动合同法》第八十七条」，字符串匹配必然失败。这里统一解析成
   `(法名归一化, 条号整数)` 再比对。

2. **时效区间判定（法不溯及既往）**
   每条法条带 `valid_from` / `valid_to`：
     · `valid_from <= 案件发生日 <= valid_to` → 当时有效
     · `valid_to` 为哨兵 2099-12-31           → 视为现行有效
     · 案件发生日晚于 valid_to                → **当时已废止**（如《合同法》2020-01-01 起废止）
   这解决「用民法典判 2019 年案子」这类错题 —— 模型"答错"其实是题错。

3. **交叉验证接口（可选）**
   构造时可注入 `cross_check` 回调作为第二个来源（如国家法律法规数据库）。
   默认单源；接入后 `exists()` 要求两源一致，否则标记 `disputed`。
   本阶段按"一切从简"只实现单源，接口先留好。
"""
from __future__ import annotations

import json
import os
import re
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from .schemas import LawRef

# 视为「现行有效」的哨兵日期
OPEN_END = "2099-12-31"

DEFAULT_CORPUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "data", "legal_corpus.jsonl")

_CN_DIGITS = "〇零一二三四五六七八九十百千两"
_CN_NUM = {"〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
           "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}

# 「劳动合同法第八十七条」/「《劳动合同法》第八十七条」/「劳动合同法第87条」
_CITATION_RE = re.compile(
    rf"^\s*[《\"]?(?P<law>[^《》\"第]{{2,40}}?)[》\"]?\s*第\s*(?P<no>[{_CN_DIGITS}\d]+)\s*条\s*$"
)
# 去掉「中华人民共和国」前缀与空白，便于归一化
_LAW_PREFIX_RE = re.compile(r"^(中华人民共和国)+")


def cn_to_int(text: str) -> Optional[int]:
    """
    中文数字 → int（支持法律条号写法：「八十七」「一百零三」「一千二百」）。

    只实现条号需要的范围，不做通用中文数字解析。
    """
    text = (text or "").strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    total, section, number = 0, 0, 0
    for ch in text:
        if ch in _CN_NUM:
            number = _CN_NUM[ch]
        elif ch == "十":
            section += (number or 1) * 10
            number = 0
        elif ch == "百":
            section += (number or 1) * 100
            number = 0
        elif ch == "千":
            section += (number or 1) * 1000
            number = 0
        else:
            return None
    return total + section + number


def normalize_law_name(name: str) -> str:
    """法名归一化：去书名号、去「中华人民共和国」前缀、去空白"""
    n = (name or "").strip().strip("《》\"'")
    n = _LAW_PREFIX_RE.sub("", n)
    return re.sub(r"\s+", "", n)


def parse_citation(text: str) -> Optional[Tuple[str, int]]:
    """
    解析一条法条引用 → `(归一化法名, 条号整数)`。

    解析不出来返回 None（调用方应按「格式不合规」处理，而不是静默放过）。
    """
    m = _CITATION_RE.match(text or "")
    if not m:
        return None
    idx = cn_to_int(m.group("no"))
    if idx is None:
        return None
    return normalize_law_name(m.group("law")), idx


class LawArticle(BaseModel):
    """语料中的一条法条"""

    model_config = ConfigDict(extra="ignore")

    law_id: str = ""
    law_name: str
    law_level: str = ""
    filename: str = ""
    publish: str = ""
    valid_from: str = ""
    valid_to: str = OPEN_END
    version: int = 1
    article_no: str
    article_index: int
    chapter: str = ""
    text: str
    source: str = ""

    @property
    def key(self) -> Tuple[str, int]:
        return normalize_law_name(self.law_name), self.article_index

    @property
    def citation(self) -> str:
        return f"{self.law_name}{self.article_no}"

    @property
    def is_open_ended(self) -> bool:
        return self.valid_to >= OPEN_END


class CitationCheck(BaseModel):
    """一次法条引用的校验结果"""

    model_config = ConfigDict(extra="forbid")

    citation: str
    exists: bool = False
    effective: bool = False
    matched: List[str] = Field(default_factory=list)
    reason: str = ""
    disputed: bool = Field(default=False, description="两个来源结论不一致")
    articles: List[LawArticle] = Field(default_factory=list, exclude=True)

    @property
    def ok(self) -> bool:
        return self.exists and self.effective and not self.disputed


class LegalCorpus:
    """
    法条库。线程安全（只读）。

    典型用法：
        corpus = LegalCorpus.load()
        corpus.check(LawRef(law_name="劳动合同法", article_no="第八十七条"), "2023-05-01")
    """

    def __init__(self, articles: Iterable[LawArticle],
                 cross_check: Optional[Callable[[str, int], Optional[bool]]] = None):
        self._articles: List[LawArticle] = list(articles)
        self._by_key: Dict[Tuple[str, int], List[LawArticle]] = {}
        for a in self._articles:
            self._by_key.setdefault(a.key, []).append(a)
        for versions in self._by_key.values():
            versions.sort(key=lambda a: (a.valid_from or ""))
        # 交叉验证源（可选）：返回 True/False=该源结论，None=该源无法判定
        self._cross_check = cross_check

    # ---------------------------------------------------------------- 加载
    @classmethod
    def load(cls, path: str = None,
             cross_check: Optional[Callable[[str, int], Optional[bool]]] = None) -> "LegalCorpus":
        path = path or DEFAULT_CORPUS_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"法条语料不存在: {path}\n"
                f"请先构建：set LAWS_REPO=<已克隆的法条库路径> 然后运行\n"
                f"    python benchmark_causal/build_corpus.py"
            )
        articles: List[LawArticle] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    articles.append(LawArticle.model_validate(json.loads(line)))
        return cls(articles, cross_check=cross_check)

    # ---------------------------------------------------------------- 查询
    def __len__(self) -> int:
        return len(self._articles)

    def versions_of(self, law_name: str, article_no: str) -> List[LawArticle]:
        """取某条法条的全部历史版本（按生效日升序）"""
        idx = cn_to_int(article_no.replace("第", "").replace("条", ""))
        if idx is None:
            return []
        return list(self._by_key.get((normalize_law_name(law_name), idx), []))

    def get(self, law_name: str, article_no: str,
            on_date: Optional[str] = None) -> Optional[LawArticle]:
        """
        取法条正文。给了 `on_date` 就取该日期**生效**的版本（法不溯及既往）。

        多版本同日期命中时取 `valid_from` 最晚的那个。
        """
        versions = self.versions_of(law_name, article_no)
        if not versions:
            return None
        if on_date is None:
            return versions[-1]
        live = [a for a in versions if self._valid_on(a, on_date)]
        return live[-1] if live else versions[-1]

    @staticmethod
    def _valid_on(article: LawArticle, date: str) -> bool:
        if article.valid_from and date < article.valid_from:
            return False
        return date <= article.valid_to

    def exists(self, law_name: str, article_no: str) -> bool:
        """该法条是否存在于语料（**不考虑时效**）"""
        return bool(self.versions_of(law_name, article_no))

    def check(self, ref: LawRef, case_date: str) -> CitationCheck:
        """
        校验一条引用：存在性 + 在 `case_date` 是否有效 + 可选的交叉验证。
        """
        versions = self.versions_of(ref.law_name, ref.article_no)
        citation = f"{ref.law_name}{ref.article_no}"

        if not versions:
            return CitationCheck(citation=citation, exists=False, effective=False,
                                 reason=f"语料中不存在该法条：{citation}")

        effective_versions = [a for a in versions if self._valid_on(a, case_date)]
        matched = [a.citation for a in effective_versions]
        result = CitationCheck(
            citation=citation,
            exists=True,
            effective=bool(effective_versions),
            matched=matched,
            articles=versions,
        )

        if not effective_versions:
            spans = "; ".join(f"{a.valid_from or '?'}~{a.valid_to}" for a in versions)
            result.reason = (f"{citation} 在案件发生日 {case_date} 无效"
                             f"（该法条有效期：{spans}）—— 「法不溯及既往」")
        else:
            result.reason = f"{citation} 在 {case_date} 有效"

        # 可选：第二来源交叉验证
        if self._cross_check is not None:
            try:
                second = self._cross_check(normalize_law_name(ref.law_name),
                                           cn_to_int(ref.article_no.replace("第", "").replace("条", "")) or 0)
            except Exception:
                second = None
            if second is not None and second != result.exists:
                result.disputed = True
                result.reason += "；⚠️ 第二来源结论不一致，需人工复核"
        return result

    def check_many(self, refs: Iterable[LawRef], case_date: str) -> List[CitationCheck]:
        return [self.check(r, case_date) for r in refs]

    # ---------------------------------------------------------------- 统计
    def stats(self) -> Dict[str, object]:
        expired = sum(1 for a in self._articles if not a.is_open_ended)
        return {
            "articles": len(self._articles),
            "laws": len({a.law_name for a in self._articles}),
            "expired_articles": expired,
            "source": sorted({a.source for a in self._articles if a.source}),
        }
