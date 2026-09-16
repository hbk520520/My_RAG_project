"""
benchmark_causal —— 法律领域的**因果**评测基准
============================================
不用 ROUGE / BLEU，用「做没做对因果推断」来评。三层结构：

  L1 观测（Seeing）        →  L2 干预（Doing）      →  L3 反事实（Counterfactual）
        ↓ do(X=x) 重算            ↓ 翻转+钉住 重算
  三层标准答案**全部**由同一个确定性 SCM（`scm.py`）推导，
  因此天然自洽 —— 不会出现"L1 的答案和 L3 的答案互相矛盾"把模型冤判成错。

模块职责：
  `schemas.py`       数据契约（题目 / 作答 / 裁判 / 得分）
  `legal_corpus.py`  法条库：存在性 + 时效性（法不溯及既往）判定
  `scm.py`           确定性结构因果模型执行器（含 do 算子与反事实）
  `scm_labor.py`     劳动争议领域的两个 SCM（一个含混杂因子、一个不含）
  `scenarios.py`     内置测试场景（案情 + 干预 + 反事实，全部显式声明）
  `generator.py`     由 SCM 派生 L1/L2/L3/抗扰动 四道题
  `gates.py`         硬门禁 + 打分 + Case 级因果一致性聚合
  `build_corpus.py`  从开源法条库构建 `data/legal_corpus.jsonl`

评分原则（继承本项目纪律）：
  **能用 Python 确定性判断的，绝不交给 LLM。**
  法条真实性、时效性、格式、金额精确匹配都是确定性门禁；
  只有「因果链命中率」与「混杂因子是否隔离」才交给裁判。
"""
__all__ = [
    "schemas",
    "legal_corpus",
    "scm",
    "scm_labor",
    "scenarios",
    "generator",
    "gates",
]
