# 因果证据：当前实现与验证边界

## 当前结论

新增 `causal_evidence/`，在语义检索图之外，由 LLM 提取、程序组装因果—时间超图；没有 GNN 训练。
验证组件检查引用和端点，再独立调用 LLM 区分原文明确支持的因果、相关、否定和假设。
这验证的是**文本是否支持该关系**，并不证明现实中的因果关系成立。

参考方法：[CHAIN v2](https://arxiv.org/html/2609.36689v2)，第 3–4.1 节、附录 B.1。
论文作者为 Wenjin Liu、Chenxi Wang、Yue Lu、Zhe Cui、Haoran Luo；论文署名机构包括
Hithink research、Nanyang Technological University。这里依据论文署名核实，没有独立核实各作者履历。

## 数据流与接口

`EvidenceRecord → 512-token/64-overlap → Extraction → 原文检查 → Verification → G=(V,E_H,C)`。
`graph.py::build_graph(records, call, cutoff=..., calibrator=..., extractor_id=...)` 的 `call(messages)`
接收消息列表并返回 JSON 字符串，可注入现有 LLM 接缝；不自行建立网络客户端。
512/64 基于 `cl100k_base` 的全篇 token 索引。多字节 Unicode 边界处向内对齐，实际 overlap 可能小于 64。
原文字符偏移、chunk_id、record_id、命题、引句、验证理由均保留。
额外的 `evidence_quote`、`proposition_index` 是本项目的审计契约，不声称是论文原始 JSON 字段。

`score_graph(graph, targets, cutoff)` 先筛选有效证据，再反向 BFS；按原文公式计算重复支持项、
时间有效性、端点平均有效性及链权乘积。目标必须显式对应图中实体，不能把全文问题强行当作实体。
只返回以目标结束的无重复节点链；`prevents` 改变链的关系极性，不把链分数解释为事件发生概率。
搜索受 `d_max`、`max_chains`、`max_expansions` 限制，达到搜索预算明确标记 `search_truncated`。

时间按日粒度处理，已知时间 `>= cutoff` 排除。年月使用该月末日，无法唯一解析的日期视为缺失。
缺失时间必须提供经过外部核实的 `available_at < cutoff`，且 `availability_verified=true`；
LLM 不能自行授予此资格。已知的晚发布日期优先阻止信息进入提取器。
缺失时间的有效性为显式参数 0.25；缺少相似度时不增加重复支持加分。
无截止时间表示一般证据分析，**不具备预测防泄漏保证**。

规范化只做 Unicode NFKC、大小写和空白处理，不猜测同义实体。
同一 record_id 的重叠块不算多次支持，同一来源的重复命题也不重复参与端点平均。
不同 record_id 不等于独立来源；转载稿应在输入阶段使用共同 source/record 标识，否则仍可能高估重复支持。

## 支持度校准

`raw_support` 保留 LLM 原始 strength；`calibrated_support` 是基于人工文本支持标签的等距回归输出。
二者均不是统计因果效应。无校准模型时保留 `null` 并报告 `uncalibrated`，不生成假校准分。
原始支持度 `(0,1]`，校准输出允许 `[0,1]`，因而可以把不可信边降到 0。

标注 JSONL 每行：

```json
{"source_id":"original-document-or-source-family","raw_support":0.8,"supported":1}
```

标签评价原文是否支持**准确的方向及关系类型**；包括提取出的错误边。不要只标记被验证器接受的正例。
应使用与上线提取器一致、且与后续验证流程匹配的标注分布；被拒绝边供验证消融使用。
相同原文及其改写、转载、重叠块必须属于同一 source_id。source_id 用 SHA256 固定排序，
整组划分训练与留出集；默认 25% 来源留出，至少四个来源，训练集需两类标签和不同分数。

```powershell
.\.venv\Scripts\python.exe tools/calibrate_causal_support.py labels.jsonl --extractor-id model-and-prompt-v1 --trained-through 2025-01-01 --out outputs/support-calibration.json
```

产物保存曲线、标注哈希、来源划分、提取器标识及标注可用日期；输出留出集原始/校准后的 Brier、10-bin ECE。
不使用留出标签重新拟合。预测截止时间必须晚于标注可用日期，提取器版本必须匹配。
`trained-through` 是标注完成时间，不是文中事件时间；当前依赖调用者提供真实日期。

## 验证、限制与下一决策门

测试在 Python 3.13.9 环境运行；tiktoken 0.14.0、scikit-learn 1.9.1、Pydantic 2.12.3。
分词词表首次使用需预先缓存；测试用注入分词器保持离线，并另行实测真实词表的 Unicode 分块。
生产镜像原有 Python 3.10 尚需独立验证；本次不升级整个项目依赖。

当前全量 370 项测试通过。测试使用脚本化模型响应；覆盖工程行为，**不是提取准确率实验**。
尚无人工标注语料，没有拟合上线校准器，没有实测真实 LLM 的因果边精确率或下游预测收益。
独立验证调用也可能重复提取模型的错误，必须用人工标签评估。

下一决策门：固定来源划分和相同提取候选，对照原始提取、加验证、加校准、二者同时使用；
报告边级 precision/recall/F1、留出 Brier/ECE、每块调用数及耗时。若校准未改善留出 Brier，
或验证以明显召回损失换取有限精确率收益，保留失败结果并停止宣称收益，调整标注或验证策略。
最小流程一次提取；存在合法候选时再一次验证，不按每条边单独请求。

相较论文，本项目**在实现上更进一步**：把原文可追溯验证、非因果拒绝、独立标注校准、
来源隔离评估作为显式组件。是否在实验效果上优于论文，仍需上述对照与消融，当前不能下结论。
