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
原文字符偏移、chunk_id、record_id、可选 source_id、命题、引句、验证理由均保留。
被拒绝的候选也保留原始字段和拒绝理由，支持后续错误分析。
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
转载稿可使用不同 record_id，但应设置共同 source_id；未提供 source_id 时按 record_id 计数，
不同 record_id 不等于独立来源，仍可能高估重复支持。

## 两条执行链的启用方式

`config.yaml` 中设置 `causal_evidence.enabled: true`。提取和验证都复用已有
`llm.judge_model`（不强制切换为论文的 GPT-4o-mini）；校准标识是 `<judge_model>:causal-v1`。
每个块一次提取，有通过原文检查的候选才额外一次独立验证请求。验证器仍可能和提取器犯相同错误。
检查新配置不会自动发起 LLM 请求；执行启用后的任务才会使用现有 API 配置。

同步入口保持单参数接口兼容，并支持：

```python
router.process("预算削减是否导致资金紧张？",
               prediction_cutoff="2025-10-01", causal_targets=["资金紧张"])
```

调用方仍需为检索图注入语料。节点 metadata 可提供 `record_id`、`source_id`、`timestamp`、
`available_at`、`availability_verified`、`similarity`；FAISS 内积和图游走衰减分不冒充存储的语义相似度。
`prediction_cutoff`、`causal_targets` 可以放入 Kafka 会话状态，也可以在配置中提供默认值。
启用的旧会话若已越过 Retriever 且缺少来源记录，需重新检索；不静默伪造元数据。
截止日过滤在提取前执行，Grader、Reasoner 和最终报告接收同一份可用原文及因果权重。
无目标时报告 `no_target_anchor`，全部过滤时报告 `no_admitted_evidence`，不伪造权重。
启用时绕过旧答案缓存，避免没有时间边界的答案泄漏未来信息；不全局删除缓存。

本次只采用 CHAIN 的图构建和数据权重，不包含 Noisy-OR、概率融合及其他预测模块。

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

阶段一全量 370 项测试通过；最终集成检查结果在下方维护。测试使用脚本化模型响应；
覆盖工程行为，**不是提取准确率实验**。
尚无人工标注语料，没有拟合上线校准器，没有实测真实 LLM 的因果边精确率或下游预测收益。
独立验证调用也可能重复提取模型的错误，必须用人工标签评估。

下一决策门：固定来源划分和相同提取候选，对照原始提取、加验证、加校准、二者同时使用；
报告边级 precision/recall/F1、留出 Brier/ECE、每块调用数及耗时。若校准未改善留出 Brier，
或验证以明显召回损失换取有限精确率收益，保留失败结果并停止宣称收益，调整标注或验证策略。
最小流程一次提取；存在合法候选时再一次验证，不按每条边单独请求。

相较论文，本项目**在实现上更进一步**：把原文可追溯验证、非因果拒绝、独立标注校准、
来源隔离评估作为显式组件。是否在实验效果上优于论文，仍需上述对照与消融，当前不能下结论。

## 最终工程验证（2026-10-18）

- 全量 `pytest`：388 项通过（原有 325 项 + 新增 63 项）；既有 SQLite 测试有未关闭连接的 ResourceWarning。
- `python tools/ci.py`：五类检查全部通过，包含新模块入口导入、compileall、pytest、6 个离线 demo、跨进程恢复实验。
- 分词实测：cl100k_base 处理含中文和 emoji 的长文本，块最大 512 tokens，未引入替换字符。
- `git diff --check`：通过。
- 未执行真实 LLM/API 准确率评测、Kafka/Redis 服务集成、Docker 镜像构建或 Python 3.10 运行验证；本机没有 Docker CLI。

Windows 复现命令（PowerShell，在已安装依赖的项目根目录）：

```powershell
New-Item -ItemType Directory -Path outputs/temp -Force | Out-Null
$env:TEMP = Join-Path (Get-Location) 'outputs/temp'
$env:TMP = $env:TEMP
$env:OMP_NUM_THREADS = '1'
$env:PYTEST_ADDOPTS = '-o cache_dir=outputs/pytest-cache --basetemp=outputs/pytest-ci'
.\.venv\Scripts\python.exe tools/ci.py
```

先前失败记录：初次 pytest 因系统临时目录/原有缓存权限错误，改用项目内目录后通过。
首次完整 CI 的 query 导入因系统 PyTorch Python 文件与二进制不匹配失败，sandbox_server 缺少 FastAPI。
仅在 `.venv` 补齐 FastAPI/uvicorn 并重装同版本 torch 2.10.0 后，入口检查通过；没有为此修改业务代码。
一条集成测试还发现“证据全部被过滤、配置目标实体不存在”的异常，现已改为明确报告无可用证据。
原始最终 CI 输出见 [validation/causal_evidence_ci_2026-10-18.txt](validation/causal_evidence_ci_2026-10-18.txt)。
