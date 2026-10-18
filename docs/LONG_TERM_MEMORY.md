# 项目知识长期记忆与 PageIndex 参考接入

当前能力：同一项目的原文知识、生成摘要、来源元数据和图向量可跨任务、跨进程重启恢复。
默认关闭；开启后同步 Reasoner 与 Kafka Retriever 共用这份知识。运行状态检查点仍由 `agent_runtime` 管理。
没有检索准确率提升或生产吞吐的实测结论。

## 工程依据与方案选择

原有调用链：`query.UnifiedQueryRouter_Query.process` → `LegalDenseGraphBuilder.from_config` →
`soul.Reasoner._retrieve_hits`；异步入口为 `retriever_worker.build_engine` → `retrieve_hits`。
两者已有 igraph + FAISS 和元数据接口，但构造后是空图；`IncrementalMemoryManager.MockGraphDB`
也只在进程内。`generate_summary_node` / `nightly_recalc_summaries` 生成的摘要没有存储层。

本次核对的参考仓库为 [VectifyAI/PageIndex](https://github.com/VectifyAI/PageIndex)，
其 [pyproject.toml](https://github.com/VectifyAI/PageIndex/blob/main/pyproject.toml)
标记 SDK 0.2.10 为 Alpha（查阅日期：2026-10-18）。按本项目“稳定库、最小修改”的约束，
没有安装 SDK、复制上游源码或依赖其云服务；只实现独立的导出格式适配器。

| 选择 | 兼容性、成本和风险 | 决定 |
|---|---|---|
| 扩展现有摘要/图引擎，SQLite 保存快照，接入 PageIndex 目录 JSON | 复用已有向量和接口；无新第三方依赖；适合本机中小规模，整图保存存在开销 | 本次采用 |
| 把索引和检索替换成 PageIndex SDK 的推理式导航 | 需引入 Alpha SDK、额外依赖和模型调用，改变现有检索行为，需要新的质量/成本基线 | 本次不采用 |
| 局部重构成逐节点持久存储、独立向量服务 | 可减少整图写放大并支撑多机器，但涉及数据模型、事务边界和运维改造 | 有实测规模瓶颈后再决定 |

借鉴的是**目录层级、摘要导航、页码溯源和索引复用**；检索仍由本项目的向量召回驱动，
不是 PageIndex 的无向量推理检索复现。参考仓库的
[搜索成本树优化器](https://github.com/VectifyAI/PageIndex/blob/main/pageindex/tree_optimize.py)
没有移植；已有独立法条可直接入库，不强加目录或额外冗余。

## 接入位置与行为

- `dataset/long_term_memory.py`：SQLite 原子保存 JSON + zlib 快照，包含原文、摘要、元数据、
  向量、语义边、软删除、脏摘要和聚类待处理标记；载入时从已有向量重建 FAISS，不调用 LLM 或重新编码。
- `dataset/graph.py`：现有入库、摘要生成/重算和软删除的成功写入保存到记忆库。
  批量导入和重算使用 `memory_transaction()`；异常恢复暂存图并原样抛出。
  直接改 igraph 内容后需调用 `persist_memory(rebuild_vectors=True)`，仅改元数据可省去参数。
- `dataset/memory_graph_bridge.py`：冷启动从持久图恢复 GMM 的节点、摘要、关联和 ID 空间；
  已生成并同步的摘要也持久保存，摘要同步失败不会提前清除运行时脏标记。
- `dataset/page_index_memory.py`：把已有目录树映射成 Summary 导航节点，把 PDF 原页映射成 Raw 证据。
  `source_node_ids` 记录可追溯原页；摘要命中展开回原文后再排序。摘要措辞不充当证据。
- `multiple-search/soul.py`、`asynchronization/workers/retriever_worker.py`：每次检索前刷新已提交知识，
  返回格式保持兼容；旧答案缓存缺少记忆版本，启用后跳过其读写。

SQLite 按 `project_id` 隔离；写入使用版本比较，旧写者抛 `MemoryConflict`，不能覆盖新版本。
本地失败写入会保留未保存标记，拒绝在下次检索时静默丢弃。
需要舍弃本地失败变更时，调用者明确执行 `refresh_memory(discard_unsaved=True)` 后重试业务操作。

## 启用与使用

在 `config.yaml` 开启：

```yaml
long_term_memory:
  enabled: true
  path: ".memory/rag.sqlite"
  project_id: "legal-rag"
  embedding_space_id: "bge-m3-base-v1"
```

路径相对项目根目录，数据目录已加入 `.gitignore`，不会随代码上传 GitHub。
入库后仍需自行保存原始 PDF；记忆库存储提取的页文本，不复制 PDF 文件。
Docker 重建或容器重启要保留宿主机磁盘并挂载到该路径；目前没有自动修改部署配置。

长期记忆模式下，入库和两条检索链使用 `graph.model_name` 对应编码器。
Reasoner 不使用路由分类器的另一套向量；Worker 不自动加载 `retriever.lora_path`。
原来关闭记忆时的 LoRA 行为保持不变。
`embedding_space_id` 是调用者声明的空间版本，不是自动权重校验器：更换模型、LoRA 或自定义编码器时，
必须为入库/查询同时配置相同编码器，并选择新的空间标识和库（或项目命名空间），重新入库。
模型名、维度、空间标识或建图参数与已存数据不符时，显式报错，不自动迁移。

普通原文/现有摘要继续使用已有接口：

```python
from dataset.graph import LegalDenseGraphBuilder

graph = LegalDenseGraphBuilder.from_config()
graph.add_node(1, "原文事实", "Raw", {"record_id": "source-v1:1", "source_id": "source-v1"})
graph.generate_summary_node(["原文事实"], existing_summary_function, child_ids=[1])
# 后续进程调用 from_config() 即恢复；新增 ID 必须与现有节点不同。
```

`existing_summary_function` 表示项目调用方已有的摘要函数，并非新增内置函数。
用多个原文生成摘要时应始终提供 `child_ids`；没有来源关联的摘要不会作为证据返回。
对多条知识写入，用 `with graph.memory_transaction():` 批量提交，减少重复快照。

接入已有 PageIndex 导出目录与 PDF：

```powershell
python tools/long_term_memory.py import-pageindex --tree document_structure.json --pdf law.pdf --document-id law-v1
python tools/long_term_memory.py inspect
```

CLI 显式导入会保存到配置的库，即使运行时功能开关还未开启；它不调用 PageIndex 建树服务。
默认使用真实图编码器，需要本地模型或下载条件。离线演示使用隔离库和空间：

```powershell
python tools/long_term_memory.py --path outputs/memory-demo.sqlite --space-id stub-v1 import-pageindex --tree document_structure.json --pdf law.pdf --document-id law-v1 --stub-encoder
```

接收的导出格式如下；页码是 PDF 的 1 起始物理页码，包含尾页：

```json
{"structure": [{"node_id": "0001", "title": "劳动合同", "start_index": 1,
  "end_index": 3, "summary": "导航摘要", "nodes": []}]}
```

也接收根节点列表。检查重复章节 ID、越界/倒置页码和子章节超出父章节范围。
同一 `document_id` 的相同内容/目录/元数据重复导入不重复建节点；改变资料须显式使用新文档版本。
扫描 PDF 无文本时明确要求先 OCR；本次不增加 OCR 服务。部分空白页不作为证据。

Python API `import_tree(..., metadata={...}, summary_fn=...)` 可接入已有摘要函数，
仅在导出节点缺少摘要时生成；未提供时使用标题导航，不伪造摘要。
`metadata` 可保存来源、发布时间和核实的可用性边界，继续交给因果证据组件；
导入时间不能冒充文档发布时间，也不自动推断截止日前可用性。

## 可靠性边界与下一决策门

- 本地 SQLite WAL + 原子快照适用于同一机器的持久磁盘；不用于网络共享盘或多机器 Worker。
  本次没有 Redis/Kafka 部署变更。跨进程写者通过版本冲突显式协调；同一图对象需串行读写，未提供线程安全接口。
- 每次保存包含整图向量，时间/空间开销随节点和边增加；刷新也读取快照。
  不声称降低成本或提升吞吐。大库上线前先测原始图检索、启用持久化、再加目录导航三组：
  相同查询和证据集下的 recall@k、页码引用正确率、p50/p95 延迟、批量写耗时、库大小和峰值内存。
  若导航没有召回收益或写入超过业务预算，保留结果，选择平面证据或逐节点存储，不扩大机制。
- 保存的是最新知识快照，不保留历史版本，不是 Agent 任务回滚或外部副作用补偿。
  未同步到图的 GMM 草稿簇/待同步队列不属于持久化内容；已成功写入图的原文不会因此丢失。
- 不把 Agent 回答、模型推测自动写成事实；PDF 目录摘要只做导航，证据来自原页。
  没有真实 LLM 质量实验、真实 Kafka/Redis 联机验证、Docker 构建或 Python 3.10 运行结果。

## 验证

专项测试：`python -m pytest tests/test_long_term_memory.py tests/test_memory_graph_bridge.py`。
覆盖跨进程恢复、项目隔离、并发旧写者冲突、损坏拒绝、向量空间不匹配、文档幂等、原文引用、
无文本 PDF、越界目录、生成失败回滚、摘要重算、GMM 冷启动，以及两条真实检索函数刷新。
统一工程检查使用 `python tools/ci.py`。

2026-10-18 工程验证：全量 419 项测试通过（此前 388 项 + 本次 31 项）；
入口安全导入、compileall、6 个离线 demo 和跨进程检查点恢复全部通过。
既有测试仍有 SQLite 连接未关闭的 ResourceWarning；新增存储连接均显式关闭。
初次专项测试的 PDF 旧模块名警告污染 CLI JSON，现改用官方 `pymupdf` 模块名；
PyMuPDF 最低版本相应从 1.23 调整为 1.24.3，没有新增依赖或批量升级。
跨进程测试随后暴露 Windows GBK/UTF-8 解码警告，已明确使用 UTF-8，最终回归不再出现该警告。

本机版本：Windows、Python 3.13.9、SQLite 3.51.0、numpy 2.3.3、faiss-cpu 1.15.1、
igraph 1.0.0、pydantic 2.12.3、PyMuPDF 1.28.2、pytest 9.0.3。
没有验证这些库的最低版本组合或 Python 3.10，也没有独立配置的 lint/类型检查可执行。
原始最终输出见 [validation/long_term_memory_ci_2026-10-18.txt](validation/long_term_memory_ci_2026-10-18.txt)。

实际全量测试命令（PowerShell，临时目录均位于忽略的 `outputs/`）：

```powershell
$env:TEMP = Join-Path (Get-Location) 'outputs/temp'
$env:TMP = $env:TEMP
$env:OMP_NUM_THREADS = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
.\.venv\Scripts\python.exe -m pytest -q -o addopts=--strict-markers -o cache_dir=outputs/pytest-cache --basetemp=outputs/pytest-memory-complete
$env:PYTEST_ADDOPTS = '-o cache_dir=outputs/pytest-cache --basetemp=outputs/pytest-ci-long-term-memory-final'
.\.venv\Scripts\python.exe tools/ci.py
git diff --check
```

前一条全量 pytest 为 418 项通过；补充软删除召回测试后，最终 CI 覆盖 419 项。
最终执行通过内存中的日志包装调用同一个 `tools/ci.py::main`，附带保留成功 pytest 的原始输出，
没有修改 CI 规则或跳过检查。再次运行时换一个空的 `--basetemp` 路径，避免 Windows 临时目录清理权限问题。

简历/面试的两项提醒独立保存在 [RESUME_REMINDERS.md](RESUME_REMINDERS.md)，不作为简历正文。
