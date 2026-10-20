# 按资料用途和提取需要选择加工深度

当前实现使用固定版本 `rule-engine==5.1.0` 执行 `dataset/material_rules.json` 中的规则。适配层只映射已有字段、复用项目导入流程，不新增语义分类模型。PDF 可选检查器是 `pdf-inspector==1.25.2`，安装清单为 `requirements-ingestion.txt`；它只调用 `detect_pdf`，不启动 OCR、下载模型或上传文件。

## 默认规则

| 资料用途 | 轻量路径 | 深加工路径 |
|---|---|---|
| 规范资料 `normative` | 原始法条已有正文和结构，或短小、可读的 PDF | 存在提取困难，或未结构化 PDF 超过 8 页 |
| 案件资料 `case` | 不超过 8 页且没有发现提取问题 | 存在提取困难，或超过 8 页 |
| 参考资料 `reference` | 原文页和基础向量索引；不生成目录节点或摘要 | 不自动升级；需要时重新选择处理策略 |
| 派生资料 `derived` | 原件留存，返回 `archive`，不加入原始证据库 | 不自动处理 |
| 无法分类 `unknown` | 原件留存，返回 `deferred`，不启动图引擎 | 不自动降为参考资料 |

`auto` 只根据 JSONL 中非空的 `law_name`、`article_no`、`text` 识别现有法条结构，或者根据显式 `derived=true` 隔离派生资料；其他未声明用途的数据保留待分类。该匹配不证明资料真实、具有特定法律效力或仍然有效。PDF 没有相同结构字段时，不凭文件名或标题猜测重要性，可通过 `--material-kind` 声明来源类型。

原始法条的历史版本完整保留。缺少来源或时间字段会标记 `review_flags`，不降低资料类别、不修改原文。8 页是首期可调整规则，不是实测的性能最优阈值。规则按文件顺序匹配，派生标记优先，未知资料最后处理；编译结果只缓存一份，修改策略后应重启进程。

## 现有 JSONL：流式分流报告

```powershell
Set-Location 'D:\project(now)'
.\.venv\Scripts\python.exe tools/long_term_memory.py route-corpus --input benchmark_causal/data/legal_corpus.jsonl --output outputs/intake/law-routing.jsonl
```

报告逐条保存源文件绝对路径、物理行号、UTF-8 规范化行文本的 SHA-256，以及类别、加工路径、规则 ID 和版本。原始正文不复制进报告，也不重新编码或建图；它不是第二份知识库。一次只读取一个 JSONL 记录，内存仍受最大单条记录大小约束。输出使用同目录临时文件和原子替换，输入出错不会覆盖旧报告，不允许输出覆盖原始语料。

输出路径由 CLI 限制在 D 盘。处理报告必须与原始语料一起保留才能追溯全文；行哈希可以发现对应记录已改变。报告不会自动将整个评测语料混入在线证据库，也不会自动迁移训练集。

## PDF：实际限制加工深度的导入入口

```powershell
.\.venv\Scripts\python.exe tools/long_term_memory.py import-document --pdf 'D:\资料\参考.pdf' --document-id reference-v1 --material-kind reference
.\.venv\Scripts\python.exe tools/long_term_memory.py import-document --pdf 'D:\资料\案件.pdf' --document-id case-v1 --material-kind case --pdf-inspector
```

新入口先选择资料用途，再进行原生文字提取；不渲染整份 PDF 为图像。未知或派生资料只计算原文件哈希并返回状态，不打开 PDF 解析器、不创建图引擎。可选检查器必须显式选择，未安装会直接报错，不静默切换解析器。

轻量路径调用现有 `import_tree`，只传递原文页及空目录，不建立层次摘要节点，不调用摘要模型。它仍使用项目原有的向量编码、FAISS、原文语义图和持久化流程，不宣称整个入库过程没有模型或图内存。

深加工路径目前复用现有目录构建器，只有显式 `--generate-summaries` 时才生成摘要。需要 OCR 或检测到复杂版面、尚不能可靠处理时，返回 `pending_extraction` 和具体页码／检查信息，**不会先创建图引擎或把缺页结果写成完整证据**。本期没有接入 OCR、PaddleOCR 或复杂表格模型，因此这些资料尚不能自动完成入库。`detect_pdf` 也不是完整的版面质量保证，未报告复杂版面不能证明阅读顺序正确。

原生检查保守地把空文本页和含替换字符的页列为待提取；真正空白页也可能被拦截，需后续确认。可选检查器的疑似 OCR 页与原生检查取并集，不会覆盖掉原生发现的问题。使用 `PdfResult` 的 1 起始物理页码，不混用另一个 `classify_pdf` 接口的 0 起始页码。检测策略可能采样，不能用未命中当作“不存在扫描页”的保证。

原 PDF 不修改、不移动。已导入的原文页保留文档 ID、物理页码、原文件路径与文件哈希；导入事务继续由原有实现管理。调用方必须保留原 PDF，并保存命令 JSON 输出以追踪 `deferred`／`archive`／`pending_extraction` 状态；本期没有新增持久作业队列或自动重试。加工路径和规则版本放在输出报告中，不混入 CHAIN 分数或法律效力判断。

原 `import-pageindex` 接口及其行为保持兼容，不自动套用新规则。它是显式兼容入口，不能作为新分流边界的强制保障。新旧入口使用的目录／来源元数据不同，同一资料更换加工表示时应采用明确的新文档版本 ID，不能假定可以覆盖已有版本。

## 验证与性能边界

测试用例位于 `tests/test_material_routing.py`，覆盖用途与结构的组合、8／9 页边界、历史数据保留、失败报告原子性、轻量路径不生成摘要、缺页不入库，以及可选检查器页码约定。按此前暂不运行测试的要求，本轮未运行这些测试、完整 CI、OCR 联调或性能基准。

已实际执行 `route-corpus` 为现有语料生成 `outputs/intake/law-routing.jsonl`：1,428 条均识别为规范资料并选择轻量路径，1,087 条标记缺少 `valid_from`；原语料无 Git diff。该结果只验证这批结构化资料的分流，不代表 PDF 分支已经通过运行验收。命令和检查记录见 `validation/material_routing_2026-10-20.txt`。

组件按记录流式分流，仅保留固定数量的已编译规则，不保存全量语料向量。PDF 原生解析、已提取页文本和后续图引擎仍会消耗内存；原文件哈希也需要顺序读取文件。没有实测峰值 RSS、每页延迟或总体加速比例，不能把依赖 wheel 大小当作内存占用。
