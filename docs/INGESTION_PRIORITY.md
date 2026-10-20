# 资料入库处理优先级

这是资料处理顺序，不是法律效力、证据真实性、检索相关性或 CHAIN 文本支持度。组件只根据声明的资料角色、加急要求和处理截止日期执行确定性规则，不读取 PDF、不调用模型，也不删除、过滤资料。

## 分级规则

| 等级 | 规则 | 处理含义 |
|---|---|---|
| P0 紧急 | 明确加急，或处理截止日期已过／在评估日期之后 7 天内（含当天与第 7 天） | 建议优先解析，并安排必要复核 |
| P1 核心 | 声明 `role=core`，且不满足 P0 | 当前任务指定的核心材料优先处理 |
| P2 常规 | 声明 `role=supporting`，或尚未分类 | 常规处理，未分类者另标记待分类 |
| P3 参考 | 声明 `role=reference`，且不满足 P0 | 保留并后续处理，不代表不可信或可以丢弃 |

窗口按自然日计算，可通过 `urgent_within_days`／`--urgent-within-days` 调整，允许为 0。`deadline` 接受日期对象或 `YYYY-MM-DD` 字符串，不接受数值时间戳；它是调用方声明的**资料处理截止日期**，本组件不推算任何法定期限。已过期任务不会因超期自动降级。评估日期必须明确提供，不隐式读取系统时钟。

`review_required` 与优先级独立。例如 P3 扫描件可以需要复核，P1 原生文字资料可以暂未要求复核。这一标记由调用方提供，并非组件自动验证了 OCR 质量。未提供角色时默认 `unclassified`、P2，并设置 `classification_required=true`；不能仅凭文件名或关键词推断核心材料。

如提供 `priority_override`，它优先于所有规则，但必须同时提供非空 `override_reason`；返回理由仍保留加急、期限和资料角色信息。仅提供覆盖理由或提供未知字段会报错，避免拼写错误被忽略。

## 接口与接入边界

`dataset/ingestion_priority.py` 提供 `IngestionMaterial`、`assess_priority` 和 `plan_ingestion`。
计划按 P0→P3、同级处理截止日期从早到晚排列；截止日期相同时保持输入顺序，没有截止日期者放在同级最后。重复 `document_id` 拒绝处理，同一原始文件的不同版本应使用不同文档 ID。空清单返回空计划。

每条计划包含原始声明 `material` 以及 `assessment`：策略版本、评估日期、窗口、等级、匹配理由、待分类和待复核标记。调用方可保存计划作为审计产物；这些字段不写入原文元数据，也不改变文档指纹。改变处理顺序不应制造新的证据版本。

当前接入已有 `tools/long_term_memory.py` 的 `prioritize` 子命令。它输出计划，不自动消费队列或启动后续导入；现有 `import-pageindex`、检索排序、快照和知识库格式保持原样。后续批量导入器可以按计划依次消费，OCR／版面分析后端再根据页面情况决定是否启用，不能用 P3 等级跳过证据提取。

清单是 JSON 数组，每项必须提供 `document_id`、`source_file`；其他字段见示例：

```powershell
Set-Location 'D:\project(now)'
.\.venv\Scripts\python.exe tools/long_term_memory.py prioritize --manifest docs/ingestion_manifest.example.json --as-of 2026-10-20
```

示例中的 PDF 路径是占位路径，生成计划不会打开这些文件，也不验证它们是否存在。角色和日期来自声明，不代表已经核实其真实性。`source_file` 按输入保存；实际导入时再按项目既有规则解析路径。

本组件不增加模型和依赖；单项评估只计算声明事实，计划排序复杂度为 O(n log n)。未做运行耗时基准，不据此宣称加速。

## 验证状态

已补充 `tests/test_ingestion_priority.py`：等级、期限边界、手工覆盖、严格输入、重复 ID、稳定排序、输入不变以及 CLI 不打开知识库／PDF 等用例。
按此前暂不运行测试的要求，本次这些用例和完整 CI 均未执行；静态检查结果见本次交付说明。当前没有法律材料分类准确率或大批量导入性能结论。
