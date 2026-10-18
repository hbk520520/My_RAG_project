# PageIndex 思路的本地融合

本项目独立实现适用部分，复用现有依赖、图引擎和 SQLite；未安装 Alpha SDK，也未复制上游源码。
选择依据见 [八项能力复核](PAGEINDEX_INTEGRATION_REVIEW.md)。

## 已实现：文档读取与引用

`dataset/document_store.py::DocumentReader` 从现有图快照提供文档清单、结构树、原页读取。
`from_file(path, project_id)` 使用 SQLite 只读连接，不创建缺失数据库，不加载 BGE、FAISS 或 LLM；
`from_engine(engine)` 创建当前任务的一致读取视图。`version` 可绑定明确的文档内容版本。
错误文档/版本、越界或已软删除的页会报错；不同 project_id 的资料隔离。

```powershell
python tools/long_term_memory.py documents
python tools/long_term_memory.py structure --document-id law-v1
python tools/long_term_memory.py pages --document-id law-v1 --pages 1 3
```

源位置增加可选 `document_version`，来自入库时已有 `document_fingerprint`；无需数据库迁移。
CHAIN 的可用原文分块保留这一版本。导航摘要不能加入事实来源表。

启用 `pageindex.enabled` 时，对含版本化文档证据的回答采用结构化引用：
模型只选择本次已读证据表中的 E1/E2 等标识，并提供对应原文的连续引句。
程序检查标识存在且确实使用、引句属于该原文、引用无重复，然后渲染普通文本答案和结构化 `citations`。
简单问答、LangGraph 最终报告和异步 Reasoner 都使用相同校验代码。
没有版本化文档的旧法条流程保持原来的文本回答接口；没有已读证据时不能生成带引用的答案。

**这验证来源身份、版本和引句存在，不证明引句在语义上充分支持结论。**
不能把该检查描述为事实正确率实验或引用蕴含准确率。真实文档和模型的评估仍需独立进行。

## 验证入口

`tests/test_pageindex.py` 覆盖只读/重启/项目隔离、版本和页码错误、删除页、合法及伪造引用，
以及同步节点、简单问答与异步 Worker 的真实调用路径；模型使用脚本化替身。
阶段 A 初次专项检查：51 项通过（包含 CHAIN 既有集成用例）；补充编号边界后全量 454 项通过。
既有五类 CI 全部通过（补充编号用例前），随后全量 pytest 覆盖了最终变更。
仍有既有检查点测试的 SQLite 连接 ResourceWarning。
命令为 `python tools/ci.py` 及 `python -m pytest -q -o addopts=--strict-markers -o cache_dir=outputs/pytest-cache
--basetemp=outputs/pytest-pageindex-a-final`，Windows TEMP/TMP 指向 `outputs/temp`。
原始记录见 [阶段 A 验证](validation/pageindex_a_2026-10-18.txt)。

## 已实现：结构构建与层次摘要

`dataset/document_index.py` 使用已有 PyMuPDF，从合法书签建立主框架，以明确编号/字体标题补充细节。
校验物理页码、顺序、标题和层级；无效书签及有歧义的结构补充进入 `diagnostics`。
节点和文档保存 `bookmarks` / `detected` / `hybrid` / `flat` 来源；没有可靠标题时保留平面根节点。
这是轻量确定性实现，不代表完整移植上游 flash 的版面识别质量；扫描件仍要求先 OCR。

```powershell
python tools/long_term_memory.py import-pageindex --pdf path/to/document.pdf --document-id document-v1 --generate-summaries
python tools/long_term_memory.py cost --document-id document-v1
```

不提供 `--tree` 时从 PDF 建树，仍可提供上游 JSON；不提供 `--generate-summaries` 时不调用模型，
只保留已有摘要/标题。调用现有 JSON 模型接口，模型或编码失败则回滚整个导入。
物理空白页照样保存，全文无文本则拒绝。保留 `key_items`；文档来源 ID 不随摘要改写变化。

摘要导入和重算均先子后父，父输入包括子摘要与未被子节点覆盖的正文。
脏标记覆盖共享原页的所有章节；删除所有原文时清除旧摘要；重算失败不发布部分新摘要。
图引擎、增量记忆管理器及桥接器均使用真实依赖排序，不再依赖任意集合顺序或手填 level。
PageIndex 文档按其明确目录挂载，不参与普通语义簇的自动挂载。

`diagnose_cost` 包含子节点未覆盖的页，只给出最坏分支的页成本与 merge 建议。
它不改变树、不自动展开，也不是实测 token、时延或性能收益。

专项验证：`tests/test_pageindex_structure.py` 与既有记忆/桥接/阈值测试共 99 项通过。
既有五类 CI 全部通过；随后补充空白章节与中文标题层级边界，再次执行上述专项检查。
执行记录见 [阶段 B 验证](validation/pageindex_b_2026-10-18.txt)。
按用户要求，本轮不进行真实 PDF 问题集效果评测；不宣称检索质量或生产性能提升。
