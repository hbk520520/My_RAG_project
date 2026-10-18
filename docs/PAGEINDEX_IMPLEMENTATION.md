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

当前工作区未提供真实 PDF 或人工问题集。未据此宣称检索质量或生产性能提升。
