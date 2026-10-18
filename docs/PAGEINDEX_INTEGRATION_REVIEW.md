# PageIndex 八项能力：本项目适用性复核

复核日期：2026-10-18；依据上游 main 分支实际函数，而非固定行号（用户给出的部分行号已移动）。
本文保留融合前的分析基线与取舍。后续实际完成项和当前行为见
[PAGEINDEX_IMPLEMENTATION.md](PAGEINDEX_IMPLEMENTATION.md)；未将八项全部宣称为已实现。

## 结论与当前基线

优先完善 **6 来源溯源、8 职责分离、3 分层摘要、4 结构来源融合**。
**1 布局建树、5 Agent 按需读分支**适合长 PDF，先做小实验；**2 搜索成本优化**须在树导航真正使用后评价；
**7 多模型统一接口**按真实后端需求局部整合，不引入整套多框架依赖。
SDK 的 Alpha 标记影响依赖选择，不意味着其中的独立方法都不适用。

已核实的本地事实：

- `dataset/page_index_memory.py::import_tree` 已接受已有目录 JSON；本身不从 PDF 自动检测标题。
  `Section` 只接收 ID、标题、页范围、摘要、子节点，未保留上游 `toc_source` 和 `key_items`。
- 导航仍是向量命中摘要 → `dataset/long_term_memory.py::source_evidence_hits` 展开关联原文 → 向量排序。
  **这还不是 LLM 逐层选择分支的 PageIndex 检索。**
- `dataset/graph.py::generate_summary_node` 和 `::_rewrite_summaries` 已有摘要和脏标记；后者按集合遍历，
  未确保父节点等待子节点新摘要。`IncrementalMemoryManager.nightly_rewrite` 的层级排序依赖外部 level，
  不能代替真实依赖排序。导入缺失摘要时，目前各章节仍直接读其覆盖的原页。
- `MemoryStore` 已用 SQLite 保存原文/摘要/向量并检测写入版本冲突，但载体仍是整图压缩快照；
  构建与检索都由 `LegalDenseGraphBuilder` 承担，尚无独立的文档/目录/页读取接口。
- `soul` 与 Worker 已有规划、补搜和重规划；当前选择的是语义图游走或全局向量检索，
  不是文档级 `get_structure` / `read_pages` 的主动工具循环。
- 原文元数据已有文件名/页码；本轮 CHAIN 补齐了它们在证据记录、分块和模型上下文中的传递。
  最终回答仍是自由文本，没有程序逐条校验引用是否来自实际已读证据。

当前只有工程测试基线，没有真实文档上的 PageIndex 对照数据。以下收益均为待验证假设。

## 八项能力逐项判断

| 技术点 | 上游已核实行为 | 本项目建议及接入点 | 改动与风险 |
|---|---|---|---|
| **1 初始结构树后再做语义处理** | `extract_toc` / heading detection 用布局、编号、字体和邻域信号组装目录；LLM 语义处理出现在后续摘要、展开或 Agent 阶段。[源码](https://github.com/VectifyAI/PageIndex/blob/main/pageindex/flash/main.py) | **部分可直接融合**：优先 PyMuPDF 书签/明确标题，再对少量歧义节点做语义处理；结果交现有 `import_tree`。纯法条优先已有条号/章节字段。 | 轻量入口中等；整套 flash 依赖许多布局规则，不宜直接搬入。中文、页眉和扫描件需独立验证。 |
| **2 按搜索成本 merge / expand** | `tree_cost` 包含未被子节点覆盖的正文；merge 保留标题到 `key_items`。expand 同时要求页成本下降、达到最小收益比例和读取 token 估计下降。[源码](https://github.com/VectifyAI/PageIndex/blob/main/pageindex/tree_optimize.py) | **确定性成本诊断可以复用，自动展开暂缓**。先保留 `key_items` 并离线比较树/平读成本；不能拿这个模型直接改 FAISS 图边或 CHAIN 因果边。 | 成本函数小，模型提议/验收/失效传播较大。R=1 页是近似，不是实测时延，也不包含全部建索引成本。 |
| **3 自底向上分层摘要** | `SummaryScheduler` 等待子节点完成；叶子读原页，父节点组合子摘要及自身未覆盖正文。[源码](https://github.com/VectifyAI/PageIndex/blob/main/pageindex/utils.py) | **适合直接完善**：在现有摘要入口按 parent_id 后序处理脏节点，明确自己的正文；先串行，复用 summary_fn，不新造调度框架。 | 中等；重点防止父摘要过期、遗漏父节点正文、同页重复输入。摘要始终不是因果验证的事实来源。 |
| **4 多结构来源融合** | 校验/分类书签后选择 FULL、SKELETON、IGNORE 对应的融合策略，输出 bookmarks、hybrid、detected。[源码](https://github.com/VectifyAI/PageIndex/blob/main/pageindex/flash/embedded_toc.py) | **轻量版本可直接融合**：用已安装 PyMuPDF 读取书签；验证页码/顺序/标题，保存 `toc_source` 与节点来源；章框架与细节标题按范围融合。 | 中等；书签页偏移、同名标题和互相冲突的层级必须显式处理。不能简单拼接两个树。 |
| **5 Agent 按需获取信息** | 暴露文档浏览、目录和页内容工具；长文先看结构，短文可直接读取。[源码](https://github.com/VectifyAI/PageIndex/blob/main/pageindex/agent_tools.py) | **可以融合，但不是一个开关**：在现有 Executor/Worker 加只读目录/页动作，保留向量检索为基线；补搜时按问题选择下一步。 | 中等偏高；需要动作契约、页/token/轮次预算及一致的同步/异步行为。只开放项目库中的文档 ID，不能让模型直接指定任意文件路径。 |
| **6 原文来源与引用解析** | 页范围用于原文读取；`get_citations` / `resolve_citations` 解析引用并提供可展示结果，本地引用是页级。[源码](https://github.com/VectifyAI/PageIndex/blob/main/pageindex/client.py) | **最适合直接完善**：沿已有 record/document/page ID 建证据表，最终回答引用已读 ID，程序检查文档版本与页范围，再渲染。 | 小到中等；“页存在”不等于“原文支持结论”，后者仍需引用文本检查/人工评估。本轮只补齐 CHAIN 来源传递。 |
| **7 多模型统一接口** | Client 区分本地/云及不同模型协议，含多种框架适配。[源码](https://github.com/VectifyAI/PageIndex/blob/main/pageindex/client.py) | **局部采纳即可**：复用 `_call_messages` 注入点，逐步统一 Worker API 调用与现有 vLLM `.chat()`；明确模型/JSON能力和错误返回。 | 接口整理中等，整套 SDK/框架成本高。当前主要使用 OpenAI 兼容端点，不需要预先支持所有供应商。 |
| **8 构建、存储、检索分离** | LocalAPI 组织索引过程，DocStore 分别保存 meta/tree/pages 并维护清单，JSON 写入采用临时文件替换。[存储](https://github.com/VectifyAI/PageIndex/blob/main/pageindex/local_store.py)、[本地 API](https://github.com/VectifyAI/PageIndex/blob/main/pageindex/local_api.py) | **职责可直接拆清**：先在现有存储上提供纯读取视图，使检索无需再次建树/摘要；物理分表和逐节点写入在实测写放大后再定。 | 逻辑拆分中等。保留现有 SQLite 原子提交/版本检查；多 JSON 文件分别原子写不等于整个文档跨文件原子事务。 |

特别核实：上游 `DocStore.lock` 在 Windows 没有 fcntl 时采用 best effort；读取损坏 JSON、写 manifest 的某些异常会降级。
本项目采用显式失败和版本冲突检测，**不应原样移植这些错误处理和并发语义**。
同样，外部仓库的 SDK 版本、模型后端和框架清单不能作为本项目已支持的能力清单。

## 与 CHAIN 的接合位置

未来推荐的数据路径：文档解析/树构建 → 持久化原页与导航结构 → 向量或 Agent 按需读取 →
原文 EvidenceRecord → CHAIN 提取验证/时间权重 → Grader/Reasoner → 已读证据引用校验。

目录树描述文档结构；语义图表达相似性；CHAIN 超图表达从原文提取并验证的关系。
三者不能互相当作因果证据，也不需要合并为一个大图。目录上更近不等于因果距离更近。
模型生成摘要、重新编号树节点或合并目录时，原页引用和稳定 record_id 不能随之失效。

## 最小实施顺序和可独立验收阶段

| 阶段 | 目标与涉及位置 | 验收、回归风险和依赖 |
|---|---|---|
| A：6 + 8 的读取契约 | `page_index_memory.py` / `long_term_memory.py` 提供项目内 list_documents、read_structure、read_pages；答案引用绑定已读 record_id | 跨重启同文档版本/页文本一致；越界/错误版本拒绝；错误来源无法渲染为有效引用。保留现有文本检索签名；不要求立刻迁移数据库。 |
| B：3 + 4 的可靠摘要树 | 在现有摘要重算中按真实父子依赖处理，覆盖父正文；导入时保留 toc_source/key_items，增加书签校验 | 子节点更新先于父；部分失败不发布半成品；页覆盖完整；重叠章节不重复增算支持度。依赖 A 的稳定来源 ID。 |
| C：1 + 5 的小规模导航实验 | 轻量建树入口 + 现有 Agent 的只读目录/页动作；同步和 Worker 使用同一动作实现 | 固定同一查询集，与向量检索和现有摘要展开比较；记录所有读取页/token/LLM调用。没有召回收益或超出预算则停止扩展，保留原检索。 |
| D：2；必要时 7 | 先成本诊断，再决定 merge/expand；确有第二模型后端需求时统一调用接口 | merge 前后原页可达性不变；key_items 不丢；expand 只能采用原文标题且实测读取成本有收益。模型切换通过相同 JSON/失败契约测试。 |

物理存储分表若涉及已有库迁移，须在实施前另行批准；当前分析不执行迁移。
其他阶段可沿现有模块局部扩展，无需更换检索框架、引入 GNN、复制整个 PageIndex 或大规模升级依赖。

## 最小实验与停止条件

先固定少量已整理长 PDF 和纯法条文件，给同一批问题标注所需原文页/条号；不把模型回答当真值。
三组基线为：现有向量原文检索、现有摘要展开、候选树导航。模型、语料版本、问题、截止日与输出预算保持一致。
记录 evidence recall@k、人工核对的引用支持率、引用 ID/页码合法率、输入输出 tokens、调用数、
p50/p95 查询延迟、离线构建耗时和峰值内存；构建成本与查询成本分别计量。

工程门槛是原页覆盖/来源一致性不回退、非法引用不能通过验证；收益门槛是在预先约定的检索质量下减少实际成本，
或在相同预算下提高证据召回。页成本模型降低但实测 token/时延未降低，不能报告性能提升。
长 PDF 与独立法条分组报告；若纯法条增加层级没有收益，保留其已有编章条结构即可。

尚未确认的是哪种资料分布/查询负载能获益，以及真实模型的目录路由错误率；不是代码能否拼接。
简历中的取舍解释仍见 [RESUME_REMINDERS.md](RESUME_REMINDERS.md)，不能把这些待验证假设写成实验结论。
