# 项目完整功能清单

核对日期：2026-11-09。源码基线：`804914c`，分支 `main`。本文件记录该基线已有代码；生成本文件不改变业务实现。

范围：在线问答与任务执行、资料处理、检索与记忆、因果证据、计算与沙箱、模型接口、数据生成、训练、评测、运维、兼容入口及工程验证。附录逐文件索引覆盖该基线全部受 Git 管理的 Python 源码和测试文件。配置或权重存在并不代表训练已经完成；有测试文件也不代表本轮运行过测试。

状态用语：**默认路径**表示默认配置允许或默认入口使用；**按需**表示任务或调用者选择后执行；**可选**表示需显式配置、参数或依赖；**离线组件**表示独立使用；**历史兼容**表示保留代码但不具有当前主流程全部保证。本文的“支持”首先指实现能力，不自动表示已通过真实模型、GPU 或线上服务验收。

## 目录

1. [项目定位、入口与架构](#1-项目定位入口与架构)
2. [任务运行服务与生命周期](#2-任务运行服务与生命周期)
3. [持久化、恢复、回滚与备份](#3-持久化恢复回滚与备份)
4. [规划、执行、重规划与信息交接](#4-规划执行重规划与信息交接)
5. [并行调度和共享资源](#5-并行调度和共享资源)
6. [检索、事实判断与报告生成](#6-检索事实判断与报告生成)
7. [文本语料处理](#7-文本语料处理)
8. [资料分流和 PDF 导入](#8-资料分流和-pdf-导入)
9. [语义知识图、增量记忆和摘要](#9-语义知识图增量记忆和摘要)
10. [跨任务长期知识记忆](#10-跨任务长期知识记忆)
11. [文档目录、导航和引用](#11-文档目录导航和引用)
12. [CHAIN 因果证据和权重](#12-chain-因果证据和权重)
13. [确定性计算与代码执行](#13-确定性计算与代码执行)
14. [模型调用、本地推理和缓存](#14-模型调用本地推理和缓存)
15. [PA-Tool 工具 Schema 对齐](#15-pa-tool-工具-schema-对齐)
16. [本地模型行为优化组件](#16-本地模型行为优化组件)
17. [合成数据与旧数据生成器](#17-合成数据与旧数据生成器)
18. [训练代码](#18-训练代码)
19. [检索、轨迹和因果评测](#19-检索轨迹和因果评测)
20. [异步消息、历史兼容和部署文件](#20-异步消息历史兼容和部署文件)
21. [日志、指标、配置和工程验证](#21-日志指标配置和工程验证)
22. [功能边界与未实现事项](#22-功能边界与未实现事项)
23. [默认开关和预算](#23-默认开关和预算)
24. [公开操作入口](#24-公开操作入口)
25. 源码逐文件索引（附录自动按本次基线盘点）
26. 测试与验证证据索引（附录自动按本次基线盘点）

## 1. 项目定位、入口与架构

实现位置：[README](../README.md)、[模块职责](MODULAR_ARCHITECTURE.md)、[运行服务](../agent_runtime/service.py)、[流程](../legal_agent/workflow.py)。

- 面向单机或受信小团队的法律 RAG Agent，围绕法律问题检索、分解任务、收集证据、判断信息充分性、补充信息、推理和生成报告。
- 模块分工：`legal_agent` 编排业务，`agent_runtime` 管理持久执行，`agent_tools` 提供计算与沙箱，`dataset` 管理知识，`causal_evidence` 处理文本因果关系，`model` 保存模型与训练组件，`benchmark_causal` 提供独立评测。
- 默认 CLI、HTTP、Kafka 作业和查询包装入口共用 `AgentService` 及 LangGraph 流程。
- 统一运行时之前的查询、缓存和五阶段 Worker 代码保留兼容，不能把这些旧入口等同于默认执行路径。
- 主流程节点为 L0 网关、Planner、Executor、Replanner、AwaitInput、Generate、WriteCode、ExecuteCode、InjectResult、Cleanup，按条件跳转。
- 闲聊直接返回固定问候；简单问答构造一个直接检索任务；复杂任务由 Planner 分解。
- `pyproject.toml` 提供 `legal-rag-agent` 包及 `legal-agent` 命令；保留旧模块的导入兼容和显式执行入口。
- 项目运行路径、脚本缓存和模型缓存按当前约定放在 D 盘。

## 2. 任务运行服务与生命周期

实现位置：[contracts](../agent_runtime/contracts.py)、[service](../agent_runtime/service.py)、[store](../agent_runtime/store.py)、[HTTP API](../agent_runtime/api.py)。

- 提交问题并生成独立 `run_id`，记录项目 ID、提交参数、传输方式和预算。
- 请求契约包含问题、项目 ID、可选预测截止日、目标实体、运行预算和 `allow_code`。
- 请求校验：拒绝空白问题、过长问题、不合法项目 ID、非法日期和额外未知请求字段。
- 生命周期：`queued`、`running`、`waiting_input`、`succeeded`、`failed`、`cancelled`、`timed_out`。
- 查询单次运行的状态、结果、错误类别、预算使用及运行元数据；按项目列出最近运行。
- 持久记录运行事件，支持按事件游标读取后续事件。
- 用户主动取消；租约和 epoch 阻止已经取消任务的旧执行者写回结果。
- 缺少用户事实时挂起并保存问题，收到补充答案后从检查点继续。
- 补充答案要求非空、长度受限；等待和恢复沿用原 deadline 与预算。
- 支持失败运行的显式重试、排队任务执行，以及过期执行租约后的恢复扫描。
- 将预算、资源冲突、模型传输、契约校验、结果不确定、执行权丢失和内部错误区分记录。
- 完成检查包括流程没有未完成节点、存在报告，以及未出现预算耗尽或计算失败标记。
- `succeeded` 是运行生命周期结果，不是独立的法律结论正确性或业务完成质量证明。
- HTTP 使用小团队共享 Bearer Token；除 `/ready` 外的任务接口需鉴权。
- HTTP `/ready` 检查运行库可读及本地执行线程状态；不检查外部模型服务质量。
- HTTP 本地模式后台推进任务；Kafka 模式只提交持久化作业，由同机 Worker 执行。
- HTTP 错误映射包含不存在 404、状态冲突／结果不确定 409、输入错误 422 和就绪失败 503。

## 3. 持久化、恢复、回滚与备份

实现位置：[store](../agent_runtime/store.py)、[检查点写入](../agent_runtime/checkpointer.py)、[执行上下文](../agent_runtime/context.py)、[备份恢复](../agent_runtime/backup.py)。

- SQLite 运行库保存任务状态、租约、epoch、预算、事件、工具执行记录、固定知识视图和消息 outbox。
- 配对 SQLite 检查点库保存真实 LangGraph 状态，支持进程退出后续跑。
- 执行租约配合周期续租，限制同一运行只有一个有效拥有者。
- 检查点状态和增量写入校验执行权，防止旧进程污染新运行。
- 首次使用知识时固定该运行的图快照与知识 revision；后续恢复继续读取同一视图。
- 固定模型相关配置、提示词及启用的工具命名映射内容哈希；配置变化时拒绝混用旧检查点。
- 工具调用先进入持久台账，再提交节点检查点；已完成的读取或分支结果可在重放时复用。
- 工具调用标识区分运行、epoch、节点、步骤、任务 ID、任务 revision 和输入摘要。
- 工具异常保留失败信息；沙箱变更请求意图可保留于同一台账，供不确定结果核对。
- 支持列出检查点并选择当前运行的历史检查点回退。
- 回退提高 epoch，使旧执行、旧消息及旧分支不能覆盖回退后的状态。
- 对未知外部副作用保留 `outcome_unknown`，不把它自动当成“未执行”。
- 消息 outbox 与运行变更原子落库；Broker 确认发送后再标记发送完成。
- 备份前要求没有有效运行租约，并在复制期间阻止新的执行认领。
- 使用 SQLite backup API 备份运行库和检查点库，检查完整性，发布带 SHA-256 的 manifest。
- 恢复检查文件哈希、数据库完整性，只恢复到新目录，不覆盖已有数据库。
- 备份包含运行固定的证据视图；项目知识库、模型、原始 PDF 与凭据需另外备份。
- 回滚范围是任务执行状态及可重放结果；不会撤销已经产生的模型账单，也不是任意外部事务补偿。
- 没有自动清理运行历史、检查点、快照或备份的保留策略。

## 4. 规划、执行、重规划与信息交接

实现位置：[双层计划](../double_layer_plan.py)、[operators](../legal_agent/operators.py)、[nodes](../legal_agent/nodes.py)、[scheduling](../legal_agent/scheduling.py)、[硬规则](../replanner_rules.py)。

- L0 使用中文、英文及分隔符关键词规则识别部分提示注入模式；这是启发式检查。
- 共享免模型前置分类：闲聊、简单问答、复杂任务。
- Planner 可输出双层蓝图：抽象骨架 DAG `S_q` 与具体检索实例化 `C_q`。
- 校验计划依赖、检测环、拓扑排序、计算可并行组，并将蓝图转为可执行队列。
- 兼容扁平 `task_queue`、`strategy_queue` 和旧字符串任务；保留旧 `[WORMHOLE]` 语义。
- `TaskSpec` 保存任务 ID、任务描述、依赖、检索引擎、理由、证据策略和 revision。
- 简单问答免去 Planner 模型调用，直接使用一个关闭因果分析的检索任务。
- 单任务检索、Grader 事实提取、充分性判断、按需独立推理、生成观察并更新完成列表。
- Grader 输出 `sufficient`／`partial`／`irrelevant`，事实、评分理由、缺失信息和缺失类型分别记录。
- 缺失的是用户事实时请求补充；缺失的是证据时走重试或重规划。
- 重试保留任务独立的失败上下文；达到阈值或检索不相关时触发 Replanner。
- Replanner 可修改检索词、任务及检索引擎；只处理失败分支，不重做已成功兄弟任务。
- 全局稠密“虫洞”用于绕过当前语义图局部邻域，不代表另一个独立知识数据库。
- 重规划提高任务 revision，区分旧计划结果与新计划结果。
- 递归深度和运行预算限制重复检索、重规划和执行循环。
- 完整性规则在合并后的观察上检查，包括最后一个任务完成后的检查。
- 硬规则具有确定性任务 ID 和去重，规则新增步骤按顺序依赖执行。
- 已实现的七类硬规则：未签劳动合同的二倍工资、工伤认定及赔偿、社保缴纳和金额、加班费、竞业限制和补偿、试用期合法性和解除条件、劳务派遣责任及法律关系。
- `replanner_rules_report.py` 的扩展场景是分析材料，不应按“已注册规则”计算。
- 完成任务保存事实、任务版本和用户事实；依赖任务接收前置任务事实。
- 用户补充按任务隔离，仅传给对应任务及显式依赖任务；不自动广播到无关兄弟分支。
- 观察列表追加保存，事实池与评分理由分开；Generator 汇总观察和已读证据。
- 并不是每个子任务固定运行一个 Extractor 再运行一个 Reasoner；常规事实抽取在 Grader 中完成，独立 Reasoner 仅在策略要求且证据充分时执行。

## 5. 并行调度和共享资源

实现位置：[scheduling](../legal_agent/scheduling.py)、[resources](../agent_runtime/resources.py)、[共享编码器](../agent_runtime/models.py)、[并行说明](PARALLEL_AGENT.md)。

- 从依赖已完成的就绪任务中选取有界批次，由线程池并发执行。
- 任务开始前校验 ID 唯一、依赖存在、无依赖环。
- 每个分支有独立运行上下文、重试信息和检索轨迹，共享同一运行只读知识视图。
- 分支返回结果，由单一协调器按计划顺序合并；不按完成快慢改变事实顺序。
- 批次内结果先落台账，兄弟分支失败或进程退出后可复用已完成结果。
- 运行级并发、任务级并发、远程模型、本地模型、编码器和沙箱分别准入。
- 有界等待者数量，等待期间检查取消、deadline 和执行权。
- 操作系统文件锁约束同机不同进程的资源使用，进程退出释放锁。
- 本地加速器拥有者锁阻止第二个执行进程重复加载常驻 GPU 模型。
- 一个进程内复用本地 vLLM 实例；修改已加载模型配置要求重启拥有者。
- 编码器按模型、设备、精度、维度及嵌入空间身份复用，缓存数量有上限。
- 同一编码器的调用串行化；自定义注入编码器保留其身份。
- 记录资源等待和工具执行耗时，供后续性能分析。
- 当前采用批次屏障式推进，没有连续抢占调度、严格公平、多机调度、GPU 字节级分配或自动多 GPU 放置。
- 借鉴 TokenSpeed 的调度分层与资源准入思想；没有直接使用其推理引擎或内核库。

## 6. 检索、事实判断与报告生成

实现位置：[retrieval](../legal_agent/retrieval.py)、[reasoning](../legal_agent/reasoning.py)、[operators](../legal_agent/operators.py)、[evidence](../legal_agent/evidence.py)。

- 使用 BGE-M3 稠密向量及 FAISS HNSW 召回候选。
- 普通图检索在向量候选附近做语义邻接扩展；全局稠密模式不做局部图扩展。
- 过滤软删除节点；空索引在编码前结束检索。
- 摘要命中展开到关联原文，按原文重新组织结果；无来源关联摘要不充当原始证据。
- 长期记忆／因果模式查询复用知识图编码器，避免和路由编码器混用向量空间。
- `evidence.navigation` 选择自动、向量或 Agent 文档导航。
- `evidence.causal` 选择自动、关闭、文本验证或时间权重；`analysis` 选择事实或推理。
- 自动因果选择使用截止日／目标和因果问题词启发式；普通法规问答可跳过因果提取。
- 截止日过滤在原文进入事实判断或因果提取前执行。
- 原文证据审计保留记录、来源、文档版本、页码和导航轨迹。
- Grader 提取事实并判断证据充分性；评分理由不冒充业务推理。
- `analysis=reasoning` 且证据充分时才调用独立 Reasoner。
- 最终报告汇总已完成观察、因果／普通证据、用户补充和程序计算结果。
- 有版本化文档证据时使用结构化引用生成与校验；旧无版本法条回答保持文本接口。
- 证据不足、无目标、检索为空、模型失败、格式错误分别保留明确状态或错误，不应解释为成功回答。
- 用户补充是案件事实，不混入 CHAIN 的原文来源。

## 7. 文本语料处理

实现位置：[chunk](../dataset/chunk.py)、[prepare_corpus](../dataset/prepare_corpus.py)。

- PDF 原生文本提取及命令行分块预览；该旧工具提取全文，不等于版本化 PDF 导入。
- 支持按自然段或中英文句末标点分块，合并过短片段。
- 可按最大字符长度再次按句子切分；这不是严格 token 长度上限，也不会强切所有超长句。
- 文本语料目录支持 `.txt`、`.md`、`.text`，按文件名排序，保持节点 ID 可复现。
- 当前目录扫描不递归，不直接把该入口当成 PDF 批量扫描器。
- 节点保存整数 ID、正文、Raw 类型、源文件名、块序号、字符长度、原文指纹、record/source ID。
- 使用默认图编码器、注入单条编码器或显式离线 StubEncoder 向量化。
- 导出 `nodes.jsonl` 和 `vectors.npy`，保证行数和顺序对应。
- 支持预计算向量直接建图，避免重复编码。
- `CORPUS_DIR` 或 CLI 参数指定输入；缺失目录、无支持文件等问题明确报错。

## 8. 资料分流和 PDF 导入

实现位置：[material_routing](../dataset/material_routing.py)、[规则](../dataset/material_rules.json)、[pdf_intake](../dataset/pdf_intake.py)、[说明](MATERIAL_ROUTING.md)。

- 用固定版本 `rule-engine` 执行声明式规则，不额外加载语义分类模型。
- 按资料用途、已有结构、页数及提取困难选择加工深度；不把“重要”简化为单一相关性分数。
- 规范资料 `normative`：已有法条结构或短小可读资料走轻量路径；提取困难或未结构化长 PDF 走深加工。
- 案件资料 `case`：短且可读走轻量路径；提取困难或未结构化长 PDF 走深加工。
- 参考资料 `reference`：保留原文页及基础索引，不自动建立复杂层级。
- 派生资料 `derived`：返回 `archive`，不自动混入原始证据库。
- 未知用途：返回 `deferred`，保留原件，等待明确分类。
- 自动识别现有 JSONL 法条的非空法名、条号和正文结构；显式派生标记优先。
- 不从 PDF 文件名或标题自动猜测用途，调用者可声明 `material-kind`。
- 缺来源或时间的记录带 `review_flags`；保留历史法条版本，不修改正文。
- JSONL 流式分流报告保存源路径、物理行号、记录哈希、类别、路径、规则 ID 和规则版本，不复制整份正文。
- 报告使用临时文件和原子发布；拒绝输出覆盖原语料。
- PDF 分流先判断用途；未知或派生资料不启动 PDF 解析或图引擎。
- 原生 PDF 检查识别空文本页及替换字符；可显式调用 `pdf-inspector` 的 detect-only 接口。
- 检查器页码使用 1 起始物理页；检查结果与原生检查取并集。
- 轻量路径只导入原文页和空目录，不生成层次摘要、不调用摘要模型；仍有向量编码和基础图开销。
- 深加工复用现有目录构建，只有显式要求才生成摘要。
- 需 OCR 或复杂版面尚不能可靠提取时，返回 `pending_extraction` 和页码／检查信息，不先把缺页结果写成完整证据。
- 原 PDF 不修改、不移动；新导入元数据保存源路径、原文件哈希、文档 ID 和物理页码。
- 当前没有接入 OCR、PaddleOCR、复杂表格提取、自动复核队列或自动重试。
- 原 `import-pageindex` 保留兼容，不自动套用新分流策略。

## 9. 语义知识图、增量记忆和摘要

实现位置：[graph](../dataset/graph.py)、[IncrementalMemoryManager](../dataset/IncrementalMemoryManager.py)、[桥接器](../dataset/memory_graph_bridge.py)、[摘要依赖](../dataset/summary_tree.py)。

- BGE-M3 输出归一化 dense 向量及 sparse 词权重，FAISS 存向量，igraph 存节点、语义边和元数据。
- 节点整数 ID 与 igraph 字符串名之间有统一转换契约。
- 混合相似度结合稠密余弦和稀疏词权重内积；无有效稀疏信号时使用纯余弦。
- 支持初始批量建图、预计算向量、分批近邻搜索及相似度阈值建边。
- 增量添加 Raw／Summary 等节点，择优连边，极高相似节点标记 `similar_to`；该标记不表示物理删除原文。
- 增量原文可挂载到语义接近的摘要，向祖先传播脏标记。
- 度数／结构条件触发聚类待处理标记；不是每次插入都立即调用模型重聚类。
- 软删除保留 tombstone，检索过滤，传播脏摘要状态并更新索引。
- 摘要节点保留来源子节点关联，支持摘要生成和脏摘要重算。
- 层次摘要按依赖后序生成：先子后父；父摘要输入包括子摘要及未覆盖正文。
- 共享原页对应的多个章节可一起标脏；原文删除后可清除失去来源的摘要。
- 批量摘要更新失败时不发布部分新摘要。
- 文档目录节点按明确目录关系组织，不参与普通语义簇自动挂载。
- 独立增量管理器用 GMM 区分摘要相似度的接收簇，辅以绝对相似度下限。
- 样本过少、得分退化或 GMM 拟合问题时，旧组件有中位数阈值路径；不能把它说成无兜底设计。
- 支持新知识挂载到多个接收簇；无合适簇时创建新簇并挂载首个叶子。
- 夜间重写接口、摘要索引重建接口可显式调用；没有自动夜间调度服务。
- 独立管理器使用进程内 MockGraphDB，未提供摘要函数时有演示用拼接摘要；不是生产语义摘要器。
- MemoryGraphBridge 一次编码同时供增量管理器和图引擎使用，避免同一节点两套向量。
- 桥接器支持冷启动重建、增量注入、脏摘要同步、失败待同步重试、软删除及统计。
- 支持图编码器 LoRA 及 SentenceTransformer 格式适配；载入训练产物仍需匹配实际编码器和空间版本。

## 10. 跨任务长期知识记忆

实现位置：[long_term_memory](../dataset/long_term_memory.py)、[graph](../dataset/graph.py)、[说明](LONG_TERM_MEMORY.md)。

- SQLite 保存 JSON＋zlib 图快照，跨任务、跨进程重启恢复知识。
- 快照包含原文、摘要、元数据、向量、语义边、软删除、脏摘要和聚类待处理标记。
- 按 `project_id` 隔离命名空间，保存知识 revision。
- 保存采用版本比较，旧写者抛 `MemoryConflict`，不覆盖新提交版本。
- 批量写入通过 `memory_transaction` 统一提交；异常恢复暂存图并报告错误。
- 载入使用已有向量重建索引，不自动重新编码或调用摘要模型。
- 检索刷新已提交知识；持久运行则使用自己的固定快照，不随中途入库改变证据。
- 本地保存失败保留未提交状态，后续刷新不能静默丢弃；显式舍弃后才能重新读取。
- 校验模型身份、维度、图参数和 `embedding_space_id`；不兼容时拒绝自动迁移。
- 更换模型／LoRA 时由调用者选择新空间标识和库或命名空间，并重新入库。
- 保存最新项目快照，不提供完整的项目知识版本历史；运行固定快照是另一套用途。
- 已同步图内容持久化；增量管理器尚未同步的草稿簇／队列不属于持久快照。
- 不把回答或模型推测自动写成原文事实；不复制原始 PDF 二进制文件。
- 显式提供知识库检查、文档列表、目录读取、页读取和文档导入 CLI。
- 适用于同机持久磁盘；没有多机共享存储、跨机分布式事务或共享网络盘部署保证。

## 11. 文档目录、导航和引用

实现位置：[document_index](../dataset/document_index.py)、[page_index_memory](../dataset/page_index_memory.py)、[document_store](../dataset/document_store.py)、[document_navigation](../dataset/document_navigation.py)、[citations](../dataset/citations.py)。

### 11.1 结构构建和导入

- 接受已有 PageIndex 风格根列表或 `structure` JSON 树，也可直接从原生文本 PDF 本地建树。
- PyMuPDF 书签作为主框架，编号及字体标题检测补充结构；校验标题、页码、顺序和层级。
- 记录 `bookmarks`／`detected`／`hybrid`／`flat`／外部导入来源及 diagnostics。
- 无可靠标题时保留平面结构；不假造复杂目录。
- 节点保留章节 ID、标题、页范围、子节点、摘要和 `key_items`。
- 检查重复章节 ID、页码倒置／越界、子章节超出父范围。
- 原页映射 Raw 证据，目录映射 Summary 导航节点，通过 `source_node_ids` 关联。
- 同一文档 ID 的相同内容／结构／元数据重复导入保持幂等；内容变化要求新版本 ID。
- 缺少摘要时可显式注入或调用摘要函数；不启用生成时保留已有摘要或标题导航。
- 导入的编码或摘要生成失败时回滚整个导入事务。
- 结构成本诊断考虑路由和子节点未覆盖页，给出最坏分支页成本及合并建议。
- 成本诊断不修改树；没有自动 Merge／Expand，也不是实测时延或 token 成本优化器。

### 11.2 只读文档服务

- `DocumentReader.from_file` 只读打开知识库，不创建缺失库，不加载 BGE／FAISS／LLM。
- `from_engine` 为当前运行建立一致读取视图。
- 支持列出文档、读取结构、按版本读取指定物理页。
- 文档版本来自内容指纹；错误文档、版本、越界页和已删除页明确报错。
- 目录支持按层和偏移分页，不允许模型以工具参数读取任意宿主机路径。

### 11.3 按需检索循环

- 五个动作：`list_documents`、`read_structure`、`read_pages`、`vector_search`、`finish`。
- 自动模式：预算内完整短文档直接读；长文档或混合资料走模型导航；无版本文档继续向量检索。
- 可显式选择向量基线或 Agent 导航。
- 独立限制模型轮数、原文记录／页数、原文 tokens、目录／清单 tokens。
- 重复读取不重复计页，空白页计页；超限请求整批拒绝，不截断原页冒充完整内容。
- 记录动作轨迹、模型调用数、读取量及 `empty`、`turn_budget_exhausted` 等状态。
- 轮数耗尽只返回先前成功读取的原文；非法动作、错误版本和模型失败上报。
- 导航与向量工具最终输出原文，再交给 Grader 或因果证据组件。

### 11.4 原文引用

- 为本次实际读取的证据构造 E1／E2 标识表，保留文档、版本、页码、记录和块 ID。
- 模型输出答案、引用 ID 和连续原文引句。
- 校验引用 ID 已读且存在、正文确实使用、无重复／漏配／未使用引用，以及引句确实存在于原文。
- 渲染普通文本来源编号、原文件名、物理页码和引句，同时返回结构化 `citations`。
- 普通证据与因果证据使用统一引用表；摘要不是可引证原文。
- 该检查证明来源身份与引句存在，不证明引句语义足以支持整段答案。
- 借鉴 PageIndex 思路，未安装其 SDK、复制其实现或依赖其云服务。

## 12. CHAIN 因果证据和权重

实现位置：[causal_evidence](../causal_evidence/)、[校准 CLI](../tools/calibrate_causal_support.py)、[方法说明](CAUSAL_EVIDENCE.md)。

### 12.1 证据分块和结构化提取

- `EvidenceRecord` 保存文本、记录 ID、来源 ID、时间、核实可用性、相似度及位置。
- 默认基于 `cl100k_base` 分成 512-token 块、64-token 重叠，记录全篇 token／字符位置。
- 多字节 Unicode 边界向内对齐，避免分块引入损坏字符；实际重叠可能略少于 64。
- 每块调用 LLM 提取实体、事实命题和因果关系；只接受 `causes`／`enables`／`prevents`。
- 因果关系端点须关联已提取实体，保留原始支持分、证据引句和所属命题。
- 原文实体、引用及命题关系先程序校验，再对合法候选进行独立 LLM 文本支持验证。
- 验证区分明确因果、相关、否定、假设及不受支持关系；拒绝项保留字段和理由。
- 实体只做 NFKC、大小写及空白规范化，不自动猜测同义词。
- 程序组装实体集合、事实超边、有向因果边；没有 GNN 训练。
- 图保存来源块、原文偏移、文档版本／页码和拒绝记录。

### 12.2 时间边界、CTVF 和链权

- 从明确日期解析日／月时间；月份用月末，无法唯一判断的日期按缺失处理。
- 已知时间或已知可用日期不早于截止日的证据不进入后续可用证据图。
- 缺事件时间时，需要外部核实的 `availability_verified=true` 及截止日前 `available_at`；模型不能自行赋予资格。
- 先筛选可用图，再从显式目标沿反向因果边 BFS 计算最短距离。
- 依据距截止日天数和因果距离计算时间衰减。
- 依据同一规范化事实的来源数量和已有相似度计算重复支持加分；缺相似度不补造加分。
- 合并时间项与支持项，截断有效性到 1；缺失时间采用显式配置有效性。
- 同来源同事实的重叠记录不重复增加端点支持。
- 根据端点平均有效性及距离得到边调整因子，与原始或校准支持度相乘。
- 枚举以目标结束、不重复节点的有向链，累乘边权；`prevents` 改变链极性。
- 搜索受最大深度、扩展次数、返回链数和最低得分约束；返回搜索截断状态。
- 无目标锚点或无可用证据明确返回状态；无截止日可一般分析，但不计算预测意义的时间权重。
- 分数语义是文本支持度与时间有效性，不是真实因果效应、发生概率或法律效力。

### 12.3 支持度校准

- 保留 `raw_support`；人工标注拟合等距回归后才提供 `calibrated_support`。
- 没有校准器时明确 `uncalibrated`／空校准值；校准结果允许降到 0。
- 校准样本包含来源 ID、原始支持度和二值文本支持标签。
- 按来源组确定训练／留出集，避免重叠块、改写和转载跨集合泄漏。
- 验证来源数量、两类标签及分数变化，不用留出标签重新拟合。
- 输出校准曲线、标签哈希、来源划分、提取器身份、标注可用日期。
- 报告留出原始／校准 Brier 和分箱 ECE。
- 提取／验证模型配置变化时拒绝误用旧校准；预测截止日必须晚于标注可用日期。
- 同步和统一 Kafka 主流程共用因果工具；独立验证请求不是数学因果证明。
- 本项目只采用 CHAIN 图构建及数据权重，没有接入其 Noisy-OR 或完整预测融合。

## 13. 确定性计算与代码执行

实现位置：[calculator](../agent_tools/calculator.py)、[节点](../legal_agent/nodes.py)、[sandbox](../agent_tools/sandbox/)、[后端说明](SANDBOX_BACKENDS.md)。

### 13.1 默认结构化金额计算

- 模型生成有限输入、公式、步骤、结果变量和假设；程序执行加减乘除。
- 计算计划校验有限数值、大小范围、变量名、步骤数量、前序引用和重复赋值。
- 使用 Decimal 和固定精度，金额按 HALF_UP 保留两位小数。
- 运算错误明确失败；计算结果、公式与假设保留在报告中。
- 计算作为可重放工具进入台账；程序核算后再生成最终报告。
- 只验证算式执行，不独立证明法律适用性或案件输入真实性。

### 13.2 可选代码路径和 Docker

- `allow_code=true` 才启用统一运行时的模型生成 Python 代码路径。
- 执行异常传回代码生成器作有界修改；配置错误或后端不可用不当成代码缺陷重写。
- 重试耗尽保存计算失败并终止，不以失败文字冒充计算成功。
- 统一返回计算结果、错误、会话句柄、执行后端和相关错误／清理状态。
- 默认 Docker：断网、非 root、只读根文件系统、CPU／内存／PID 限额、输出上限。
- 通过 Docker exec 传递代码，默认在线工具不开放沙箱 HTTP 服务端口。
- 每次执行新 Python 进程；优先读取 `result`，否则使用 stdout；不保留跨计算变量状态。
- 管理沙箱创建、执行、超时、销毁、空闲和孤儿清理；清理函数需显式调用或走已有生命周期。
- 执行后及运行结束尝试清理，失败保留句柄和错误线索，不假称资源已释放。
- `run_in_process` 仅是显式注入的可信离线测试辅助，不是隔离环境或在线自动兜底。
- 兼容 FastAPI 沙箱服务与旧包装文件仍存在，不是默认 Docker exec 主路径。

### 13.3 可选 SPACE

- 默认关闭，适配固定上游版本的 native `/v1` M1 一次性 Python 执行子集。
- 支持创建沙箱、查询 operation、等待沙箱和进程服务就绪、创建命令、SSE 读取、删除并等待删除完成。
- 管理 Bearer Token 与事件流 Session Token 分离；日志与请求意图不保存秘密凭据。
- 发送创建／命令／删除前持久保存请求方法、路径、请求体、幂等键及关联 ID。
- 对创建响应丢失等不确定结果提供意图核对；超出允许重放条件不盲目重发。
- 校验不可变模板摘要引用、资源单位、端点、超时、幂等窗口和部署网络隔离核对记录。
- 独立连接、空闲读取、操作阶段和总 deadline，以及请求、帧、流和输出上限。
- SSE 支持分片、多行 data、CRLF 和心跳；核对命令进程身份及正常退出事件。
- 必须收到正确进程的退出事件、零退出码及合法 runner JSON 才判执行成功。
- 断流、错误帧、resync、缺退出事件、异常退出或非法结果明确失败。
- 一次性句柄不用于第二条命令；恢复的缓存命中不初始化远程后端。
- 网络及虚拟机隔离依赖真实部署和模板保证；适配器本身不构成隔离证明。
- 无虚拟机暂停恢复、快照、预热池、文件传输、跨任务工作区或 E2B 兼容实现。

## 14. 模型调用、本地推理和缓存

实现位置：[llm](../legal_agent/llm.py)、[vllm_engine](../model/utils/vllm_engine.py)、[语义缓存](../legal_agent/cache.py)、[缓存存储](../multiple-search/SemanticCache/engine.py)。

- 统一 OpenAI 兼容接口；默认 API 后端，支持配置地址、凭据、模型和角色覆盖。
- 使用 ContextVar 记录角色，避免并行任务切换全局角色污染。
- 提示模板集中在 `prompts.py`，消息构建函数接收运行事实，不复制多份静态模板。
- JSON 请求做结构解析，空内容和非法 JSON 明确报错。
- 有界模型请求超时及可重试传输错误重试；每次尝试扣除模型预算。
- 运行中请求超时受剩余 deadline 限制；调用结果进入可重放工具台账。
- 默认主接口 `api`／`auto` 使用 API；显式 `vllm` 失败上报，不静默转 API。
- vLLM 包装提供前缀缓存、温度和输出长度设置，当前固定最大模型长度 8192。
- 一个基座挂载多个同基座 LoRA，角色映射选择适配器，稳定适配器 ID，未知映射报错。
- 当前没有不同基座自动切换、4bit 推理配置透传、CPU 卸载或张量并行配置。
- 独立旧 `create_llm_backend` 工厂仍有自动检测及失败退回 API 行为；它与当前主接口策略不同。
- 独立语义缓存支持内存向量存储或 RedisVL，按向量相似度查答案并写回问答。
- 旧路由包装器可在路由前命中缓存；启用长期记忆或 CHAIN 时绕过该旧缓存读写，避免缺版本／时间上下文。
- 旧路由保留规则层、嵌入语义分类层和 LLM 回退层；默认持久运行使用共享免模型前置分类。
- 不把 vLLM 前缀缓存、旧答案缓存、知识长期记忆和运行检查点视为同一种缓存。

## 15. PA-Tool 工具 Schema 对齐

实现位置：[schema_alignment](../agent_tools/schema_alignment.py)、[CLI](../model/tool_schema.py)、[说明](TOOL_SCHEMA_ALIGNMENT.md)。

- 可拔插配置，默认 `profiles: {}`，不开启时保留现有命名与行为。
- 覆盖 Replanner 的两个引擎名和四个字段，以及导航的五个动作和六个参数。
- 用目标模型温度 0 生成参考名称，再按温度和数量采样候选；支持注入调用回调。
- 使用精确字符 Levenshtein 距离和候选频数计算 peakedness。
- 距离阈值按最大候选长度乘 alpha；聚集分平分时按距参考名称远近排序。
- 名称争用按优先级锁定，其他组件选后续候选；与保留字段冲突或候选耗尽明确报错。
- 输出绑定角色、后端、基座和适配器配置的 profile，保存采样和排序记录。
- 可从已保存候选构建 profile，也可显式从本地目标 vLLM 模型采样。
- 生成入口要求显式后端及现有本地模型，不自动下载或训练。
- 静态说明和 JSON Schema 同步改名，包括字段、required、选择器枚举和默认值。
- 运行结果可逆解码为项目原始字段及动作，再使用既有契约和执行器。
- 案情、原文、目录、日志及自由文本值不做全局替换；不使用模糊名称纠错执行。
- profile 内容参与运行配置哈希；移除对应角色配置即可拔除。
- 模型配置身份不是完整权重指纹；更换目标模型后需重新采样验收。
- 没有真实目标模型映射采样或工具成功率提升结论；旧 Redis 阶段 Replanner 未扩展。

## 16. 本地模型行为优化组件

实现位置：[behavior](../model/behavior.py)、[依赖](../requirements-behavior.txt)、[说明](MODEL_BEHAVIOR.md)。

- 复用固定 Heretic 2.0.0 的 KeywordRate、KLDivergence 和 ARA，作为独立离线可选组件。
- 目标支持减少指定迎合表达、减少指定空泛套话或两者；支持替换关键词表。
- 准备完整本地模型及 good／bad／evaluation 提示文件，验证输入和新输出目录。
- 默认只生成本地提示快照、配置和 manifest，不加载重模型或启动外部程序。
- 显式 `--run` 才调用指定独立 Python，核对上游版本并强制离线获取模型。
- 独立设置子进程缓存、临时目录、检查点、搜索次数、种子和输出长度。
- 导出单独候选 LoRA，不覆盖原模型、不自动上传、不自动挂载。
- 保存提示／配置／模型 config 哈希及执行状态；导出仍标记质量未验证。
- 使用现有角色适配器配置接入和拔除；一次请求选一个 LoRA，不能自动叠加现有任务 LoRA。
- 不处理最终 Generator API，不采用默认拒答抑制目标。
- 关键词出现率和 KL 约束不等于理性、真实性或任务成功率提升；没有实际权重优化实验。

## 17. 合成数据与旧数据生成器

实现位置：[synthetic](../model/data/synthetic/)、[Evol-Instruct](../model/data/evol_instruct.py)、[说明](SYNTHETIC_DATA.md)。

### 17.1 独立合成数据组件

- 默认离线、不加载模型、不调用 API、不启动训练、不修改在线证据库。
- 六类任务：查询、组合汇总、依赖链、并行分支与角色交接、临时工具故障恢复、信息／权限／预算约束。
- 模拟数值与状态环境，程序生成可验证目标；写操作限于当前任务内存状态。
- 配置任务混合权重、组数、组内变体、记录数量及依赖复杂度。
- 复杂度与变体数量使用有上限的幂律采样；不把它冒称真实业务重尾分布。
- 可加入文档内复述、过期版本、低优先级冲突及同级未决冲突，变换后重算目标。
- 来源优先级和 active 标记是显式模拟规则，不等于现实法律效力。
- 干扰工具、工具名称变化、JSON 字段变化和问题改写用于有限结构泛化测试。
- 先按稳定组 ID 切分 train／validation／test，再生成变体；OOD 使用独立实体组。
- 用户提供种子资料可生成待审候选，共享来源 ID 或去首尾空白后相同的文本不能跨来源组。
- 输入给求解者的公开部分与完整环境／标签分开，避免把答案字段整体喂入模型。
- 数值／状态任务发布前实际回放参考轨迹，验证只限模拟规则。
- 法律、因果和开放推理候选进入 `pending`，可附 proposed answer，但不自动晋升为 verified。
- 插件 `CandidateProvider` 接收请求及 Schema 回调；只有显式注入且请求额度大于零才调用。
- provider 错误直接上报，不自行重试或换模型；额度耗尽后的离线模板有明确来源标记。
- 可选输出程序参考、有效但冗余、伪造观测、非法工具和错误答案轨迹。
- 轨迹是 `scripted_reference`，不是模型 rollout 或真实 Agent 运行日志。
- 分别计算答案正确、步骤有效、最终状态正确，并记录固定调用成本惩罚与 reward version。
- 只为同任务非平分候选生成排序对，没有实现评分模型损失或自动 RL 训练。
- 按 split 输出 verified／pending，按需输出 trajectories／ranking，另有组曝光统计。
- 曝光统计区分环境来源出现和参考轨迹实际读取，不冒充训练 token 曝光。
- manifest 保存配置、运行版本、代码哈希、文件哈希、实际分布和 provider 请求次数。
- 按组生成、流式写出，临时目录完整生成后再发布，禁止覆盖既有输出。
- 所有生成任务保留 synthetic 标记，种子资料另标来源，不把合成内容包装成实际证据。

### 17.2 旧 Evol-Instruct 组件

- 根据法条锚点请求模型生成案件问题、双层 Planner 蓝图、事实和计算代码候选。
- 从判决文本生成口语化、高噪声或缺信息问题。
- 从案件集合生成失败与重规划场景，保存 JSONL。
- 使用集中提示模板及 AsyncOpenAI；部分批量循环仍顺序请求，不是自动高并发生成服务。
- 名称中的 Ground Truth 不代表已有独立程序或法律审核保证；没有自动接入新组件的 verified 数据集。
- 该旧模块顶层创建客户端，部分请求失败会记录后跳过或返回空列表，不能套用新组件的失败契约。

## 18. 训练代码

实现位置：[训练目录](../model/training/)、[QLoRA 加载器](../model/utils/unsloth_loader.py)、[训练依赖](../requirements-train.txt)。

| 训练组件 | 现有实现 | 输入和产物 | 当前限制 |
|---|---|---|---|
| Planner | Qwen2.5-14B，QLoRA SFT；Semantic DTW 样本筛选 | `planner_trajectories.jsonl`；Planner LoRA | DTW 是语义相似代理，不是独立任务完成证明；筛选存在旧异常兜底 |
| Replanner | Qwen2.5-14B，GRPO；8→4→2 候选评分 | `replanner_scenarios.jsonl`；Replanner LoRA | 使用模拟检索、相关性及关键词／虫洞奖励，没有跑真实完整任务验证 |
| Extractor／Grader | Qwen2.5-3B，chosen／rejected DPO | `extractor_pairs.jsonl`；DPO LoRA | 自然语言偏好样本不等于已对齐当前结构化 Grade 契约 |
| Reasoner | Qwen2.5-7B，长上下文 IRAC SFT | `reasoner_qa.jsonl`；SFT LoRA | 现有脚本和示例答案不构成法律正确性验收 |
| Retriever | BGE-M3，q／v LoRA，MultipleNegativesRankingLoss | `retriever_triplets.jsonl`；SentenceTransformer 产物 | 模块名、模型格式及当前依赖组合需实际训练验证 |

- 加载器默认 4bit，支持模型覆盖、序列长度覆盖、梯度检查点和 Chat Template 格式化。
- LoRA 覆盖 q／k／v／o／gate／up／down；各角色有独立 rank、alpha 和上下文配置。
- 适配器和 tokenizer 单独保存，不必为同基座角色保存重复完整模型。
- 训练脚本包含批量、梯度累积、epoch、学习率、日志和检查点设置；部分实际值写在脚本中，不全由 YAML 控制。
- Planner 使用嵌入相似度矩阵和 DTW 对齐计划与关键事实，兼容双层／旧扁平计划解析。
- Replanner 评分器可使用 CrossEncoder，缺依赖时旧代码使用词重合；最终奖励不是外部强裁判完整执行结果。
- 部分训练脚本数据文件缺失时使用内置示例；不能据此认定训练数据已经就绪。
- 这些训练脚本部分有顶层加载和训练行为，不应作为服务模块导入；`model/Unsloth.py` 只演示加载基座并挂载 LoRA，不启动训练。
- 当前没有完成实际训练、GPU 容量／速度验收或训练后业务收益验证。
- 没有 FSDP／ZeRO 分片、CPU／NVMe 训练卸载、自动顺序训练调度或自动多基座部署实现。

## 19. 检索、轨迹和因果评测

实现位置：[benchmark](../benchmark.py)、[benchmark_causal](../benchmark_causal/)。

### 19.1 通用检索与轨迹指标

- 可注入检索器和 Agent runner，使用统一样本、召回结果和轨迹数据结构。
- 每条查询取一次最大 K 结果，计算 HR@K、首个相关结果的 MRR 和噪声占比。
- Graph-NIAH 统计多跳证据节点覆盖率及整条所需节点全部召回的比例；不是证明实际沿图走通路径。
- 用模型裁判评分计划、执行步骤、重试及最终答案，汇总轨迹分。
- 提供完整评测入口、示例样本和模拟检索／运行器；需要接入实际 runner 才能评项目效果。
- 旧裁判调用失败返回 ERROR，非法数值评分归零；指标存在不代表已有可信业务基准结果。

### 19.2 确定性 SCM 因果沙盘

- 明确定义外生／内生变量、依赖边、结构方程、来源法条和数值域。
- 校验变量、方程、依赖及 DAG，支持拓扑顺序、父子、祖先后代和混杂结构检查。
- 支持观测计算、`do` 干预、反事实、因果链、差异和图描述。
- 内置加班费／夜班津贴、违法解除赔偿场景，覆盖有混杂与无混杂结构。
- 从同一显式场景生成 L1 观测、L2 干预、L3 反事实及无关扰动版本，保留关联来源 ID。
- 参考答案由场景结构方程计算，法条作为显式锚点；方程本身是人工声明的建模假设。
- 法条库读取 JSONL，规范化法名与中文条号，支持多版本、指定日期有效性和引用检查。
- 可从已存在的 LawRefBook 风格 SQLite／Markdown 仓库构建选定法条语料，保存源版本与来源信息，不自动下载。
- 结构化响应解析支持从模型文本提取完整 JSON，验证结论、因果链、引用和金额字段。
- 硬门禁检查格式、法条存在性／时效性、结论关键词／数字及金额容差；失败直接零分。
- 推理软评支持可注入 matcher，默认词法／数值匹配；混杂判断缺失或不适用时不强行计算该项。
- 输出单题得分、理由、分层均值／通过率及同源题组一致性。
- 当前聚合按实际提供的层评分，不强制补齐所有 L1／L2／L3；扰动版同层可能覆盖，不能冒称已完整验证鲁棒性聚合。
- SCM 是离线评测图，没有注册为在线推理工具，不从文本 CHAIN 图直接推导结构方程。

## 20. 异步消息、历史兼容和部署文件

实现位置：[transport](../agent_runtime/transport.py)、[Worker](../agent_runtime/worker.py)、[旧链路](../asynchronization/)。

- 新 Kafka 通知协议包含 version、event_id、run_id 和 epoch，任务内容以运行库为权威。
- 支持持久 outbox 派发、Broker 确认、通知接收与状态核对；结果通知用于按 ID 读取结果。
- 旧 epoch、已终止或不相符通知不能重新取得执行权；不以消息到达次数决定执行次数。
- Worker 消费线程保持轮询，执行线程独立扫描运行库，避免长任务阻塞消费循环。
- 新默认 pending／result topic 与旧五阶段 topic 分离。
- 旧 Planner→Retriever→Grader→Replanner／Reasoner 五阶段消息链保留。
- 旧 Redis StateManager 支持 JSON 状态读写及 TTL，缺失／过期会话报错；不是新运行恢复权威。
- 旧 Kafka 工厂集中 producer／consumer 配置，手动位移提交，使用会话键分区。
- 旧消息隔离写本地 DLQ；写隔离记录失败不能当已成功隔离提交位移。
- 旧 Retriever 保留向量／图扩展、文档导航、来源证据和 CHAIN 适配。
- 旧 Reasoner 保留报告、金额检测、代码生成／沙箱和有限重试；旧 Replanner 保留独立结构化接口及硬规则。
- `runtime.enabled=false` 可显式保留历史查询行为；容器 `AGENT_RUNTIME=0` 选择旧阶段入口用于排空。
- 提供 Worker Dockerfile、entrypoint、旧 Kubernetes 部署示例和 Docker 沙箱镜像文件。
- Kubernetes 示例没有为新单机运行库与知识库提供正确共享存储，不能直接作为新架构多机部署方案。
- 新 Kafka 模式要求 API 与 Worker 使用同机同一份数据库及资源目录；本地模式不依赖 Kafka／Redis。

## 21. 日志、指标、配置和工程验证

实现位置：[observability](../observability.py)、[config_loader](../config_loader.py)、[CI](../tools/ci.py)、[入口审计](../tools/audit_entrypoints.py)。

### 21.1 可观测性

- JSON 或普通文本日志，包含时间、级别、模块、函数、行号、异常及额外字段。
- 线程安全 Counter、Gauge 和 Histogram 样本收集及 count／p50／p95／p99／min／max 快照。
- 独立旧 HTTP 健康服务提供 `/health`、`/ready` 和 JSON `/metrics`；不是 Prometheus 文本导出器。
- 该独立健康服务的 ready 为简单状态，与运行 API 的实际就绪检查不同，不默认等于模型已可用。
- 可复用指数退避装饰器，以及独立主备 API 客户端、调用耗时与失败计数。
- 主备客户端是独立旧组件，默认新 LLM 接口不自动采用其兜底策略。
- 本地 JSONL DeadLetterQueue 支持写入、查看、回放；不是 Kafka DLQ Topic 服务。
- LLMTimer 上下文计时；任务持久事件、工具执行、资源等待和导航 trace 提供运行诊断。

### 21.2 可选性能路径

- 批量编码按字符长度分组，限制每批数量、保留 sparse、归一化 dense 并恢复原始顺序。
- 接入文本向量化、无预计算向量的初始建图和 PDF 原页导入；空输入不加载模型。
- 原单条编码保留，注入单条编码器不被自动变成批量契约。
- 快照索引缓存保存本进程序列化 CPU FAISS 字节，以完整快照及命名空间哈希为键。
- LRU 同时限制条目数和保留字节；冷构建锁避免相同快照重复构建。
- 命中时反序列化为运行私有索引，不共享可变图；超大索引可用但不保留。
- 默认两个性能开关均关闭；不使用 GPU 自定义算子，也没有零拷贝或跨进程缓存。
- 缓存上限不包括活跃任务图／索引及构建临时内存，批量编码不限制全语料输出大小。
- 记录语料编码、批量编码、知识视图、索引缓存墙钟时间，不等于 CUDA 算子计时。
- 尚无针对这些路径的真实数值一致性和性能收益结论。

### 21.3 配置、依赖与工程工具

- YAML 单例配置，支持 `MY_RAG_CONFIG`、环境变量替换及缺省表达式，集中分层读取。
- 配置文件缺失、非法 YAML 或非法顶层结构报错；缺凭据告警，支持显式严格校验。
- `.env.example` 说明环境变量，不自动读取 `.env`。
- 配置覆盖模型、向量、图、记忆、导航、因果、消息、缓存、沙箱、调度、预算相关部署项、日志及训练声明。
- 运行路径检查及 `tools/run.ps1` 约束 D 盘；脚本设置临时文件、包缓存、模型缓存及 UTF-8。
- 依赖分别维护运行、开发、训练、资料检查和行为优化；runtime constraints 不是完整传递依赖锁。
- 打包配置定义包、模块、控制台入口及规则 JSON 包数据；支持 editable 安装和 wheel 构建。
- `.gitignore` 排除凭据、运行数据库、缓存、模型权重和生成训练数据等本地产物。
- 入口审计在隔离子进程检查显式清单中的导入行为；不等于审计所有训练脚本顶层行为。
- CI 五类检查：入口导入、compileall、pytest、离线 demo、跨进程检查点恢复 spike。
- 七个离线 demo：持久运行时、语义图、记忆桥接、增量阈值、LangGraph 全图、因果沙盘、沙箱返回契约。
- spike 用外部派发台账演示 interrupt 前节点重跑、持久检查点、跨进程 resume 与幂等派发。
- 测试辅助提供确定性模型／编码器／消息／沙箱替身以及图状态执行 harness。
- 本轮只核对源码和文档，未重跑业务测试或训练；附录列的是现有测试覆盖入口和历史证据。
- 没有独立既定的 lint／类型检查配置；不把 compileall 等同于类型检查。
- `training_data_guide.py`、旧 README、规则报告和简历提醒是资料／研究记录，不是运行功能。

## 22. 功能边界与未实现事项

以下集中列出容易与现有代码混淆的边界；它们不计入已实现功能：

- 四类图保持不同语义：知识语义图、文档目录关系、文本因果超图、离线 SCM。另有任务 DAG；它不是证据数据库。没有将所有关系合并为同一种图。
- 没有自动将同一原文复制成四份常驻 GPU 数据库；已有 CPU 向量副本、图快照和运行快照仍存在存储成本。
- 没有 GNN 学习因果图、统计识别真实因果效应或完整 CHAIN 概率预测系统。
- 没有 PageIndex 完整 flash 流程、自动成本驱动树展开／合并、全套供应商集成或其云服务。
- 没有 OCR、复杂法律表格／版面完整解析、AnyDoc／Firecrawl 全量接入或自动材料人工复核平台。
- 没有 JEV 决策模型、CASCADE 阈值拟合、Shadow Mode、工作流搜索、MASS 轨迹蒸馏或 Terminal-RSI 训练实现。
- 没有在线自我改写、自动训练发布、训练失败回滚或模型质量自动晋升流水线。
- 没有自定义 Triton／CUDA 算子、TokenSpeed 推理引擎移植或硬件吞吐收益实验。
- 没有训练分片、显存自动流式加载、不同基座热切换或多个 LoRA 自动叠加；这些资源优化仍需实现。
- 没有多租户用户权限、项目级 ACL、分布式存储、多机高可用、严格公平／抢占调度或分布式 exactly-once。
- 没有任意外部写入工具的通用补偿系统；沙箱资源生命周期请求仍是需要记录和核对的外部副作用。
- 没有证明检查点回滚能撤销外部世界操作，也不把新 epoch 当作已发送请求未生效的证明。
- 没有真实模型质量、提取／校准收益、检索收益、训练收益或 GPU 性能的完整验收结论。
- 没有以历史离线测试替代真实 Kafka／Redis／Docker／SPACE 或干净依赖环境部署验收。
- 面试准备提醒单独保存在 [RESUME_REMINDERS](RESUME_REMINDERS.md)，不自动写入简历；本清单不修改该提醒内容。

## 23. 默认开关和预算

来源：[config.yaml](../config.yaml)、[Budgets](../agent_runtime/contracts.py)、[模型接口](../legal_agent/llm.py)。数值是本次源码基线值，不是经过容量或效果调优的推荐值。

| 配置／契约 | 当前值 | 含义 |
|---|---|---|
| `runtime.enabled` | true | 默认使用持久运行服务 |
| `llm.backend` | api | 默认外部 API |
| `llm.role_models` | 空映射 | 默认角色共用配置模型 |
| `embedding.device` | cuda | 原嵌入配置默认 GPU；不能声称当前配置无需 GPU |
| `long_term_memory.enabled` | true | 允许读取项目长期知识 |
| 知识库 | `.memory/rag.sqlite` | 项目知识快照 |
| 运行库 | `.runtime/runs.sqlite` | 任务与执行权威 |
| 检查点库 | 配对 `runs-checkpoints.sqlite` | LangGraph 检查点 |
| `pageindex.enabled`／navigation | true／auto | 自动选择短文直接读或导航 |
| 导航轮数／原文记录数 | 4／8 | 有界工具循环 |
| 原文／目录 tokens | 6000／3000 | 阅读与目录内容预算 |
| `causal_evidence.enabled` | true | 部署允许；是否调用由任务策略决定 |
| 校准文件／默认截止日／目标 | null／null／null | 不预设校准或预测边界 |
| CTVF d_max／eta／缺失时间有效性 | 3／5.0／0.25 | 权重参数 |
| 因果返回链／扩展／最低分 | 200／10000／0.005 | 搜索预算 |
| `allow_code` | false | 默认结构化 Decimal 计算 |
| 沙箱后端／SPACE enabled | docker／false | 代码模式默认 Docker |
| Docker 内存／超时／输出 | 512m／10 秒／65536 字节 | 代码执行资源限制 |
| 沙箱最大重试 | 3 | 有界改写与重试配置 |
| 并发运行／并发任务 | 2／2 | 同机运行和分支额度 |
| 远程／本地模型 slots | 4／1 | 模型调用准入 |
| 编码器／沙箱 slots | 1／1 | 编码和代码执行准入 |
| 等待者／模型缓存条目 | 64／2 | 队列与模型缓存数量上限 |
| 运行默认预算 | 300 秒、40 次模型请求、80 节点步骤、100000 文本 token 上界 | 恢复不重置 |
| 模型超时／额外传输重试 | 30 秒／1 次 | 还受运行剩余时间限制 |
| 最大递归深度／虫洞阈值 | 5／2 | 控制任务反复重试 |
| 批量编码／快照索引缓存 | false／false | 保留原实现为默认 |
| 编码批量／缓存条目／缓存字节 | 32／2／268435456 | 可选性能配置 |
| PA-Tool profiles | 空映射 | 默认不改名 |
| Heretic 行为优化 | 显式命令，默认不运行 | 不自动挂载候选适配器 |

预算的“文本 tokens”是按 UTF-8 输入字节加预留输出构造的保守上界，不等于供应商账单。并发 slots 不是显存容量保证；源码中未消费的配置键也不能单凭名称视为已生效。

## 24. 公开操作入口

以下列出已有操作，不包含本轮执行承诺。命令参数详见各模块 `--help` 及专题文档。

| 入口 | 全部已有子命令／用途 |
|---|---|
| `legal-agent`／`python -m agent_runtime` | run、status、events、cancel、resume、retry、checkpoints、rewind、recover、backup、restore、serve |
| `python -m agent_runtime.worker` | Kafka 通知消费、outbox 和持久任务推进 |
| `tools/run.ps1` | ci、demo、serve、worker |
| `tools/long_term_memory.py` | route-corpus、inspect、documents、structure、pages、cost、import-pageindex、import-document |
| `dataset/chunk.py` | PDF 原生文本分块、预览 |
| `dataset/prepare_corpus.py` | 文本语料扫描、分块、向量化与导出 |
| `tools/calibrate_causal_support.py` | 来源隔离校准拟合、留出指标及校准文件输出 |
| `python -m model.data.synthetic` | 配置、种子、可选轨迹的离线合成导出 |
| `model/data/evol_instruct.py` | 旧模型数据生成示例入口 |
| `python -m model.tool_schema` | 候选采样／读取、工具命名 profile 生成 |
| `python -m model.behavior` | 行为优化配置准备、显式外部运行 |
| `model/training/train_*.py` | 五个独立训练脚本，不由服务启动自动运行 |
| `model/Unsloth.py` | QLoRA 加载与挂载示例，不训练 |
| `benchmark.py` | 可注入检索／轨迹评测与示例 |
| `benchmark_causal/build_corpus.py` | 从已存在法条仓库构建评测语料 |
| `benchmark_causal/run_demo.py` | 确定性 SCM 评测演示 |
| `tools/audit_entrypoints.py` | 导入检查及清单审计 |
| `tools/ci.py` | 五类统一离线工程检查 |
| `spikes/spike_checkpoint_resume.py` | 中断、派发去重、跨进程恢复验证 |

HTTP 全部任务路由：

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/ready` | 就绪检查 |
| POST | `/runs` | 提交运行，返回 202 |
| GET | `/runs` | 项目运行列表 |
| GET | `/runs/{run_id}` | 状态和结果 |
| GET | `/runs/{run_id}/events` | 游标事件读取 |
| POST | `/runs/{run_id}/cancel` | 取消 |
| POST | `/runs/{run_id}/resume` | 补充事实后排队继续 |
| GET | `/runs/{run_id}/checkpoints` | 检查点列表 |
| POST | `/runs/{run_id}/rewind` | 选择检查点回退 |

FastAPI 还提供默认 OpenAPI／交互式接口说明。CLI 的 retry、备份和恢复没有额外对应的任务 HTTP 路由；没有任务 WebSocket、SSE 答案推流或浏览器产品界面。


## 25. 源码逐文件索引

该基线共盘点 124 个非测试 Python 文件。以下列出全部文件及顶层类／非私有函数，包含历史组件、演示和兼容文件；符号存在不代表全部由默认主流程调用。内部私有方法的具体职责已归入前文功能项。

| 文件 | 入口符号／性质 |
|---|---|
| [agent_runtime/__init__.py](../agent_runtime/__init__.py) | 包初始化、导出或命名空间 |
| [agent_runtime/__main__.py](../agent_runtime/__main__.py) | `main` |
| [agent_runtime/api.py](../agent_runtime/api.py) | `ResumeInput`、`RewindInput`、`create_app` |
| [agent_runtime/backup.py](../agent_runtime/backup.py) | `restore_bundle` |
| [agent_runtime/checkpointer.py](../agent_runtime/checkpointer.py) | `FencedSqliteSaver` |
| [agent_runtime/context.py](../agent_runtime/context.py) | `record_effect_intent`、`RunContext`、`task_context`、`tool_call` |
| [agent_runtime/contracts.py](../agent_runtime/contracts.py) | `local_path`、`RunStatus`、`Budgets`、`RunRequest`、`EvidencePolicy`、`TaskSpec`、`Grade`、`RunConflict`、`RunStopped`、`BudgetExceeded`、`OutcomeUnknown` |
| [agent_runtime/demo.py](../agent_runtime/demo.py) | `State`、`workflow`、`main` |
| [agent_runtime/models.py](../agent_runtime/models.py) | `SharedEncoder`、`shared_encoder` |
| [agent_runtime/resources.py](../agent_runtime/resources.py) | `SchedulerSettings`、`settings`、`ResourceManager`、`manager`、`resource`、`accelerator_owner` |
| [agent_runtime/service.py](../agent_runtime/service.py) | `sandbox_policy_identity`、`AgentService` |
| [agent_runtime/store.py](../agent_runtime/store.py) | `dumps`、`RunStore` |
| [agent_runtime/transport.py](../agent_runtime/transport.py) | `Envelope`、`KafkaTransport` |
| [agent_runtime/worker.py](../agent_runtime/worker.py) | `main` |
| [agent_tools/__init__.py](../agent_tools/__init__.py) | 包初始化、导出或命名空间 |
| [agent_tools/calculator.py](../agent_tools/calculator.py) | `Step`、`CalculationPlan`、`calculate` |
| [agent_tools/sandbox/__init__.py](../agent_tools/sandbox/__init__.py) | 包初始化、导出或命名空间 |
| [agent_tools/sandbox/errors.py](../agent_tools/sandbox/errors.py) | `SandboxError`、`SandboxConfigError`、`SandboxUnavailableError`、`SandboxRemoteError`、`SandboxProtocolError`、`SandboxTimeoutError`、`SandboxOutcomeUnknown`、`SandboxCleanupError` |
| [agent_tools/sandbox/executor.py](../agent_tools/sandbox/executor.py) | `run_in_process`、`run_code_once`、`destroy_session` |
| [agent_tools/sandbox/manager.py](../agent_tools/sandbox/manager.py) | `DockerSandboxManager`、`sandbox_backend`、`create_sandbox_manager`、`get_sandbox_manager`、`reset_sandbox_manager` |
| [agent_tools/sandbox/runner.py](../agent_tools/sandbox/runner.py) | `CappedOutput`、`calculate` |
| [agent_tools/sandbox/server.py](../agent_tools/sandbox/server.py) | `CodeRequest`、`execute_code` |
| [agent_tools/sandbox/space_client.py](../agent_tools/sandbox/space_client.py) | `validate_id`、`SpaceConfig`、`HttpResponse`、`UrllibTransport`、`SpaceEventStream`、`SpaceClient` |
| [agent_tools/sandbox/space_manager.py](../agent_tools/sandbox/space_manager.py) | `SpaceSandboxManager` |
| [agent_tools/sandbox/space_runner.py](../agent_tools/sandbox/space_runner.py) | SPACE 客体执行 runner 源码与版本标识 |
| [agent_tools/schema_alignment.py](../agent_tools/schema_alignment.py) | `schema_for`、`model_identity`、`edit_distance`、`rank_candidates`、`sample_names`、`build_profile`、`SchemaAlignment`、`profile_paths`、`for_role`、`profile_hashes` |
| [asynchronization/kafka_utils.py](../asynchronization/kafka_utils.py) | `create_producer`、`create_consumer`、`quarantine_message` |
| [asynchronization/state_manager.py](../asynchronization/state_manager.py) | `StateManager` |
| [asynchronization/workers/grader_worker.py](../asynchronization/workers/grader_worker.py) | `call_grader_llm`、`prepare_grader_evidence`、`main` |
| [asynchronization/workers/planner_worker.py](../asynchronization/workers/planner_worker.py) | `call_planner_llm`、`main` |
| [asynchronization/workers/reasoner_worker.py](../asynchronization/workers/reasoner_worker.py) | `call_reasoner_llm`、`call_generator_llm`、`detect_calculation_needed`、`call_code_generator_llm`、`run_calc_with_sandbox`、`main` |
| [asynchronization/workers/replanner_worker.py](../asynchronization/workers/replanner_worker.py) | `ReplanTask`、`ReplanOutput`、`ReplannerOps`、`main` |
| [asynchronization/workers/retriever_worker.py](../asynchronization/workers/retriever_worker.py) | `build_engine`、`retrieve_docs`、`retrieve_hits`、`main` |
| [benchmark.py](../benchmark.py) | `llm_judge`、`TestSample`、`RetrievalResult`、`TrajectoryLog`、`LegalBenchmark`、`mock_retrieve`、`mock_run_agent`、`generate_test_samples`、`generate_niah_samples` |
| [benchmark_causal/__init__.py](../benchmark_causal/__init__.py) | 包初始化、导出或命名空间 |
| [benchmark_causal/build_corpus.py](../benchmark_causal/build_corpus.py) | `cn_to_int`、`parse_markdown_law`、`build`、`main` |
| [benchmark_causal/gates.py](../benchmark_causal/gates.py) | `normalize_text`、`extract_numbers`、`ReasoningMatcher`、`LexicalMatcher`、`run_hard_gates`、`ScoringWeights`、`score_case`、`aggregate_by_case` |
| [benchmark_causal/generator.py](../benchmark_causal/generator.py) | `CausalScenario`、`CausalCaseGenerator` |
| [benchmark_causal/legal_corpus.py](../benchmark_causal/legal_corpus.py) | `cn_to_int`、`normalize_law_name`、`parse_citation`、`LawArticle`、`CitationCheck`、`LegalCorpus` |
| [benchmark_causal/run_demo.py](../benchmark_causal/run_demo.py) | `main` |
| [benchmark_causal/scenarios.py](../benchmark_causal/scenarios.py) | `scenarios_for`、`scenario_by_id` |
| [benchmark_causal/schemas.py](../benchmark_causal/schemas.py) | `CausalLevel`、`LawRef`、`LegalCausalTestCase`、`ModelResponse`、`GateResult`、`JudgeVerdict`、`CaseScore` |
| [benchmark_causal/scm.py](../benchmark_causal/scm.py) | `SCMValidationError`、`Variable`、`Edge`、`Equation`、`StructuralCausalModel` |
| [benchmark_causal/scm_labor.py](../benchmark_causal/scm_labor.py) | `build_overtime_scm`、`build_unlawful_dismissal_scm`、`get_scm`、`all_article_refs` |
| [causal_evidence/__init__.py](../causal_evidence/__init__.py) | 包初始化、导出或命名空间 |
| [causal_evidence/calibration.py](../causal_evidence/calibration.py) | `SupportLabel`、`SupportCalibrator`、`calibration_metrics`、`fit_calibration` |
| [causal_evidence/chunking.py](../causal_evidence/chunking.py) | `parse_timestamp`、`admissible`、`chunk_record` |
| [causal_evidence/extraction.py](../causal_evidence/extraction.py) | `canonical`、`extract_verified` |
| [causal_evidence/graph.py](../causal_evidence/graph.py) | `build_graph`、`active_graph`、`reverse_distances`、`score_graph` |
| [causal_evidence/models.py](../causal_evidence/models.py) | `Contract`、`SourceLocation`、`EvidenceRecord`、`TextChunk`、`Entity`、`Proposition`、`CausalRelation`、`Extraction`、`Verdict`、`Verification`、`Hyperedge`、`CausalEdge`、`Rejection`、`CausalTemporalGraph` |
| [causal_evidence/pipeline.py](../causal_evidence/pipeline.py) | `EvidenceSettings`、`evidence_from_hit`、`causal_completion`、`analyze_evidence`、`analyze_configured`、`extraction_identity` |
| [config_loader.py](../config_loader.py) | `ConfigError`、`Config` |
| [dataset/IncrementalMemoryManager.py](../dataset/IncrementalMemoryManager.py) | `MockGraphDB`、`DynamicThresholder`、`IncrementalMemoryManager` |
| [dataset/__init__.py](../dataset/__init__.py) | 包初始化、导出或命名空间 |
| [dataset/chunk.py](../dataset/chunk.py) | `extract_text_from_pdf`、`chunk_text` |
| [dataset/citations.py](../dataset/citations.py) | `Citation`、`CitedAnswer`、`evidence_catalog`、`requires_citations`、`generate_cited_answer` |
| [dataset/document_index.py](../dataset/document_index.py) | `build_pdf_tree`、`read_pdf_structure`、`diagnose_cost` |
| [dataset/document_navigation.py](../dataset/document_navigation.py) | `NavigationSettings`、`Action`、`NavigationResult`、`retrieve` |
| [dataset/document_store.py](../dataset/document_store.py) | `DocumentReader` |
| [dataset/encoding.py](../dataset/encoding.py) | `encode_batches` |
| [dataset/graph.py](../dataset/graph.py) | `vname`、`as_num_id`、`LegalDenseGraphBuilder`、`StubEncoder`、`make_offline_engine` |
| [dataset/index_cache.py](../dataset/index_cache.py) | `SnapshotIndexCache` |
| [dataset/long_term_memory.py](../dataset/long_term_memory.py) | `MemoryConflict`、`MemoryStore`、`graph_snapshot`、`rebuild_index`、`restore_graph`、`source_evidence_hits` |
| [dataset/material_routing.py](../dataset/material_routing.py) | `MaterialFacts`、`route_material`、`route_record`、`route_corpus`、`write_routing_report` |
| [dataset/memory_graph_bridge.py](../dataset/memory_graph_bridge.py) | `MemoryGraphBridge` |
| [dataset/page_index_memory.py](../dataset/page_index_memory.py) | `Section`、`import_tree`、`import_pdf` |
| [dataset/pdf_intake.py](../dataset/pdf_intake.py) | `inspect_with_pdf_inspector`、`import_document` |
| [dataset/prepare_corpus.py](../dataset/prepare_corpus.py) | `discover_files`、`build_nodes`、`vectorize`、`vectorize_batch`、`save`、`prepare_corpus`、`main` |
| [dataset/summary_tree.py](../dataset/summary_tree.py) | `postorder`、`render_section`、`summary_dependencies`、`section_inputs` |
| [double_layer_plan.py](../double_layer_plan.py) | `DAGNode`、`SkeletonGraph`、`Concretization`、`DoubleLayerPlan`、`parse_double_layer_plan`、`plan_steps_from_raw` |
| [legal_agent/__init__.py](../legal_agent/__init__.py) | 包初始化、导出或命名空间 |
| [legal_agent/cache.py](../legal_agent/cache.py) | `SemanticCache`、`UnifiedQueryRouter_Soul` |
| [legal_agent/compat.py](../legal_agent/compat.py) | 说明、常量、兼容层或显式脚本；见对应功能章节 |
| [legal_agent/demo.py](../legal_agent/demo.py) | `main` |
| [legal_agent/evidence.py](../legal_agent/evidence.py) | `source_audit` |
| [legal_agent/gateway.py](../legal_agent/gateway.py) | `classify` |
| [legal_agent/llm.py](../legal_agent/llm.py) | `model_role`、`create_client`、`complete` |
| [legal_agent/nodes.py](../legal_agent/nodes.py) | `node_l0_gateway`、`node_planner`、`node_executor`、`node_replanner`、`node_generate`、`node_write_code`、`node_execute_code`、`node_inject_calc_result`、`node_cleanup_sandbox`、`node_await_input` |
| [legal_agent/operators.py](../legal_agent/operators.py) | `LegalLLMBase`、`ExecutionPlan`、`AgenticNodesOperator` |
| [legal_agent/reasoning.py](../legal_agent/reasoning.py) | `Reasoner` |
| [legal_agent/retrieval.py](../legal_agent/retrieval.py) | 说明、常量、兼容层或显式脚本；见对应功能章节 |
| [legal_agent/routing.py](../legal_agent/routing.py) | `UnifiedQueryRouter_Query` |
| [legal_agent/scheduling.py](../legal_agent/scheduling.py) | `tasks`、`validate_plan`、`user_facts_for`、`execute_ready`、`replan_pending` |
| [legal_agent/state.py](../legal_agent/state.py) | `AgentState` |
| [legal_agent/workflow.py](../legal_agent/workflow.py) | `build_plan_replan_agent` |
| [model/Unsloth.py](../model/Unsloth.py) | `main` |
| [model/__init__.py](../model/__init__.py) | 包初始化、导出或命名空间 |
| [model/behavior.py](../model/behavior.py) | `prepare`、`run`、`main` |
| [model/data/__init__.py](../model/data/__init__.py) | 包初始化、导出或命名空间 |
| [model/data/evol_instruct.py](../model/data/evol_instruct.py) | `synthesize_batch`、`generate_noisy_queries`、`generate_replanner_scenarios`、`main` |
| [model/data/synthetic/__init__.py](../model/data/synthetic/__init__.py) | 包初始化、导出或命名空间 |
| [model/data/synthetic/__main__.py](../model/data/synthetic/__main__.py) | `main` |
| [model/data/synthetic/candidates.py](../model/data/synthetic/candidates.py) | `CandidateProvider`、`pending_candidate` |
| [model/data/synthetic/contracts.py](../model/data/synthetic/contracts.py) | `Contract`、`Config`、`Evidence`、`Record`、`Outcome`、`Task`、`Step`、`Trajectory`、`Source`、`SeedGroup`、`Draft` |
| [model/data/synthetic/export.py](../model/data/synthetic/export.py) | `dumps`、`validate_seeds`、`export_dataset` |
| [model/data/synthetic/generation.py](../model/data/synthetic/generation.py) | `split_for`、`weighted_count`、`generate_groups`、`trajectory_candidates` |
| [model/data/synthetic/simulation.py](../model/data/synthetic/simulation.py) | `resolve`、`oracle`、`Environment`、`reference`、`score` |
| [model/tool_schema.py](../model/tool_schema.py) | `main` |
| [model/training/__init__.py](../model/training/__init__.py) | 包初始化、导出或命名空间 |
| [model/training/train_extractor_grader.py](../model/training/train_extractor_grader.py) | 顶层训练脚本，见第 18 节 |
| [model/training/train_meta_planner.py](../model/training/train_meta_planner.py) | `SemanticDTWRewarder` |
| [model/training/train_reasoner.py](../model/training/train_reasoner.py) | 顶层训练脚本，见第 18 节 |
| [model/training/train_replanner_grpo.py](../model/training/train_replanner_grpo.py) | `GRPOTournamentEvaluator`、`reward_func` |
| [model/training/train_retriever.py](../model/training/train_retriever.py) | 顶层训练脚本，见第 18 节 |
| [model/utils/__init__.py](../model/utils/__init__.py) | 包初始化、导出或命名空间 |
| [model/utils/unsloth_loader.py](../model/utils/unsloth_loader.py) | `UnslothLoader`、`load_planner`、`load_replanner`、`load_extractor_grader`、`load_reasoner` |
| [model/utils/vllm_engine.py](../model/utils/vllm_engine.py) | `VLLMPromptCacheEngine`、`create_llm_backend` |
| [multiple-search/SemanticCache/engine.py](../multiple-search/SemanticCache/engine.py) | `InMemoryVectorStore`、`RedisVectorStore`、`build_semantic_cache` |
| [multiple-search/legal_sandbox/example_usage.py](../multiple-search/legal_sandbox/example_usage.py) | `get_manager`、`node_write_code`、`node_execute_code`、`terminate_session` |
| [multiple-search/legal_sandbox/sandbox_exec.py](../multiple-search/legal_sandbox/sandbox_exec.py) | 兼容导出／显式示例入口 |
| [multiple-search/legal_sandbox/sandbox_manager.py](../multiple-search/legal_sandbox/sandbox_manager.py) | 兼容导出／显式示例入口 |
| [multiple-search/legal_sandbox/sandbox_server.py](../multiple-search/legal_sandbox/sandbox_server.py) | 兼容导出／显式示例入口 |
| [multiple-search/soul.py](../multiple-search/soul.py) | 兼容导出／显式示例入口 |
| [observability.py](../observability.py) | `JsonFormatter`、`setup_structured_logging`、`MetricsCollector`、`get_metrics`、`HealthHandler`、`start_health_server`、`retry_with_backoff`、`ResilientLLMClient`、`DeadLetterQueue`、`get_dlq`、`push_to_dlq`、`LLMTimer` |
| [prompts.py](../prompts.py) | `build_meta_planner_messages`、`build_extractor_messages`、`build_grader_messages`、`build_reasoner_messages`、`format_evidence_chain`、`build_generator_messages`、`build_code_generator_messages`、`build_replanner_messages`、`build_router_l2_messages`、`build_benchmark_judge_messages`、`build_causal_extraction_messages`、`build_causal_verification_messages`、`get_prompt`、`list_prompts` |
| [query.py](../query.py) | 兼容导出／显式示例入口 |
| [replanner_rules.py](../replanner_rules.py) | `task_has_kw`、`apply_hard_rules`、`list_rules` |
| [replanner_rules_report.py](../replanner_rules_report.py) | 说明、常量、兼容层或显式脚本；见对应功能章节 |
| [spikes/spike_checkpoint_resume.py](../spikes/spike_checkpoint_resume.py) | `state_path`、`TaskLedger`、`Cycle`、`node_plan`、`node_dispatch`、`node_finish`、`build_graph`、`make_saver`、`Bus`、`phase_suspend`、`phase_resume`、`Check`、`main` |
| [tools/audit_entrypoints.py](../tools/audit_entrypoints.py) | `EntryPoint`、`probe_import`、`error_excerpt`、`main` |
| [tools/calibrate_causal_support.py](../tools/calibrate_causal_support.py) | `main` |
| [tools/ci.py](../tools/ci.py) | `main` |
| [tools/long_term_memory.py](../tools/long_term_memory.py) | `main` |
| [training_data_guide.py](../training_data_guide.py) | 说明、常量、兼容层或显式脚本；见对应功能章节 |

非 Python 的运行、部署和打包文件也纳入盘点：

| 文件 | 用途 |
|---|---|
| [.continue/rules/new-rule.md](../.continue/rules/new-rule.md) | 编辑器规则占位文件，不是业务实现 |
| [.env.example](../.env.example) | 环境变量声明，不含实际凭据 |
| [.gitattributes](../.gitattributes) | 文本与 Git 属性 |
| [.gitignore](../.gitignore) | 运行和敏感产物排除 |
| [MANIFEST.in](../MANIFEST.in) | 发行包清单 |
| [asynchronization/Dockerfile.worker](../asynchronization/Dockerfile.worker) | Worker 镜像构建声明 |
| [asynchronization/entrypoint.sh](../asynchronization/entrypoint.sh) | 新运行时／旧阶段 Worker 显式选择 |
| [asynchronization/explain](../asynchronization/explain) | 旧异步链说明 |
| [asynchronization/k8s_worker_deployment.yaml](../asynchronization/k8s_worker_deployment.yaml) | 旧阶段链 Kubernetes 示例 |
| [benchmark_causal/data/legal_corpus.jsonl](../benchmark_causal/data/legal_corpus.jsonl) | 版本化法条评测语料 |
| [config.yaml](../config.yaml) | 部署配置真源 |
| [dataset/material_rules.json](../dataset/material_rules.json) | 声明式加工深度规则 |
| [multiple-search/legal_sandbox/Dockerfile](../multiple-search/legal_sandbox/Dockerfile) | 代码沙箱镜像构建声明 |
| [pyproject.toml](../pyproject.toml) | 包、入口和依赖声明 |
| [pytest.ini](../pytest.ini) | 测试发现与 pytest 选项 |
| [requirements-behavior.txt](../requirements-behavior.txt) | 可选固定版本行为优化依赖 |
| [requirements-dev.txt](../requirements-dev.txt) | 开发和测试依赖 |
| [requirements-ingestion.txt](../requirements-ingestion.txt) | 资料规则及可选 PDF 检查依赖 |
| [requirements-runtime.constraints.txt](../requirements-runtime.constraints.txt) | 已有直接运行依赖约束 |
| [requirements-train.txt](../requirements-train.txt) | 训练及本地推理依赖 |
| [requirements.txt](../requirements.txt) | 运行依赖 |
| [tools/run.ps1](../tools/run.ps1) | D 盘运行与统一命令包装 |

## 26. 测试与验证证据索引

以下为全部 33 个测试目录 Python 文件；测试函数数量不是 pytest 参数化后的用例数量，本轮未执行它们。

| 测试文件 | 测试函数定义数 |
|---|---:|
| [tests/conftest.py](../tests/conftest.py) | 0 |
| [tests/harness.py](../tests/harness.py) | 0 |
| [tests/test_agent_runtime.py](../tests/test_agent_runtime.py) | 8 |
| [tests/test_agent_tools.py](../tests/test_agent_tools.py) | 8 |
| [tests/test_causal_corpus.py](../tests/test_causal_corpus.py) | 15 |
| [tests/test_causal_evidence.py](../tests/test_causal_evidence.py) | 29 |
| [tests/test_causal_gates.py](../tests/test_causal_gates.py) | 24 |
| [tests/test_causal_generator.py](../tests/test_causal_generator.py) | 17 |
| [tests/test_causal_pipeline.py](../tests/test_causal_pipeline.py) | 15 |
| [tests/test_causal_scm.py](../tests/test_causal_scm.py) | 22 |
| [tests/test_chain_integration.py](../tests/test_chain_integration.py) | 12 |
| [tests/test_corpus_ingest.py](../tests/test_corpus_ingest.py) | 25 |
| [tests/test_double_layer_plan.py](../tests/test_double_layer_plan.py) | 25 |
| [tests/test_e2e_graph.py](../tests/test_e2e_graph.py) | 19 |
| [tests/test_graph_vertex_names.py](../tests/test_graph_vertex_names.py) | 18 |
| [tests/test_long_term_memory.py](../tests/test_long_term_memory.py) | 29 |
| [tests/test_material_routing.py](../tests/test_material_routing.py) | 14 |
| [tests/test_memory_graph_bridge.py](../tests/test_memory_graph_bridge.py) | 31 |
| [tests/test_memory_thresholds.py](../tests/test_memory_thresholds.py) | 21 |
| [tests/test_model_behavior.py](../tests/test_model_behavior.py) | 11 |
| [tests/test_module_boundaries.py](../tests/test_module_boundaries.py) | 2 |
| [tests/test_pageindex.py](../tests/test_pageindex.py) | 10 |
| [tests/test_pageindex_navigation.py](../tests/test_pageindex_navigation.py) | 16 |
| [tests/test_pageindex_structure.py](../tests/test_pageindex_structure.py) | 12 |
| [tests/test_parallel_agent.py](../tests/test_parallel_agent.py) | 23 |
| [tests/test_replanner_rules.py](../tests/test_replanner_rules.py) | 18 |
| [tests/test_runtime_api_transport.py](../tests/test_runtime_api_transport.py) | 4 |
| [tests/test_runtime_recovery.py](../tests/test_runtime_recovery.py) | 4 |
| [tests/test_runtime_store.py](../tests/test_runtime_store.py) | 12 |
| [tests/test_sandbox.py](../tests/test_sandbox.py) | 36 |
| [tests/test_schema_alignment.py](../tests/test_schema_alignment.py) | 12 |
| [tests/test_space_sandbox.py](../tests/test_space_sandbox.py) | 42 |
| [tests/test_synthetic_component.py](../tests/test_synthetic_component.py) | 13 |

### 专题说明与历史验证记录

以下全部为已存在文件。记录的日期、源码阶段和替身范围决定证据适用性；旧通过记录不能自动推广到新代码或真实服务。

- [docs/AGENT_RUNTIME.md](AGENT_RUNTIME.md)
- [docs/CAUSAL_EVIDENCE.md](CAUSAL_EVIDENCE.md)
- [docs/CONTRACTS.md](CONTRACTS.md)
- [docs/LONG_TERM_MEMORY.md](LONG_TERM_MEMORY.md)
- [docs/MATERIAL_ROUTING.md](MATERIAL_ROUTING.md)
- [docs/MODEL_BEHAVIOR.md](MODEL_BEHAVIOR.md)
- [docs/MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md)
- [docs/PAGEINDEX_IMPLEMENTATION.md](PAGEINDEX_IMPLEMENTATION.md)
- [docs/PAGEINDEX_INTEGRATION_REVIEW.md](PAGEINDEX_INTEGRATION_REVIEW.md)
- [docs/PARALLEL_AGENT.md](PARALLEL_AGENT.md)
- [docs/PERFORMANCE_PATHS.md](PERFORMANCE_PATHS.md)
- [docs/PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)
- [docs/RESUME_REMINDERS.md](RESUME_REMINDERS.md)
- [docs/SANDBOX_BACKENDS.md](SANDBOX_BACKENDS.md)
- [docs/SYNTHETIC_DATA.md](SYNTHETIC_DATA.md)
- [docs/TOOL_SCHEMA_ALIGNMENT.md](TOOL_SCHEMA_ALIGNMENT.md)
- [docs/archive/README_before_modularization.md](archive/README_before_modularization.md)
- [docs/validation/agent_runtime_2026-10-18.txt](validation/agent_runtime_2026-10-18.txt)
- [docs/validation/causal_evidence_ci_2026-10-18.txt](validation/causal_evidence_ci_2026-10-18.txt)
- [docs/validation/chain_active_ci_2026-10-18.txt](validation/chain_active_ci_2026-10-18.txt)
- [docs/validation/long_term_memory_ci_2026-10-18.txt](validation/long_term_memory_ci_2026-10-18.txt)
- [docs/validation/material_routing_2026-10-20.txt](validation/material_routing_2026-10-20.txt)
- [docs/validation/model_behavior_2026-11-04.txt](validation/model_behavior_2026-11-04.txt)
- [docs/validation/pageindex_a_2026-10-18.txt](validation/pageindex_a_2026-10-18.txt)
- [docs/validation/pageindex_b_2026-10-18.txt](validation/pageindex_b_2026-10-18.txt)
- [docs/validation/pageindex_c_2026-10-18.txt](validation/pageindex_c_2026-10-18.txt)
- [docs/validation/parallel_agent_2026-10-18.txt](validation/parallel_agent_2026-10-18.txt)
- [docs/validation/sandbox_space_2026-10-19.txt](validation/sandbox_space_2026-10-19.txt)
- [docs/validation/synthetic_component_2026-10-21.txt](validation/synthetic_component_2026-10-21.txt)
- [docs/validation/tool_schema_alignment_2026-11-07.txt](validation/tool_schema_alignment_2026-11-07.txt)

本次文档盘点使用 Git 跟踪文件清单、Python AST、当前源码、配置及专题文档交叉核对。未扫描或公开凭据、运行数据库正文、模型权重或用户未跟踪的工作区文件。完整性按上述源码基线和功能范围判断，后续实现变化应同步更新清单。
