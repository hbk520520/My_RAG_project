# 项目详细介绍：My_RAG_project — 法律智能体 RAG-plus

> **一句话定位**：一个面向中国劳动争议领域的 **Plan-and-Replan 法律智能体系统**。
> 把「基础 RAG（检索→拼接→回答）」升级为「**规划 → 多跳检索 → 判据 → 重规划 → 推演 → 沙箱计算 → 出意见书**」的可控状态机，
> 并用**自己微调的 4 个 LoRA** 替代通用大模型，配 **Kafka 异步 Worker 集群**扛生产吞吐。

---

## 0. 关键事实速览

| 维度 | 内容 |
|---|---|
| 领域 | 中国劳动法 / 劳动争议（案例库 + 法条） |
| 架构范式 | Plan-and-Replan Agent + Dense Knowledge Graph RAG |
| 检索底座 | **BGE-M3**（1024 维，dense + sparse 双信号）+ **FAISS HNSW**（`M=32`, inner-product）+ **igraph** 语义图 |
| Agent 编排 | **LangGraph StateGraph**（同步链路）+ **Kafka 5 Topic Worker 集群**（异步链路） |
| 状态管理 | **Redis** 状态脱水/复水（TTL 3600s），纯 Python 控制层（不用 LLM 管状态） |
| 模型策略 | **1 个 Qwen2.5 基座 + 4 个 LoRA**（Planner / Replanner / Extractor+Grader / Reasoner），Unsloth 4-bit QLoRA 训练，vLLM 自部署 + prefix caching |
| 数值安全 | **Docker 带状态沙箱**执行 LLM 生成的 Python 计算代码 |
| 数据工厂 | **Evol-Instruct** 反向出题（法条 → 虚构案情 → 双层蓝图 Ground Truth） |
| 评测 | HR@K / MRR / NoiseRatio + 轨迹评分 + Graph-NIAH 深跳探针 |
| 工程化现状 | 187 个离线单元/集成测试全绿；`compileall` 干净；4 个 demo 可离线跑通 |

---

## 1. 整体规划：五条设计哲学

这五条决定了后面所有技术选型，也是理解这个项目的钥匙。

### ① 不用 LLM 当控制层
> README 原话：「**注意 LLM 的本质是概率预测，因此绝对不要用单纯的 LLM 去当 Control-layer！**」

**落地**：所有状态流转、熔断、队列操作都用**纯 Python** 写在 `soul.py` 的节点函数与 `asynchronization/workers/*.py` 里；
LLM 只承担**判断器**角色（"资料够不够"、"这段相关吗"）。`AgentState` 是显式 TypedDict，不是一段自由文本。

### ② 规划与判断解耦
Replanner **只负责"重写"**，不负责"判断够不够" —— 判断全部交给 Grader 节点。
**原因**：小参数量模型同时做两件事会互相干扰；由 Grader 独立判定后，重规划输入更干净。
**落地**：`grader_worker.py` 独立成 Worker；`soul.py` 的 `node_grader` / `node_replanner` 各司其职。

### ③ 双层蓝图 P_q = {S_q, C_q}
Meta-Planner 的输出不是一串任务，而是**双层结构**：

- **S_q（骨架）**：一个 **JSON 格式的 DAG**，节点只写**抽象动作**，**不含具体实体**
  （把「核实张三与 A 公司 2022 年 3 月的劳动关系」抽象为「核实劳动关系」）
- **C_q（具象化）**：把每个抽象节点实例化成针对本案的具体检索查询

**好处**：S_q 可作为**跨案件复用的推理模板**（DTW 对齐、GRPO 奖励都在这一层比较），
C_q 才绑定当前案情。`double_layer_plan.py` 提供 Schema + **环检测** + **拓扑排序** + **并行组识别** + 展平为执行队列。

### ④ 同一基座 + 多 LoRA，而不是多个独立模型
> README 原话：「**不能使用三个完全不同的独立大模型**」

**原因**：三个独立模型会带来 (a) 微调成本三倍、(b) **语义流形不对齐**（不同模型的向量空间不互通）。
**落地**：一个 Qwen2.5 基座 + 4 个 LoRA，统一由 vLLM 承载。（`config.yaml` → `vllm.lora_modules`）

### ⑤ 数值问题绝不让 LLM 直接算
> README 原话：「在计算类似"税率、赔偿金"等强数值问题时，**绝对不能让 LLM 直接生成数字答案**」

**落地**：多一步 —— LLM **生成 Python 计算代码** → 在**完全断网的 Docker 沙箱**里执行 → 结果回注状态。
两条链路（LangGraph 与 Kafka Worker）**共用**同一个入口 `legal_sandbox/sandbox_exec.py`。

---

## 2. 分层架构

```
┌──────────────────────────────────────────────────────────────┐
│ L1 接入/路由      query.py · SemanticCache/engine.py          │
│                   L0 注入网关 → L1 规则/缓存 → L2 LLM 分类     │
├──────────────────────────────────────────────────────────────┤
│ L2 Agent 编排     multiple-search/soul.py (LangGraph FSM)     │
│                   9 节点状态机 + 熔断 + 沙箱闭环                │
├──────────────────────────────────────────────────────────────┤
│ L3 异步 Worker    asynchronization/workers/*.py (5 个)         │
│                   Kafka 5 Topic · Redis 状态脱水 · DLQ         │
├──────────────────────────────────────────────────────────────┤
│ L4 知识图谱       dataset/graph.py (BGE-M3 + FAISS + igraph)   │
│                   IncrementalMemoryManager (GMM) · bridge      │
├──────────────────────────────────────────────────────────────┤
│ L5 模型训练       model/training/*.py · model/data/*.py        │
│                   4 个 LoRA 的 SFT / DPO / GRPO 流水线          │
├──────────────────────────────────────────────────────────────┤
│ 横切能力          prompts.py (13 模板+10 构造器) · config.yaml  │
│                   config_loader.py · observability.py (DLQ)   │
│                   replanner_rules.py · double_layer_plan.py   │
└──────────────────────────────────────────────────────────────┘
```

### 各层职责与关键文件

| 层 | 文件 | 职责 |
|---|---|---|
| **接入/路由** | `query.py` | `UnifiedQueryRouter_Query`：三层漏斗。L0 正则拦截 Prompt 注入；L1 语义缓存（阈值 0.98）+ 长度/语义规则；L2 用 LLM 兜底分类为 `CHITCHAT` / `SIMPLE_QA` / `COMPLEX_TASK` |
| | `multiple-search/SemanticCache/engine.py` | 语义缓存：`redisvl` 向量索引；**未装 redisvl 或连不上 Redis 时自动退回 `InMemoryVectorStore`**，不阻断主流程 |
| **Agent 编排** | `multiple-search/soul.py` | LangGraph 9 节点 FSM：`L0_Gateway → Planner → Executor → Grader → Replanner → Generate → WriteCode → ExecuteCode → InjectResult → Cleanup` |
| **持久化执行** | `agent_runtime/` | 检查点工厂（sqlite/Redis）/ **双表幂等台账** / 任务总线（Kafka 旁路 + 内存实现）/ `DispatchTask`·`AwaitTask` 节点 / 兜底 `recoverer`。解决"挂起等远端结果 + 重启不重复副作用" |
| **异步 Worker** | `asynchronization/workers/` | 5 个独立进程：planner / retriever / grader / replanner / reasoner，靠 Kafka Topic 串联 |
| | `asynchronization/kafka_utils.py` | Topic/Group/acks/compression 全部读配置；`quarantine_message()` 把毒消息写 DLQ 而不中断流水 |
| | `asynchronization/state_manager.py` | Redis 状态脱水（大对象不塞进 Kafka 消息） |
| **知识图谱** | `dataset/graph.py` | 核心图引擎：建图、增量挂载、tombstone 软删 + 脏传播、摘要生成/夜间重算、FAISS 检索 |
| | `dataset/IncrementalMemoryManager.py` | **GMM 动态阈值**决定新知识挂到哪个簇；孤儿节点建新簇 |
| | `dataset/memory_graph_bridge.py` | 双写桥接：一次编码同时喂 GMM 与 igraph，保证两系统 ID/向量一致 |
| | `dataset/chunk.py` | PDF 抽文本（PyMuPDF）→ **规则分块**（按空行/句末切分 + 最小长度合并） |
| | `dataset/prepare_corpus.py` | 语料入库：文本目录 → `nodes.jsonl` + `vectors.npy`（行序严格对齐，ID 强制为 int） |
| **模型训练** | `model/training/*.py` | 4 条训练流水线（见 §6） |
| | `model/data/evol_instruct.py` | Evol-Instruct 数据工厂 |
| | `model/utils/unsloth_loader.py` / `vllm_engine.py` | 4-bit QLoRA 加载器；vLLM 后端 + 自动回退 API |
| **横切** | `prompts.py` | **单一真源**：13 个模板 + 10 个动态构造器 |
| | `config.yaml` / `config_loader.py` | 唯一配置源，支持 `${VAR}` / `${VAR:-默认值}` 环境变量替换 |
| | `observability.py` | 结构化日志、健康检查、Metrics、**DLQ**、指数退避 |
| | `replanner_rules.py` | 7 条硬规则，**两条链路共用** |
| | `double_layer_plan.py` | 双层蓝图 Schema（含 DAG 环检测）|
| | `benchmark.py` | HR@K / MRR / NoiseRatio + 轨迹评分 |
| **契约与验证** | `docs/CONTRACTS.md` | **接口契约冻结**：状态字段 / Kafka 消息 / 路由 / LLM 接缝 / 沙箱返回 / LangGraph 恢复语义 |
| | `tools/ci.py` | 一键回归：入口体检 + compileall + pytest + 离线 demo + P0 spike |
| | `tools/audit_entrypoints.py` | 入口点体检：**导入不得产生副作用** + 离线可跑性矩阵 |
| | `tests/harness.py` | 离线沙盘：可编程假 LLM / 内存 Kafka / 脚本化编码器 / 图装配 |
| | `spikes/spike_checkpoint_resume.py` | Checkpointer go/no-go（6 项验证，含跨进程恢复） |

---

## 3. 端到端数据流（两条链路）

### 链路 A：LangGraph 同步链路（`query.py` → `soul.py`）

```
用户提问
  │
  ├─ L0 Gateway ── 正则拦截 Prompt 注入（中英双语 + 分隔符模式），初始化熔断计数
  ├─ L1 语义缓存 ── RedisVL 向量检索，命中率阈值 0.98
  ├─ L2 路由 ──── L0/L1 未决 → LLM 分类；LLM 失败退回保守启发式
  │
  └─ COMPLEX_TASK → LangGraph StateGraph
        Planner      → 双层蓝图，展平为 task_queue
        Executor     → 向量召回 + 图游走一跳；Extractor 抽原子事实
        Grader       → 判定证据是否 sufficient（纯判断，不重写）
        Replanner    → 不足时：先跑 7 条硬规则；未命中才走 LLM 重规划
        Generate     → 汇总证据链出法律意见书
        WriteCode    → 检测到金额计算需求 → LLM 生成 Python 代码
        ExecuteCode  → Docker 沙箱执行（失败按熔断策略重试）
        Cleanup      → 销毁沙箱容器
```

### 链路 B：Kafka 异步 Worker 链路

```
   topic.planner.pending → [Planner Worker]  → topic.retriever.pending
                                              → [Retriever Worker] → topic.grader.pending
                                              → [Grader Worker]    → topic.reasoner.pending
                                                                     或 topic.replanner.pending
                                              → [Replanner Worker] → topic.retriever.pending
                                              → [Reasoner Worker]  → 出意见书（含沙箱计算明细）
```

- 每个 Worker：`consumer` 逐条处理 + `try/except → quarantine_message() → commit`，**毒消息不中断整条流水**
- 大对象（证据链、历史）存 **Redis**，Kafka 消息只传 `session_id`（**状态脱水**）
- 配置项：`enable_auto_commit: false`（手动提交，保证 at-least-once）、`acks: all`、`gzip` 压缩

---

## 4. 技术选型清单（以及为什么）

| 技术 | 用在哪 | 为什么选它 |
|---|---|---|
| **BGE-M3** | 图引擎编码器 | 同时产出 **dense（1024 维）+ sparse（词权重）**，一套模型拿到两种检索信号，省一个模型 |
| **FAISS HNSW**（`M=32`, `METRIC_INNER_PRODUCT`） | 近似最近邻 | 全局搜索是 `O(N)`，节点一多就延迟爆炸。HNSW 是图近似索引，对数级。**注意 FAISS 不支持余弦，靠"归一化后的内积"等价实现** |
| **igraph** | 语义图存储 | 轻量、C 内核、支持按名字/属性查顶点。承载 parent_id 树 + 语义边 |
| **GaussianMixture (sklearn)** | 增量挂载阈值 | 用一维 GMM 把相似度自动分成 Accept/Reject 两类，**阈值随数据自适应**，比写死阈值稳 |
| **LangGraph** | Agent FSM | 显式 `StateGraph` + 条件边，状态是结构体而非字符串；比手写 while 循环可调试、可熔断 |
| **Kafka** | Worker 编排 | 天然削峰 + 水平扩容；每个 Worker 是独立进程，可单独重启/扩容 |
| **Redis** | 状态存储 + 语义缓存 | 状态脱水；`redisvl` 提供向量索引 |
| **Pydantic v2** | 结构化输出校验 | Replanner 输出、双层蓝图都用 Pydantic Schema 强校验，**拒绝概率性的松散文本** |
| **Docker（Python SDK）** | 计算沙箱 | 带状态容器（变量跨次保留）+ 内存 512MB / CPU 0.5 核 / `network_mode: none` / `no-new-privileges` |
| **PyMuPDF** | PDF 抽取 | 速度快、中文支持好 |
| **Unsloth** | 训练加速 | 4-bit QLoRA 把 14B 压到 <5GB 显存，单卡可训 |
| **vLLM** | 推理加速 | `enable_prefix_caching=True` —— 同一 system prompt 的 KV-cache 只算一次 |
| **pytest** | 测试 | 187 个用例，全部**离线**可跑（`StubEncoder` 绕开模型下载） |

---

## 5. 核心设计决策（含踩坑记录）

这些是 README 设计笔记里最有价值的"经验"，也都是代码里真实落地的：

| 决策 | 内容 |
|---|---|
| **冷启动相似度双阈值** | 混合得分 = `0.3 × cosine + 0.7 × 词权重重叠`。`> 0.85` 才连边（过低的判为不相关），`< 0.99` 才算连边（过高的判为冗余，只打 `similar_to` 标签不连边） |
| **Summary 节点延迟生成** | 摘要节点地位与普通节点**完全平等**，作用是建全局↔局部的语义桥。但**必须等图整体构建完再集中生成** —— 作者当时只有一张卡，穿插做会拖慢写入；顺带把 Summary 的 LLM 成本压到「1/10 数据只花 7 块钱」 |
| **最大值截断 + 降序再聚类** | 一个簇的子节点数超 `max_cluster_size` 时**本次只取前 N 条**，其余留待后续降序聚类，避免插入路径上做重活 |
| **必须批量入库** | 原先是单节点入库，"饭都吃完了进度条还没动" → `build_initial_graph_batch()` 走 batch 流水线 |
| **Tombstone 软删 + 脏传播** | 法条失效不物理删除，标 `status=tombstone` 从检索隐身，同时**沿 parent_id 链向上标脏**，夜间重算只重算受影响的摘要 |
| **防多跳跑题** | 每偶数跳**强制注入原始问题**作为底线，防止长链推理逐渐偏离主题 |
| **每次询问后清上下文** | 防止上下文堆积让模型"变蠢" |
| **熔断双出口** | 成功出口（证据闭环）+ 熔断出口（`recursion_depth > max`，强制降级结案），避免死循环烧 Token |
| **硬规则前置** | Replanner 先跑 7 条领域硬规则（未签合同→双倍工资、工伤、社保断缴、加班费、竞业限制、试用期、劳务派遣），未命中才走 LLM 重规划。**两条链路共用同一份规则表** |
| **向量一致性** | 同一个节点在 GMM 与 igraph 里必须是**同一个向量** —— 桥接器把同一次编码的 dense/sparse 透传给 `add_node()`，并做维度强校验（`_check_dim`） |
| **顶点名类型统一** | igraph 顶点名恒为 `str(int)`，内部逻辑仍用 `int`；边界转换只有 `vname()` / `as_num_id()`。**混用 int/str 会让 `vs.find` 抛 ValueError，而调用点普遍 `except: continue` → 静默漏节点** |

---

## 6. 模型矩阵与训练规划

### （1）实际部署的是 4 个 LoRA，对应 5 个 Worker

| 角色 | 任务 | 尺寸 | 训练方法 | LoRA 目录 |
|---|---|---|---|---|
| 🧠 **Meta-Planner** | 生成双层蓝图骨架 S_q | **14B** | 纯 **SFT** + **SemanticDTW 数据筛选** | `planner_lora` |
| 🔄 **Replanner** | 检索失败时重写计划 | **14B** | **GRPO** + 8→4→2 淘汰赛制 | `replanner_grpo` |
| ✂️ **Extractor** + ⚖️ **Grader** | 抽原子事实 / 判证据充分性 | **3B** | **DPO**（两个角色共享一个 LoRA） | `extractor_grader_dpo` |
| 🗣️ **Reasoner** | 按事实推演 + 出意见书 | **7B**（长上下文）| 纯 **SFT** | `reasoner_sft` |

### （2）为什么这么选方法

- **Planner 用纯 SFT 而非 RL**：搜索空间太大 RL 难收敛；它的本质是「翻译与抽象」，不该自由发挥。
- **Replanner 用 GRPO**：它的奖励信号来自"检索/沙箱是否成功"，可验证，适合 RL。
- **Extractor/Grader 用 DPO**：二值判断（相关/不相关、够/不够），偏好对容易构造。
- **Reasoner 用纯 SFT**：只需严格依据资料作答，不需要额外策略学习。

### （3）两个高价值训练技巧

1. **SemanticDTW 数据筛选**：用 BGE-M3 对「模型生成的步骤序列」与「图数据库黄金路径」做**非对称语义 DTW 对齐评分**，
   `gamma_penalty` 抑制冗余步骤、`lambda_scale` 拉开好坏计划的分数断层，**低于 0.3 的样本直接剔除**，保 SFT 数据纯度。
2. **DPO 配对踩坑**：一开始调高 DeepSeek 的 `temperature` 生成 8 个答案再互相配对 —— **差距太小、效果很差**。
   正确做法：**用自己 SFT 后的输出与 DeepSeek 配对**（并调高基座 KL 散度），既保持语义对齐又不至于混乱。

### （4）数据从哪来：Evol-Instruct 反向出题

```
真实法条（锚点）
   └─ SYNTHESIZE_GROUND_TRUTH_SYSTEM ── 虚构具体案情 → 双层蓝图 Ground Truth → 计算代码
        └─ EVOL_INSTRUCT_SYSTEM ─────── 代入劳动者视角，生成 3 个「残缺 + 高噪点 + 口语化」提问
                                         强制 2 个故意遗漏核心定案事实
        └─ REPLANNER_SCENARIO_SYSTEM ── 构造「检索失败」场景，训练 Replanner
```

注意：**Prompt 正文全部来自 `prompts.py`**，数据工厂本身不含副本。

---

## 7. 成本与安全管控

| 项 | 措施 |
|---|---|
| **推理成本** | vLLM 自部署 + **prefix caching**（固定 system prompt 的 KV-cache 只算一次）；`create_llm_backend("auto")` 先试本地 vLLM，显存不够/未装则**自动回退 DeepSeek API**，上游无感 |
| **Summary 成本** | 不做专门的摘要模型，用商业 API 降维打击（实测 1/10 数据 ≈ 7 元） |
| **数值安全** | LLM 只生成**计算代码**，由断网 Docker 沙箱执行；沙箱限额 512MB / 0.5 核 / `network_mode: none` / `no-new-privileges` |
| **注入防御** | `query.py` 的 L0 网关做前置正则过滤（中英双语 + 分隔符注入模式） |
| **可靠性** | Kafka 手动提交 offset；毒消息进 **DLQ** 而非阻塞分区；Worker 逐条 `try/except` |
| **可观测性** | 结构化日志（json/text）+ 健康检查端口 + Prometheus 端口 + 阶段耗时 Metrics |

> ⚠️ **已知边界**：`sandbox_exec.run_in_process()` 的降级路径（无 Docker 时）**不是安全边界** ——
> 它用受限 `builtins` 挡住了 `open`/`__import__`/`eval`，但**挡不住属性链逃逸**。
> 模块头已显式警告"生产环境必须保证 Docker 沙箱可用"，并有测试记录这个限制。

---

## 8. 评测体系

| 维度 | 做法 |
|---|---|
| **控制变量** | 刻意**剥离 Reasoner 的模型光环** —— 用普通基座（Qwen2.5 / GLM-4）当阅读器，强迫图谱暴露真实召回能力（强模型会靠常识脑补掩盖召回缺陷） |
| **检索指标** | `HR@K`（召回命中率）、`MRR`（平均倒数排名）、`NoiseRatio`（复用 Grader 判上下文噪音率）。**三个指标由同一次 `retrieve()` 的排序列表派生**，避免重复调用 |
| **轨迹评分** | $Score_{traj} = w_1 \cdot \mathbb{I}(Plan_{optimal}) + w_2 \cdot (1 - \frac{Retries}{Max_{retries}}) + w_3 \cdot Acc_{final}$ |
| **深度探针** | **Graph-NIAH**：人工埋「极度隐蔽的跳板节点」，强制 Retriever 走 $Node_A \to Node_B \to Node_C$ 才能拿到核心证据，用于精确测量游走深度与**召回衰减率** |

### 因果评测基准（`benchmark_causal/`，2026-09 新增）

在上述检索/轨迹指标之外，另建了一套**基于 Pearl 因果之梯**的评测：

| 层 | 设问 | 标准答案来源 |
|---|---|---|
| L1 观测 | 给定完整事实 → 问结论 | **确定性 SCM 推导** |
| L2 干预 | $do(X=x)$ 后 → 问结论 | 同源（图手术：X 固定、方程跳过、下游重算） |
| L3 反事实 | 翻转已发生变量 → 问结论 | 同源（双变量干预可做"隔离混杂因子"题） |
| L3′ 抗扰动 | 同上但塞入无关细节 | 与 L3 **完全相同** |

* **三层答案同源** → 天然自洽，不会因标准答案互相矛盾而冤判模型
* **评分第一原则**：能用 Python 判的绝不交给 LLM。法条存在性、**时效性（法不溯及既往）**、
  格式、金额精确匹配都是确定性硬门禁；只有「因果链命中率」「混杂因子是否隔离」才用裁判
* **度量**：因果一致性率（L1∧L2∧L3 全对才计 1）、反事实抗扰动率、分层通过率
* **法条语料**：1428 条 / 9 部法，含 valid_from / valid_to；来源 `LawRefBook/Laws`

---

## 9. 配置与运行

### 配置分层

```
config.yaml（唯一真源）
  llm / vllm / embedding / graph / retriever / router
  kafka（5 topic + 5 consumer group）/ redis / cache / sandbox / agent
  observability / training
        │
        ├── ${DEEPSEEK_API_KEY}          ← 环境变量注入
        ├── ${KAFKA_BOOTSTRAP:-localhost:9092}
        └── ${REDIS_URL:-redis://localhost:6379}
        │
        └── config_loader.cfg  ──→  各模块的 from_config()
```

### 运行

```bash
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env                 # 填 DEEPSEEK_API_KEY

python -m pytest                     # 187 个用例，全离线
python multiple-search/soul.py                 # LangGraph 全链路（Mock）
python dataset/graph.py                        # 图引擎
python dataset/IncrementalMemoryManager.py     # GMM 动态阈值
python dataset/memory_graph_bridge.py          # 双写桥接
```

### 部署形态

- **Worker 集群**：`Dockerfile.worker`（`python:3.10-slim`）+ `entrypoint.sh` 按 `WORKER_TYPE` 分发 → `k8s_worker_deployment.yaml`
- **沙箱**：`legal_sandbox/Dockerfile` + `sandbox_server.py`（FastAPI 执行服务）

---

## 10. 工程化现状与已知边界

### 已完成（阶段 0–10）

| 阶段 | 成果 |
|---|---|
| 0–1 | 修掉 2 处语法错误；补齐依赖清单、启动脚本、镜像 COPY 源、Worker 文件名（连字符不可 import） |
| 2 | **配置收敛**：所有硬编码阈值/假 Key → `config.yaml` + `from_config()` |
| 3 | **Prompt 单一真源**：13 模板 + 10 构造器；`plan_steps_from_raw()` 统一三种历史格式解析 |
| 4 | **图引擎合并**：删掉重复的同名类；维度统一 1024；补全 3 个空实现 |
| 5 | **打通两条链路**：新增 `replanner_rules.py` / `sandbox_exec.py`；修掉队列错位、沙箱容器泄漏、Reasoner 沙箱断链 |
| 6 | **健壮性**：GMM 三处兜底；benchmark 检索去重；DLQ + 毒消息隔离 |
| 7 | **清理**：删 5 份损坏草稿 + 2 个连字符脚本 + 死代码 |
| 8–10 | **测试与清偿**：187 用例；修掉 4 处由测试发现的缺陷；顶点名类型统一；Pydantic V1→V2 迁移 |

### 已知边界（工程环境类，非代码缺陷）

- 降级沙箱路径**不是安全边界**（无 Docker 时的兜底）
- Kafka + Docker 链路**未做端到端联调**（本地无 daemon / 无 Kafka）
- 本机无法 `sh -n` 校验 `entrypoint.sh`、无法构建镜像
- 无 CI；测试需手动执行
- 首次真实运行需联网下载 BGE-M3（≈2GB）；测试与 demo 已用 `StubEncoder` 绕开

### 与设计意图的偏差（值得后续补齐）

| 项 | 现状 |
|---|---|
| `dataset/chunk.py` | README 描述为「语义分块」，**代码实为规则分块**（按空行/句末切分 + 最小长度合并），未用 embedding 做语义边界 |
| `sandbox_server.py` | 执行服务本身缺**超时与输出大小上限**（超时目前靠 `sandbox_manager` 的 HTTP 10s 兜） |
| `k8s_worker_deployment.yaml` | 未设 `securityContext`（runAsNonRoot / readOnlyRootFilesystem / drop capabilities） |
| `_count_active_children` | 每次遍历全图 `O(N)`，大图下可改用度数缓存 |
| igraph 顶点名 | 已统一为 `str`，但若未来接入**非数字 ID 语料**，`as_num_id()` 会拒绝。P0 起 `dataset/prepare_corpus.py` **强制生成 int ID**，该风险已从源头收窄 |
