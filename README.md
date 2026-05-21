# My_RAG_project — 法律智能体 RAG-plus

> 从基础 RAG 演进为 Plan-and-Replan 法律智能体系统

## 项目结构

```
.
├── config.yaml                    # 🆕 统一配置中心
├── config_loader.py               # 🆕 配置加载器 (${ENV_VAR} 替换)
├── prompts.py                     # 🆕 Prompt 模板库 (11种场景)
├── observability.py               # 🆕 可观测性 (Metrics/健康检查/DLQ/指数退避)
├── query.py                       # 路由层: UnifiedQueryRouter_Query (三层漏斗)
├── benchmark.py                   # 评测层: 检索/轨迹/Graph-NIAH
├── double_layer_plan.py           # 🆕 双层蓝图 Schema (S_q DAG + C_q 具象化)
├── training_data_guide.py         # 🆕 训练数据工程化建议
│
├── dataset/                       # 知识图谱层
│   ├── chunk.py                   # PDF 提取 + 语义分块
│   ├── graph.py                   # 核心图引擎 (BGE-M3 + FAISS-HNSW + LoRA)
│   ├── IncrementalMemoryManager.py # GMM 动态阈值增量记忆
│   ├── memory_graph_bridge.py     # 🆕 桥接: GMM记忆 ↔ igraph图引擎
│   └── prepare_corpus.py          # 批量向量化注入
│
├── asynchronization/              # 异步编排层 (Kafka + Redis)
│   ├── kafka_utils.py             # Kafka Producer/Consumer (5个Topic)
│   ├── state_manager.py           # Redis 状态脱水/复水
│   ├── Dockerfile.worker          # 统一 Worker 镜像
│   ├── entrypoint.sh              # 按 WORKER_TYPE 分发
│   ├── k8s_worker_deployment.yaml # K8s 部署配置
│   └── workers/
│       ├── planer-worker.py       # Meta-Planner Worker
│       ├── retriever_worker.py    # 图检索 Worker
│       ├── grader_worker.py       # 🆕 Grader Worker
│       ├── replanner_worker.py    # 🆕 Replanner Worker (v2 Pydantic)
│       └── reasoner_worker.py     # 🆕 Reasoner Worker
│
├── multiple-search/               # 多智能体推理层
│   ├── soul.py                    # LangGraph 主控 (含沙箱闭环)
│   ├── SemanticCache/
│   │   └── engine.py              # 语义缓存 (InMemory + RedisVL)
│   └── legal_sandbox/
│       ├── Dockerfile             # 沙箱镜像
│       ├── sandbox_server.py      # FastAPI 执行服务
│       ├── sandbox_manager.py     # Docker 调度器
│       └── example_usage.py       # 集成示例
│
└── model/                         # 训练层
    ├── data/
    │   └── evol_instruct.py       # 🆕 Evol-Instruct 数据工厂 (合并)
    ├── training/
    │   ├── train_meta_planner.py  # Meta-Planner SFT + DTW筛选
    │   ├── train_replanner_grpo.py# 🆕 Replanner GRPO 淘汰赛制
    │   ├── train_extractor_grader.py# Extractor+Grader DPO
    │   ├── train_reasoner.py      # Reasoner 长上下文 SFT
    │   └── train_retriever.py     # BGE-M3 LoRA 微调
    └── utils/
        └── unsloth_loader.py      # 🆕 Unsloth 通用加载器
```

## 数据流

```
用户 → SemanticCache → UnifiedQueryRouter_Query (L0→L1→L2)
        ├─ CHITCHAT → 闲聊
        ├─ SIMPLE_QA → 简单RAG
        └─ COMPLEX_TASK → UnifiedQueryRouter_Soul
                → LangGraph: L0_Gateway → Planner → Executor → Grader
                → Replanner (失败→虫洞) → Generate → WriteCode → ExecuteCode(沙箱)
```

## 更新日志 (2026-05-21)

| 类别 | 变更 |
|---|---|
| Worker | 新增 grader / replanner(v2 Pydantic) / reasoner Worker |
| 路由 | query.py process() 连通 soul.py LangGraph Agent |
| 配置 | config.yaml + config_loader.py 统一所有硬编码 |
| 沙箱 | WriteCode → ExecuteCode → InjectResult → Cleanup 闭环 |
| 图引擎 | load_lora_weights() 对齐检索器训练; memory_graph_bridge 双向同步 |
| 可观测性 | 结构化日志 + Metrics + 健康检查 + DLQ + 指数退避 |
| Prompt | prompts.py 统一 11 种模板 |
| Replanner | v2: Pydantic Schema + engine选择(GRAPH_TRAVERSAL/GLOBAL_DENSE_WORMHOLE) + rationale 可回溯 |

---


# 📖 系统架构与核心设计笔记 (Design Notes)

> **⚠️ 架构声明**：本项目为纯粹的 RAG/Agent 编排架构，**整个项目没有 generate 大模型**。核心重心在于图谱构建、多智能体协同、垂直领域（如法律）工程落地与推理加速。

---

## 1. 向量图谱与数据集构建 (Dataset & Graph)

图谱数据库的构建是底层的核心支撑，主要解决冷启动、节点冗余、长文本处理与系统更新问题。

### （1）冷启动与入库相似度控制
* **是什么**：新数据入库时与原粒度节点的相似度判定机制。
* **为什么**：防止完全不相关的孤立节点混入，同时剔除完全冗余的信息。
* **怎么用**：利用 **KNN** 进行初始冷启动聚类。接下来，计算新入库数据与原粒度的加权和（结合 **余弦相似度** 与 **BM25**）。
* **控制阈值**：
    * 分数必须 **`> 0.85`**（否则判定为完全不相似，不予连边）。
    * 分数必须 **`< 0.99`**（否则判定为几乎完全冗余）。
* **故障处理与维护**：设置定时任务去查看和审计是不是真的冗余，然后选取 Top 节点去进行连边。

### （2）Summary 节点的延迟生成与算力妥协
* **是什么**：在图谱中引入高层级的抽象总结节点，其连边方法和地位与普通粒度节点完全平等。
* **为什么**：建立全局和局部之间的语义桥梁（即 GNN 中的特征载荷 `P`）。
* **怎么用**：在一开始的 KNN 聚类构建过程中，对于一定数量的聚类节点，利用大模型去用自然语言描写 Summary 节点中的内容。
* **避坑指南（最大值截断与单卡限制）**：
    * 逻辑上必须有**最大值截断**，即当聚类的数量超过一定上限的时候，需要进行进一步的聚类。
    * **实战心路历程**：这一点我想了蛮久的，后面决定在构建图的时候先不管，直到构建完之后再进行多次降序。*原因在于当时我只有一张租来的卡，如果有多卡算力的开发者可以尝试同时进行，但是务必要做好备份。*

### （3）高效检索底层支持
* **是什么**：避免大数量级下的检索性能崩溃。
* **为什么**：随着节点增加，全局搜索复杂度会飙升至 `O(N)`，导致致命延迟。
* **怎么用**：在新节点入库的时候，引入 **HNSW** 作为底层支持。
* **禁忌点（工具选型）**：注意 FAISS 无法直接去算余弦相似度，但是可以通过计算**归一化后的内积**来达到相同效果。

### （4）工程吞吐优化
* **避坑指南（批量入库）**：**必须采用批量入库方案！！！** 当时系统本来设计成是一个一个节点进去的，结果“饭都吃完了进度条还没动”，单点 I/O 开销极高，必须走 Batch 流水线。

### （5）系统可纠错性与热更新
* **怎么用**：为每一个粒度入库的过程中打上标签（Tag）。
* **架构设计**：虽然标签后续可以扩展写得蛮多的，但由于当时还在确定整个具体的架构，所以选择先对其进行抽象封装，并**用一个动态元（Dynamic Meta）去占位置**，便于后期无缝升级。

---

## 2. 多智能体检索与推理管线 (Retrieval & Reasoning)

本模块旨在解决复杂的法律/垂直领域问题。为了前置降低运行成本，在核心模块启动前，加入了**分类器、正则化匹配**与 **Prompt 缓存**。

### （1）四大核心模型矩阵 (Agent Roles)
1. 🧠 **Meta-planner（元规划器）**：负责为原问题生成一个抽象的推理骨架。
2. 🔄 **Replaner（重规划器）**：当查询不到目标信息时，负责修改和重写流程。
   * *架构解耦决策*：原本其还负责跟踪子问题的解决程度（即收集到的资料是否足够解决问题、检查不到的地方是否重写）。后面想了想，因为参数量不够，且后续会用 Reasoner 的结果去反推有没有检索完整，所以需要对其进行解耦——**让 Replaner 只负责生成重写，而将判断的任务完全交给 Grader 节点**。
3. ✂️ **Extractor（抽取器）**：负责从抽取到的复杂文档中精简核心事实。
4. 🗣️ **Reasoner（推理器）**：负责对每一个子问题（子问题来自于 Meta-planner）根据搜索到的资料进行最终回答。

### （2）底线防偏离机制
* **怎么用**：在每两次（或者说偶数次的）查询中，**强制加入原始问题（Original Query）**作为底线。
* **为什么**：防止多跳检索（Multi-hop）在长链条推理中逐渐偏离主题。

### （3）双层蓝图输出规范
* **是什么**：Meta-planner 生成的内容应该是双层蓝图 **$P_q=\{S_q,C_q\}$**。
* **数据结构**：
    * **$S_q$**：一个 **JSON 格式的有向无环图 (DAG)**。抽象骨架的意思是**先不针对具体实体**。也就是把所有的具体名词抽象为其本质属性（例如：把《钢铁侠：1》抽象为“电影”）。
    * **$C_q$**：负责将上述抽象的问题进行具象化、实例化。

### （4）确定性控制层设计 (Control-layer)
* **禁忌点**：**注意 LLM 的本质是概率预测，因此绝对不要用单纯的 LLM 去当 Control-layer！** 这样会严重影响系统的状态准确率。
* **标准化做法**：应该使用**纯 Python 代码**去记录和维护系统状态。LLM 应当仅仅作为状态机内部的“判断器”。
* **故障处理**：**记得每次询问模型后都要清除上下文！！！不然后面上下文堆积模型就变蠢了。**

### （5）基于 LangGraph 的有限状态机 (FSM) 架构
利用 LangGraph 进行框架设计，每次解答问题都是在实例化一个有限状态机，包含以下核心要素：
* 📦 **State (全局状态载荷)**：这不是一个简单的字符串，而是一个贯穿整个生命周期的结构体对象。在 LangGraph 中包含：`{user_query: str, current_context: list, current_hop: int, intermediate_thoughts: list}`。
* 🚉 **Nodes (执行节点/状态)**：图上的停靠站。例如：
    * `Node_Evaluate`：大模型反思节点（判断信息是否足够）。
    * `Node_Retrieve_Law`：法律数据库检索工具。
    * `Node_Retrieve_Finance`：金融数据库检索工具。
    * `Node_Generate`：最终答案生成节点。
* 🔀 **Edges & Conditional Routing (边与条件路由)**：即状态机的转移函数 **$T(S, A)$**。在此系统中，我们微调的小模型充当“路由决策器”，它读取当前 State，动态决定下一条边指向哪里。
* 🚨 **Halt Conditions (终止与熔断状态)**：为了防止死循环导致 Token 爆炸，设定两类出口：
    * *成功出口*：`Node_Evaluate` 认为证据链已闭环，流转到 `Node_Generate`。
    * *熔断出口*：`current_hop` 达到设定的最大阈值（比如 3），强制流转到错误或降级处理节点。

### （6）底层模型部署与对齐优化
* **禁忌点**：**不能使用三个完全不同的独立大模型**。否则会面临微调训练成本过高、且不同模型间语义流形（Semantic Manifold）不对齐的问题。
* **正确做法**：换成一个**通用的大模型基座 + 挂靠 3 个不同的 LoRA 模块**，并统一使用 **vLLM** 作为底层推理加速引擎。

---

## 3. 模型训练策略 (Training Pipeline)

### （1）通用第一阶段 (Trajectory Generation)
* **怎么用**：上面提到的三个模型，训练的第一步完全一样：先使用大模型的 API **生成交互轨迹（Trajectory）**，然后对轨迹进行科学**切分**，最后交给对应模型进行第一轮 **SFT (监督微调)**。

### （2）分支微调与强化策略
* 🧠 **Meta-planner**：采用**单纯的 SFT** + **SemanticDTW 数据筛选**进行训练。
  * *原因有二*：一是其搜索空间太大，使用强化学习（RL）难以收敛；二是其本质作用在于"翻译与抽象"，不需要任何自由发挥。
  * **📐 SemanticDTW 数据筛选**：使用 BGE-M3 对 Planner 生成的步骤序列与图数据库黄金路径进行**非对称语义 DTW (Dynamic Time Warping)** 对齐评分。通过 `gamma_penalty` 抑制冗余步骤，`lambda_scale` 控制好/坏计划的分数断层。低于阈值 (0.3) 的低质量训练样本自动剔除，确保 SFT 数据纯度。
* 🔄 **Replaner**：采用 **GRPO (群体相对策略优化)** + **8→4→2 三层淘汰赛制**进行强化训练。

  **🏟️ 淘汰赛机制 (GRPO Tournament)**：
  1. 模型为每个 Prompt 生成 **8 个候选计划**
  2. **Round 1 (8→4)**：8 个候选各自执行第一跳检索，用 **BGE-Reranker-v2-m3** 作为 PRM (Process Reward Model) 进行毫秒级相关度打分，淘汰得分最低的 4 个（施加早死惩罚）
  3. **Round 2 (4→2)**：4 个幸存者执行第二跳检索，PRM 再次打分，再淘汰 2 个
  4. **Final Round (2 幸存)**：仅存的 2 个精英方案完整跑沙箱终局裁判，成功者获巨额奖金 (+15)，失败者受重罚 (-5)

  **核心收益**：8 个候选只需对 2 个跑昂贵沙箱，其余 6 个用毫秒级 PRM 快速淘汰 → 成本降低 ~75%
* ✂️ **Extractor & Grader**：采用 **DPO (直接偏好优化)** 进行对齐训练。同样使用 DeepSeek API 去生成训练对（注意：Prompt 要求步骤尽可能的少），然后与自己经过 SFT 微调后的答案去配对（构成 $M \times N$ 个对）。
  * **💡 核心踩坑省钱经验（极其重要）**：一开始我为了拉开差距，调高了 DeepSeek 的 `temperature`，让它生成 8 个不同的答案，我再按照步骤长短、匹配性去评分并进行正负对配对。**事实证明这样子生成的训练对差距太小了，效果很差**。改用自己 SFT 后生成的数据与 DS 配对，既可以保持系统的语义对齐，又不会太混乱。
  * **最终路线**：先用 DeepSeek 和自己 SFT 的结果配对（此时把自己的基座模型 KL 散度值调高一点），再用 DeepSeek 自身的结果去进行两个阶段的训练。
* 🗣️ **Reasoner**：只需要进行**单纯的 SFT**。
  * *原因*：它只需要严格根据检索到的资料得出答案即可，不需要额外的策略学习。

### （3）模型参数规模与加速
* **模型分工**：Meta-planner 和 Replaner 对逻辑链要求极高，挂靠 **14B** 规模的模型；而 Extractor 和 Grader 属于专用工具节点，只需要 **3B - 4B** 规模的模型即可。
* **加速工具**：全量微调环节全面引入 **Unsloth** 库进行加速，非常好用。

---

## 4. 成本与安全管控 (Cost & Safety)

* **Summary 生成成本控制**：最开始关于 Summary 节点的自然语言总结内容，选择在**图结构整体构建完成后**再去集中生成。这样可以大幅减少中间调用的频率。通过使用 DeepSeek 的超低价 API，**构建十分之一的数据最终仅仅花费了 7 块钱**。因此，原本计划单独训练一个总结模型的方案直接被砍掉，改用商业 API 降维打击。
* **计算沙盒与代码注入防御**：
    * **禁忌点**：在计算类似“税率、赔偿金”等强数值问题时，**绝对不能让 LLM 直接生成数字答案**（极易出错）。
    * **安全架构**：多加一个逻辑步骤，让 LLM 生成 **Python 计算代码**（注：开发时需做好代码被多次修改优化的准备）。
    * **沙箱隔离**：代码必须在完全封闭的环境中运行，否则极易遭受恶意指令注入攻击。为此，我专门实现了一个**基于本地 Docker 的“带状态沙箱调度器”**。
* **前置过滤**：在 Plus 版本中，针对恶意的 Prompt 注入，前置编写了一个轻量级过滤器。

---

## 5. 评测基准体系 (Benchmark)

### （1）控制变量执行层 (Reader Degradation)
* **怎么用**：在评测图谱召回能力时，故意**剥离强大 Reasoner 的模型光环**。
* **为什么**：强模型（如 GPT-4）具备极强的常识脑补能力，会掩盖检索召回率低的缺陷。
* **实现方式**：故意换用普通基座模型（如 `Qwen2.5` / `GLM-4`）充当阅读器，强迫底层图谱暴露最真实的召回和支撑能力。

### （2）严苛的量化标尺
* **核心指标**：使用召回命中率 (**$HR@K$**)、平均倒数排名 (**$MRR$**) 以及基于 Grader 节点复用的上下文噪音率 (**$Noise\ Ratio$**) 去综合判断。
* **轨迹质量评分公式**：后续系统升级了多跳评判标准：
  $$Score_{traj} = w_1 \cdot \mathbb{I}(Plan_{optimal}) + w_2 \cdot (1 - \frac{Retries}{Max_{retries}}) + w_3 \cdot Acc_{final}$$
  *(其中：$w_1$ 代表初始规划是否最优；$w_2$ 代表重试效率惩罚；$w_3$ 代表最终结论准确度)*

### （3）Evol-Instruct 演化数据工厂
* **怎么用**：基于真实的“法庭事实锚点”，通过模型逆向注入“残缺、高噪声、极度口语化”的用户画像问题，并辅以法官模型（Judge Model）进行交叉验证。

### （4）深度穿透探针 (Deep Hop Probe)
* **核心玩法**：在我们的图数据库中人工埋入一个**“极度隐蔽的跳板节点”**。
* **测试条件**：在测试用例中，强制要求 Retriever 必须精准经历 **$Node_A \rightarrow Node_B \rightarrow Node_C$** 的长推理链条才能拿到最终核心证据。专门用于精准监测 Retriever 的游走深度和召回衰减率。
