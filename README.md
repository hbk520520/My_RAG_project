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
├── replanner_rules_report.py      # 🆕 Replanner 规则扩展报告
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
    ├── Unsloth.py                 # 4-bit QLoRA 加速训练
    ├── path-from-ds.py            # Evol-Instruct 反向出题
    ├── query-generated.py         # 高噪点口语化 Query 生成
    └── train/
        ├── Meta-planner.py        # SFT
        ├── Repalnner.py           # GRPO
        ├── Graderand Extractor.py # DPO
        ├── Reasoner.py            # 长上下文 SFT
        └── Retriever (BGE-M3).py  # LoRA + MNR Loss
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

## 原始设计笔记

注意我们的项目是没有generate大模型的
1.dataset
（1）用knn进行冷启动，接下来计算新入库的数据与原粒度的加权和（余弦和BM25），>0.85（不然完全不相似），<0.99(不然几乎完全冗余)-->需要定时去查看是不是真的冗余，然后选取top去连边。
（2）在一开始的knn聚类构建的过程中，对于一定数量的聚类节点，生成summary节点，并且summary节点的地位和普通粒度节点平等（就是说summary节点的连边方法和普通的一样）。利用大模型去用自然语言描写summary节点中的内容（即GNN中的P），但是注意要有最大值截断，即当聚类的数量超过一定上限的时候，进行进一步的聚类--这一点我想了蛮久的，后面决定在构建图的时候先不管，直到构建完之后再进行多次降序（当然这是因为我只有一张租来的卡，如果有多卡算力的可以尝试同时进行，但是要做好备份）
（3）在新的节点入库的时候，为了避免O（N）的全局搜索，应该要引入HNSW作为底层支持（faiss不能去算余弦相似度，但是可以计算归一化后的内积）
（4）批量入库！！！当时本来是一个一个进去的，结果饭都吃完了进度条还没动
（5）这个是关于系统的可纠错性和更新性的：为每一个粒度入库的过程中打上标签，这个标签可以写的还蛮多的，但是我当时还在确定整个具体的架构，所有先对其进行封装，并用一个动态元去占位置。

2.retrieval
这是为了解决复杂的法律问题的，但是为了降低成本，在这个模块之前还要加入一个分类器，加入正则化匹配和prompt缓存。当然可能还有其他的设计，这里先不赘述了。
（1）我们要使用四个模型，第一个是Meta-planner，负责为原问题生成一个抽象的推理骨架；第二个是叫Replaner，负责当查询不到东西的时候修改流程；原本其负责跟踪子问题的解决程度，即这个问题收集到的资料可以解决了吗，对于检查不到的地方是不是要进行重写，后面想了想需要进行解耦，因为参数量不够，而且后面我们会用Reasoner的结果去反推有没有检索完整，因此只去生成重写，而判断的任务交给Grader；第三个是Extractor，从抽取到的文档精简事实。第四个是叫Reasoner，即对我们的子问题根据搜索到的资料进行回答。
（2）在每两次（或者说偶数次的）查询中加入原问题，作为底线，即不要偏离原问题
（3）Meta-planner生成的应该是生成双层蓝图 $P_q=\{S_q,C_q\}$，S_q是一个JSON格式的有向无环图，抽象骨架的意思是先不针对实体-->就是说把所有的具体名词抽象为其本质属性，比如《钢铁侠：1》抽象为电影，C_q是将抽象的问题实例化。
（4）注意LLM的本质是概率预测，因此不要用单纯的LLM去当Control-layer，因为这样子会影响准确率，应该要用纯python代码去记录状态，然后LLM只是作为判断器就好，记得每次询问都要清除上下文！！！不然后面就变蠢了
（5）还要再加入一个模型，Reasoner，负责对每一个子问题进行回答，子问题来自于Meta—planner的。
（6）对于这么多的模型，用LangGraph去进行框架设计，在具体的框架中我们可以引入有限状态机的思想-->引入有限状态机的思想，每次解答问题都是在实例化一个状态机，包含以下核心要素：
State（全局状态载荷）：
 这不是一个简单的字符串，而是一个贯穿整个生命周期的结构体对象。在 LangGraph 中，它通常包含：{user_query: str, current_context: list, current_hop: int,  intermediate_thoughts: list}。
 Nodes（执行节点/状态）：
 这是图上的一个个停靠站。比如：
 Node_Evaluate：大模型反思节点（判断信息是否足够）。
 Node_Retrieve_Law：法律数据库检索工具。
 Node_Retrieve_Finance：金融数据库检索工具。
 Node_Generate：最终答案生成节点。
 Edges & Conditional Routing（边与条件路由）：
 这是状态机的转移函数 $T(S, A)$。在这个系统中，大模型（或你微调的小模型）充当了最核心的“路由决策器”。它读取当前 State，决定下一条边指向哪里。
 Halt Conditions（终止状态）：
 为了防止死循环爆炸，必须设定硬性出口：
 成功出口：Node_Evaluate 认为证据链已闭环，流转到 Node_Generate。
 熔断出口：current_hop 达到我们设定的最大阈值（比如 3），强制流转到错误或降级处理节点。
（7）不能使用三个不同的大模型，因为其可能会面临训练成本过高且语义流形不对齐的问题，我们应该换成一个通用的大模型基座，然后挂靠上3个不同的lora模块，用VLLM去做底层引擎

3.训练
（1）上面提到的三个模型的训练，第一步都是一样的，用大模型的api先生成轨迹，然后对轨迹进行切分，接下来交给模型进行学习（sft）。
（2）对于后面的训练，Meta-planner用单纯的sft进行训练；Replaner用GRPO去训练，写一段代码去评价其生成答案的好坏（准确性和长度）；在训练Extractor和Grader的时候要用DPO，这里我们可以同样用deepseek的api去生成（注意要求步骤尽可能的少），然后与自己经过sft微调后的答案去配对（M×N个对）-->这里有一个很容易失误的点，就是一开始我是调高了deepseek的temperature，让它生成8个不同的答案，我再按照步骤长短，匹配性去评分然后配对，但是这样子生成的训练对差距太小了。而且用自己sft后生成的数据既可以保持语义对齐，又不会太混乱了。所以最终方法是这样子的：先用ds和自己的（KL值调高一点），再用ds自身的结果去进行两个阶段的训练；在训练Reasoner的时候只需要进行SFT。
（3）用Unsloth去加速训练！！非常好用。
（4）补充一点，Meta-Planer只用SFT的原因有两个，一个是空间太大了，用RL的话难以去收缩，另一个是不需要自由发挥，其作用在于翻译。而Reasoner用SFT的原因是其只需要根据资料得出答案就好了。
（5）针对于模型需要，Meta-planner和Replaner要挂靠14B的模型，而另外两个只需要3-4B的模型即可。
4.成本
（1）最开始的summary节点的自然语言的内容生成，我们的summary数据是在图构建完成后去生成的，这样子可以减少成本，并且使用deepseek的超低价api后构建十分之一的数据只花了七块钱。因此本来想要重新训练一个模型去生成总结的，后面不需要了。
5.安全性
（1）在计算一些比如说税率，赔偿金之类的问题时，我们不能用LLM去生成答案，而是应该多一个步骤，让LLM生成python代码去计算（注意做好多次修改的准备），并且注意需要在沙盒中去进行，否则很可能会被注入恶意指令。因此我们要把它放到一个封闭的环境中，我弄了一个基于本地 Docker 的 带状态沙箱调度器。
（2）在plus中对于恶意prompt先简单写了一个过滤器。
6.benchmark
（1）控制变量的执行层：剥离强大 Reasoner 的光环，用普通模型（Qwen2.5/GLM-4）充当阅读器，逼迫底层图谱暴露真实召回能力。
（2）严苛的量化标尺：使用召回命中率 ($HR@K$)、平均倒数排名 ($MRR$) 和基于 Grader 节点复用的上下文噪音率 ($Noise\ Ratio$)去判断。后面升级成这样子：
$$Score_{traj} = w_1 \cdot \mathbb{I}(Plan_{optimal}) + w_2 \cdot (1 - \frac{Retries}{Max_{retries}}) + w_3 \cdot Acc_{final}$$（$w_1$: 初始规划是否最优；$w_2$: 重试效率惩罚；$w_3$: 最终结论准确度）
（3）Evol-Instruct 演化数据工厂：基于“法庭事实锚点”，注入“残缺/高噪/口语化”用户画像，并辅以法官模型交叉验证
（4）在我们的图数据库中人工埋入一个“极度隐蔽的跳板节点”。在测试用例中，强制要求必须经历 $Node_A \rightarrow Node_B \rightarrow Node_C$ 的推理链才能拿到证据。专门监测 Retriever 的游走深度和召回召回衰减率。
