# My_RAG_project — 法律智能体 RAG-plus

> 从基础 RAG 演进为 **Plan-and-Replan 法律智能体系统**，并补上持久化执行内核与因果评测基准

📄 **完整项目介绍（技术选型 / 分层架构 / 整体规划 / 训练与评测 / 工程化现状）：见 [`docs/PROJECT_OVERVIEW.md`](docs/PROJECT_OVERVIEW.md)**

## 项目结构

```
.
├── config.yaml                    # 统一配置中心 (含 VLLM 自部署配置)
├── config_loader.py               # 配置加载器 (${ENV_VAR:-默认值} 替换)
├── prompts.py                     # Prompt 单一真源 (13 个模板 + 10 个构造器)
├── observability.py               # 可观测性 (Metrics/健康检查/DLQ/指数退避)
├── query.py                       # 路由层: UnifiedQueryRouter_Query (三层漏斗)
├── benchmark.py                   # 评测层: 检索/轨迹/Graph-NIAH
├── double_layer_plan.py           # 双层蓝图 Schema (S_q DAG + C_q 具象化)
├── replanner_rules.py             # 硬规则表 (soul.py 与 replanner_worker 共用一份)
├── replanner_rules_report.py      # 规则覆盖度报告
├── training_data_guide.py         # 训练数据工程化建议
│
├── dataset/                       # 知识图谱层
│   ├── chunk.py                   # PDF 提取 + 规则分块（按空行/句末切分 + 最小长度合并）
│   ├── graph.py                   # 核心图引擎 (BGE-M3 + FAISS-HNSW + LoRA)
│   ├── IncrementalMemoryManager.py # GMM 动态阈值增量记忆
│   ├── memory_graph_bridge.py     # 桥接: GMM 记忆 ↔ igraph 图引擎
│   └── prepare_corpus.py          # 语料入库: 文本 → nodes.jsonl + vectors.npy
│
├── asynchronization/              # 异步编排层 (Kafka + Redis)
│   ├── kafka_utils.py             # Kafka Producer/Consumer (5 个 Topic)
│   ├── state_manager.py           # Redis 状态脱水/复水
│   ├── Dockerfile.worker          # 统一 Worker 镜像
│   ├── entrypoint.sh              # 按 WORKER_TYPE 分发
│   ├── k8s_worker_deployment.yaml # K8s 部署配置
│   └── workers/
│       ├── planner_worker.py      # Meta-Planner Worker
│       ├── retriever_worker.py    # 图检索 Worker（向量召回 + 图游走）
│       ├── grader_worker.py       # Grader Worker
│       ├── replanner_worker.py    # Replanner Worker
│       └── reasoner_worker.py     # Reasoner Worker（含内联沙箱）
│
├── multiple-search/               # 多智能体推理层
│   ├── soul.py                    # LangGraph 主控 (含沙箱闭环)
│   ├── SemanticCache/
│   │   └── engine.py              # 语义缓存 (InMemory + RedisVL)
│   └── legal_sandbox/
│       ├── Dockerfile             # 沙箱镜像
│       ├── sandbox_server.py      # FastAPI 执行服务
│       ├── sandbox_manager.py     # Docker 调度器
│       ├── sandbox_exec.py        # 执行统一入口 (soul.py 与 reasoner_worker 共用)
│       └── example_usage.py       # 沙箱集成示例（离线可跑）
│
├── agent_runtime/                 # 持久化执行内核（检查点 / 幂等台账 / 任务总线 / 兜底恢复）
│   ├── contracts.py               # 任务契约 (TaskRequest / TaskResult / 状态机)
│   ├── checkpointer.py            # sqlite / Redis / memory 检查点工厂
│   ├── ledger.py                  # 双表幂等台账 (attempts + dispatched)
│   ├── taskbus.py                 # 任务总线 (Kafka 旁路 + 内存/文件实现)
│   ├── nodes.py                   # DispatchTask / AwaitTask 节点
│   ├── recoverer.py               # 兜底驱动：结果回收 + 超时补偿
│   └── run_demo.py                # 离线演示（挂起 / 跨进程恢复 / 幂等）
│
├── benchmark_causal/              # 因果评测基准（见下节）
│   ├── schemas.py / scm.py / scm_labor.py / scenarios.py
│   ├── legal_corpus.py            # 法条存在性 + 时效性判定
│   ├── generator.py / gates.py    # L1/L2/L3 出题 + 硬门禁打分
│   ├── build_corpus.py            # 从 LawRefBook/Laws 构建语料
│   ├── run_demo.py                # 离线演示
│   └── data/legal_corpus.jsonl    # 已入库：1428 条 / 9 部法
│
├── model/                         # 训练层（P9，暂挂起）
│   ├── data/evol_instruct.py      # Evol-Instruct 数据工厂
│   ├── training/                  # 5 个训练脚本 (SFT / DPO / GRPO / LoRA)
│   └── utils/
│       ├── unsloth_loader.py      # Unsloth 通用加载器
│       └── vllm_engine.py         # VLLM 自部署推理 (prefix caching)
│
├── tests/                         # 测试（全部离线可跑）
│   ├── conftest.py                # 统一 sys.path 引导 + 公共夹具
│   ├── harness.py                 # 假 LLM / 内存 Kafka / 脚本化编码器
│   └── ...
│
├── tools/
│   ├── ci.py                      # ⭐ 一键回归（先跑这个）
│   └── audit_entrypoints.py       # 入口点导入安全性体检
│
├── spikes/
│   └── spike_checkpoint_resume.py # LangGraph 检查点/恢复 go-no-go 验证
│
└── docs/
    ├── CONTRACTS.md               # ⭐ 接口契约（改契约必须同步改它）
    └── PROJECT_OVERVIEW.md        # 项目全貌介绍
```

## 快速开始

```bash
pip install -r requirements.txt          # 运行时依赖
pip install -r requirements-dev.txt      # 追加测试依赖（pytest）
pip install -r requirements-train.txt    # 训练 / GPU 侧（unsloth / vllm）

cp .env.example .env                     # 填入 DEEPSEEK_API_KEY 等
```

### 一条命令跑完所有验证

```bash
python tools/ci.py
```

它会依次跑五类检查，任一失败即退出码非 0：

| 步骤 | 内容 | 为什么不能省 |
|---|---|---|
| 1. 入口点体检 | 每个入口 `import` 都不得有副作用 | "import 即副作用"是本项目反复出问题的一类 |
| 2. compileall | 全仓语法/字节码编译 | — |
| 3. pytest | **325+** 个单元 + 端到端测试 | — |
| 4. 离线 demo | 每个子系统的 `__main__` **真跑一遍** | `compileall` 通过 ≠ 能跑 |
| 5. 持久化 spike | LangGraph 检查点 / 跨进程恢复 / 幂等 | 持久化执行的地基 |

> 第 1 步的必要性有三次实例：阶段 2 的 `import soul` 就持有写死的假 API Key；
> 阶段 6 的 `training_data_guide.py` 顶层 `print`；P0 的 `example_usage.py`
> 模块级 `DockerSandboxManager()`（无 Docker 的环境连 import 都失败）。
> **`compileall` 与 `pytest` 都抓不到这类问题** —— pytest 走 conftest 已经铺好环境。

单独跑体检：

```bash
python tools/audit_entrypoints.py          # 完整矩阵：导入安全性 + 离线可跑性
python tools/audit_entrypoints.py --check  # 只跑断言（CI 用）
```

也可以单独跑某个离线示例（**不联网、不下载模型、不要 API key**）：

```bash
python multiple-search/soul.py                  # LangGraph 全链路（Mock AgenticOps）
python dataset/graph.py                         # 图引擎：建图/删除/摘要重算/最大值截断
python dataset/IncrementalMemoryManager.py      # GMM 动态阈值 + 新簇创建
python dataset/memory_graph_bridge.py           # GMM ↔ 图引擎双写
python benchmark_causal/run_demo.py             # 因果评测：法条体检 + L1/L2/L3 出题 + 打分
python multiple-search/legal_sandbox/example_usage.py   # 沙箱工具返回契约（成功/失败）
python agent_runtime/run_demo.py                # 持久化执行：挂起 / 跨进程恢复 / 幂等
python spikes/spike_checkpoint_resume.py        # 检查点/恢复可行性（6 项验证）
```

语料入库（把一批法律文本变成图引擎能吃的节点 + 向量）：

```bash
python dataset/prepare_corpus.py --corpus-dir ./laws --out ./dataset/corpus_out --stub-encoder
# 去掉 --stub-encoder 即改用 BGE-M3 真实编码（首次会下载约 2GB）
python dataset/chunk.py <文件.pdf>              # 单独的 PDF → 语义分块
```

> 离线示例的秘诀是 `dataset/graph.py` 里的 `StubEncoder` / `make_offline_engine()`：
> 注入一个「同文本必得同向量」的假编码器，从而完全绕开 BGE-M3（约 2GB）下载。
> 生产路径仍然用 `BGEM3FlagModel`，只在真正需要编码时才懒加载。
>
> 端到端测试用的是 `tests/harness.py::ScriptedEncoder` —— 它**按关键词主题**分桶，
> 让「检索到哪一条法条」变成可断言的事实。`StubEncoder` 做不到这点：它的向量两两
> 近乎正交，会被 `Reasoner.retrieve` 的 `sim > 0.6` 全部滤掉，happy path 根本测不到。

**接口契约在 `docs/CONTRACTS.md`** —— 改契约必须同步改文档 + 实现 + 测试。

## 数据流

```
用户 → SemanticCache → UnifiedQueryRouter_Query (L0→L1→L2)
        ├─ CHITCHAT → 闲聊
        ├─ SIMPLE_QA → 简单RAG
        └─ COMPLEX_TASK → UnifiedQueryRouter_Soul
                → LangGraph: L0_Gateway → Planner → Executor → Grader
                → Replanner (失败→虫洞) → Generate → WriteCode → ExecuteCode(沙箱)
```

需要等远端长任务时，主图**不阻塞等待**，而是走持久化执行旁路：

```
Executor/AwaitTask ──派发──> TaskBus (topic.task.pending)
      │                            │
   interrupt()                     │ 远端执行
      │                            ▼
   挂起（进程可退出）          结果回传 (topic.task.result)
      │                            │
      └──── Command(resume=...) <──┘        由 recoverer 兜底驱动
                （可在另一个进程）
```

---

## 持久化执行内核（`agent_runtime/`）

> **要解决的问题**：Agent 调一个远端工具可能几十秒到几分钟。**不能阻塞进程干等**，
> 而挂起之后又必须保证**重启不重复副作用**（否则会重复派发、重复扣费、重复写库）。

### 先验证，再设计

方案落地前先写了一个 go/no-go spike（`spikes/spike_checkpoint_resume.py`），
6 项验证全部通过：检查点落盘 / `interrupt()` 真挂起且**进程正常退出** /
**另一进程**重开 sqlite 用 `Command(resume=...)` 续跑到 END / 重复投递只派发一次。

它换回来两条**反直觉的 LangGraph 语义**，这两条直接决定了整个架构：

1. `interrupt()` **之前**的代码在恢复时**会重跑**
2. LangGraph **不提交“未完成节点”的状态增量** → 恢复时 `state` 里没有上次的痕迹

→ 所以：**禁止用 `state` 记“我已经做过 X”**，副作用必须落在 **state 之外**的幂等台账上。

### 模块

| 文件 | 职责 |
|---|---|
| `contracts.py` | 任务契约：`TaskRequest`（幂等键 / `deadline_ts` / `attempt`）、`TaskResult`（`PENDING/RUNNING/DONE/FAILED/TIMEOUT` + `error_kind`）|
| `checkpointer.py` | sqlite（开发）/ Redis（生产）/ memory 工厂，按配置切换 |
| `ledger.py` | **双表幂等台账**：`attempts` 每次调用都记、`dispatched` 主键去重 → 能区分「执行了几次」与「实际生效几次」|
| `taskbus.py` | `TaskBus` 协议 + Kafka 实现（旁路 `topic.task.pending` / `topic.task.result`）+ 内存/文件实现（离线测试与跨进程复现）|
| `nodes.py` | `DispatchTask` / `AwaitTask` 两节点拆分 |
| `recoverer.py` | 兜底驱动：结果已到 → 恢复；`deadline_ts` 超时 → 写 TIMEOUT 再恢复 |

### 为什么把「派发」与「挂起」拆成两个节点

单节点（派发 + `interrupt()` 写在同一个节点里）在 spike 中被实证**会重跑派发** ——
因为未完成节点的状态不被提交。拆成两个节点后，`DispatchTask` 正常返回、
状态增量被提交，恢复时只剩 `AwaitTask` 重跑。

外部台账仍然保留，作为**第二道防线** —— 应对 checkpoint 回滚、整图重跑这类场景。

### 幂等怎么验证

不看“跑通了”，看「执行次数 vs 生效次数」：

| 事件 | attempts | dispatched | 总线任务 |
|---|---|---|---|
| 首次挂起 | 1 | 1 | 1 |
| 重复投递 3 次 | 4 | **1** | **1** |
| 跨进程恢复后 | 5 | **1** | **1** |

「执行了 5 次、实际只派发 1 次」才是幂等生效的证据 —— 只记一个数看不出这个区别。

### 超时与兜底

`interrupt()` 本身**不带超时**，所以不能指望“挂着的会自己醒”。`recoverer.py` 负责扫悬挂会话：

- **结果已到** → 用 `Command(resume=result)` 恢复（正常路径的守护）
- **`deadline_ts` 已过** → 先写一条 `TIMEOUT` 结果，再恢复

→ 把“无限等待”变成“**有界等待 + 有据降级**”。

---

## 因果评测基准（`benchmark_causal/`）

> 不评 ROUGE / BLEU，评「**有没有做对因果推断**」。基于 Pearl 因果之梯的三层结构。

### 三层结构：同一 SCM，三个设问

| 层 | 名称 | 设问方式 |
|---|---|---|
| **L1** | 观测 (Seeing) | 给定完整事实 → 问结论 |
| **L2** | 干预 (Doing) | 对变量施加 $do(\cdot)$ → 问结论 |
| **L3** | 反事实 (Counterfactual) | 翻转已发生变量 → 问结论 |
| **L3′** | 抗扰动 | 与 L3 等价，但题干塞入**无关细节**（"穿了红卫衣"）|

三层标准答案**全部由同一个确定性 SCM 推导**，因此天然自洽 —— 不会出现
"L1 的答案与 L3 的答案互相矛盾"把模型冤判成错的情况。

### 关键设计：`do(·)` 是真的图手术

不是"改改题干"。`do(X=x)` 把 X 固定，**其结构方程被跳过**，下游按剩余方程重算。

以班组加班费为例：混杂因子「项目赶工强度 Z」同时影响「加班时长 X」与「夜班津贴 M2」。
L3 设问是 *"若非赶工，但加班时长仍为 60 小时，总额多少？"* → `do(Z=normal, X=60)`：

```
观测:    Z=crunch → X=60, 加班费=3600, 津贴=300, 总额=3900
L2 干预: do(Z=normal)                  → X=10, 加班费=600,  津贴=0,  总额=600
L3 反事实: do(Z=normal, X=60)          → X=60, 加班费=3600（不变！）, 津贴=0, 总额=3600
                                                   ↑
                            天真答法会答 600（"不赶工→加班少→加班费少"）→ 被判错
```

### 评分原则：能用 Python 判的，绝不交给 LLM

| 判定项 | 谁判 | 性质 |
|---|---|---|
| 响应格式 / 结论非空 | Python | ✅ 确定性 |
| 引用法条**是否存在** | 法条库查询 | ✅ 确定性 |
| 引用法条**在案件发生日是否有效** | 时效区间判定（**法不溯及既往**） | ✅ 确定性 |
| 金额是否精确匹配（容差 1e-2） | Python | ✅ 确定性 |
| 因果链命中率 | 裁判（可注入 LLM；默认确定性词法匹配兜底） | ⚠️ 语义 |
| 混杂因子是否隔离 | 裁判 | ⚠️ 语义 |

**硬门禁不过 → 直接 0 分，连裁判都不用请。** 总分由 Python 按权重合成。

### 多维度度量

* **因果一致性率**：同一 Case 的 L1/L2/L3 必须**全对**才计 1 ——
  "L1 对、L3 错"说明是瞎猫碰上死耗子，该 Case 整体记 0
* **反事实抗扰动率**：L3′ 与 L3 标准答案相同，看模型答案是否被无关信息带偏
* **分层通过率**：L1/L2/L3 各自通过率，用于诊断能力衰减

### 演示输出（`python benchmark_causal/run_demo.py`）

| 假装作答 | L1 | L2 | L3 | 因果一致性率 |
|---|---|---|---|---|
| ① 正确 | 1.0 | 1.0 | 1.0 | **1.0** |
| ② 天真（不赶工→加班少→加班费少） | 1.0 | 1.0 | **0.0** | **0.0** |
| ③ 编造法条（内容对但引不存在的条文） | **0.0** | 1.0 | 1.0 | **0.0** |
| ④ 引用已废止法条（2024 年引《合同法》） | **0.0** | 1.0 | 1.0 | **0.0** |
| ⑤ 无引证作答 | **0.0** | 1.0 | 1.0 | **0.0** |

### 法条语料

* 1428 条 / 9 部法律 / 892 KB，含 **valid_from / valid_to**（时效判定的前提）
* 来源：`LawRefBook/Laws`（《著作权法》第五条规定法规正文不受著作权保护，可自由再分发）
* 重建：`set LAWS_REPO=<已克隆的法条库>` → `python benchmark_causal/build_corpus.py`
* 构建产物已入库（`data/legal_corpus.jsonl`），**保证 `pytest` 在干净克隆上即可运行**

### 新增模块

| 文件 | 职责 |
|---|---|
| `schemas.py` | 数据契约（题目 / 作答 / 裁判 / 得分） |
| `legal_corpus.py` | 法条库：存在性 + 时效性（法不溯及既往）判定 |
| `scm.py` | 确定性结构因果模型执行器（`do` 算子、反事实、混杂因子识别） |
| `scm_labor.py` | 劳动争议两个 SCM（一个含混杂因子、一个不含） |
| `scenarios.py` | 内置测试场景（案情/干预/反事实全部显式声明，可人工复核） |
| `generator.py` | 由 SCM 派生 L1/L2/L3/抗扰动四道题 |
| `gates.py` | 硬门禁 + 打分 + Case 级因果一致性聚合 |

---

## 更新日志 (2026-09-16：持久化执行内核 + 工程基建)

| 项 | 内容 |
|---|---|
| P0 契约冻结 | 新增 `docs/CONTRACTS.md`（**10 条**接口契约），规矩：改契约必须同步改文档+实现+测试 |
| P0 离线沙盘 | `tests/harness.py`：可编程假 LLM / 内存 Kafka / 脚本化编码器 / 禁越接缝客户端；`tests/test_e2e_graph.py` 锁定 happy path 节点序 |
| P0 一键回归 | `tools/ci.py`：入口点体检 → compileall → pytest → 6 个离线 demo 真跑 → 持久化 spike（5 步 / 33 秒）|
| P0 入口点体检 | `tools/audit_entrypoints.py`：47 个入口逐个在**清空凭据**的子进程里 import，断言「导入无副作用」|
| P0 spike | `spikes/spike_checkpoint_resume.py` 6/6 通过，产出两条 LangGraph 恢复硬契约 |
| P1 持久化执行 | 新增 `agent_runtime/`：任务契约 / 检查点工厂 / 双表幂等台账 / 任务总线 / Dispatch-Await 节点 / 兜底 recoverer |
| P1 接缝收敛 | `node_replanner` 虫洞分支不再直连客户端，改走 `agentic_ops` 注入接缝（并加 `_ForbiddenLLMClient` 防绕过）|

> P0 spike 的结论值得单独记一笔：**“未完成节点的状态增量不会被提交”**。
> 第一版 spike 靠 `state["started_tasks"]` 记“已派发”，恢复后 `dispatch_count = 0`
> 导致重复派发 —— 这就是为什么幂等必须落在外部队账，而不能靠 state。

## 更新日志 (2026-09-14：缺陷修复与工程化)

| 阶段 | 内容 |
|---|---|
| 0 止血 | 修掉 2 处语法错误（`grader_worker.py` dict 重复、`dataset/graph.py` 函数头）|
| 1 依赖/启动 | 新增 `requirements*.txt`；`entrypoint.sh` 路径修正；`Dockerfile.worker` COPY 源修正；`planer-worker.py`→`planner_worker.py` |
| 2 配置收敛 | 删掉散落的硬编码阈值/假 Key；全部改读 `config.yaml` 的 `from_config()` |
| 3 Prompt 单一真源 | `prompts.py` 扩到 13 模板 + 10 构造器；`double_layer_plan.plan_steps_from_raw()` 统一解析 |
| 4 图引擎合并 | 重复的 `LegalDenseGraphBuilder` 合并为一份；维度统一 1024；补全空实现 |
| 5 打通两条链路 | 新增 `replanner_rules.py`、`sandbox_exec.py`；**Retriever 不再 pop 队列**（修掉队列错位）；沙箱容器不再泄漏 |
| 6 健壮性 | GMM 三处兜底；benchmark 检索去重；DLQ + `quarantine_message` 毒消息隔离 |
| 7 清理 | 删 `model/train/` 5 份损坏草稿与 2 个连字符脚本；删死代码 `call_deepseek_planner` |
| 8 测试 | `tests/` 156 个用例；修掉"首个叶子没挂到根簇"与"沙箱内无异常类"两处缺陷 |
| 9 向量一致性 | `graph.add_node()` 新增 `dense`/`sparse` 参数；桥接器透传同一次编码结果 → **同一节点不再可能存成两个向量**，并省掉一次重复编码 |
| 10 清技术债 | igraph 顶点名统一 `str(int)`（29 个触点）；Pydantic **V1 风格 `@validator` 迁移到 `@field_validator`**；修掉 `bootstrap_from_graph` 三处 ID 空间错误 |

> 阶段 9 解决的问题：`MemoryGraphBridge.embedding_fn` 的输出原先**只**喂 GMM，
> 图引擎那边由 `add_node()` 用自己的编码器重新算一遍 —— 两者不一致时**不报错**，
> 同一个节点静默分叉成两个不同空间的向量。现在桥接器把同一次编码的
> `dense`/`sparse` 一并透传下去，且通过 `_check_dim` 做维度校验；
> 自定义 `embedding_fn` 若不提供稀疏权重，会打 WARNING 而不是静默降级。

> 阶段 10 解决的问题：本项目原先把 **int** 节点 ID 直接当 igraph 顶点名。
> 除每次 `add_vertex` 都打 `DeprecationWarning`（未来版本将禁止）外，还有个更隐蔽的
> 隐患：`build_initial_graph_batch` 的顶点名类型**由语料决定**，语料给字符串 ID 时
> 就会与增量 `add_node()` 产生的 int 名混用 —— 实测混用后 `vs.find(name=...)` 抛
> `ValueError`，而所有调用点都 `except ValueError: continue`，于是**静默漏节点**。
> 现在约定「图内顶点名恒为 `str(int)`，内部逻辑仍用 int」，边界转换只有
> `dataset/graph.py` 的 `vname()` / `as_num_id()` 两个函数，并由
> `tests/test_graph_vertex_names.py`（31 例）锁定契约。

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

> **关于 Prompt 缓存**：这不是简单的 Q&A 查表（那是 SemanticCache 做的事），而是指在自部署 VLLM 推理时，同一个 system prompt 的 KV-cache 只计算一次，后续所有请求共享——同一个 prompt 模板反复用，GPU 不用重复算 attention。详见下面的「VLLM 自部署推理引擎」章节。

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

#### 🔧 VLLM 自部署推理引擎 (`model/utils/vllm_engine.py`)

这才是 README 开头提到的 **"Prompt 缓存"** 的真正实现：

* **不是 API 调用**：把训练好的 LoRA 权重复用起来，一个 Qwen2.5-14B 基座 + 4 个 LoRA（Planner / Replanner / Extractor / Reasoner），全部跑在自己 GPU 上。
* **VLLM 自动前缀缓存 (`enable_prefix_caching=True`)**：每个模型的 system prompt 是固定的（比如 Planner 始终以"你是法律案件拆解专家"开头）。VLLM 在第一次推理时算出这段 prompt 的 KV-cache，后续同一个 LoRA 发出的请求直接复用，跳过了 system prompt 的 attention 计算——相当于一段固定前缀只计费一次。
* **无缝切换**：`create_llm_backend("auto")` 先尝试启动 VLLM，显存不够或没装 vllm 库就自动退回 DeepSeek API。上游代码不需要感知底层用的是什么。

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

* **VLLM 自部署推理 + 前缀缓存**：
    * **是什么**：用自己微调好的 14B 模型 + LoRA 做推理，不再每次调用都走 API。
    * **省钱的核心叫前缀缓存（VLLM `enable_prefix_caching`）**：同一个模型反复收到同样的 system prompt 开头时，VLLM 自动复用之前算好的 KV-cache，不重复计算固定前缀的 attention——这就解释了 README 开头提到的"Prompt 缓存"到底是怎么落的代码。
    * **双模式兜底**：`model/utils/vllm_engine.py` 提供了 `create_llm_backend()` 工厂函数，先试着连本地的 VLLM 实例，连不上或显存不够就自动切 DeepSeek API，上游调用方无感知。
    * **怎么启用**：在 `config.yaml` 里设 `llm.backend: "vllm"`，然后确保 `saved_loras/` 下有四个 LoRA 目录即可。训练产出的权重能直接挂上去用，不需要额外转换。
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
