# 法律 RAG Agent

面向单机或受信小团队的法律检索与任务执行项目。代码与本地运行文件保存在 D 盘。

当前实现：共享 LangGraph 流程、SQLite 任务与检查点、等待用户补充/恢复/取消/回滚、
Kafka 作业适配、Decimal 计算与可选代码执行（默认 Docker，可选一次性 SPACE 后端）。
知识侧保留 FAISS/igraph、项目知识快照、PageIndex 思路的文档导航，以及 CHAIN 文本因果支持验证。
任务支持按证据需求组合工具、有界 DAG 并行、分支结果重放和可配置资源准入。

## 从这里开始

- [模块职责](docs/MODULAR_ARCHITECTURE.md)
- [Agent 运行、配置、恢复及边界](docs/AGENT_RUNTIME.md)
- [工具策略、并行和共享资源](docs/PARALLEL_AGENT.md)
- [沙箱后端：Docker / SPACE](docs/SANDBOX_BACKENDS.md)
- [可选批量编码与快照索引缓存（默认关闭）](docs/PERFORMANCE_PATHS.md)
- [接口契约](docs/CONTRACTS.md)
- [知识长期记忆](docs/LONG_TERM_MEMORY.md)
- [PageIndex 已实现能力](docs/PAGEINDEX_IMPLEMENTATION.md)
- [CHAIN 实现与实验边界](docs/CAUSAL_EVIDENCE.md)
- [简历与面试提醒](docs/RESUME_REMINDERS.md)

## D 盘运行

已验证 Python 3.13.9。新环境先安装 requirements-dev.txt；训练依赖仍单独维护。
现有环境可进行本地可编辑安装，提供 legal-agent 命令：

```powershell
Set-Location 'D:\project(now)'
.\.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation --cache-dir outputs/pip-cache
.\tools\run.ps1 demo
.\tools\run.ps1 ci
```

`demo` 完全离线，使用真实检查点演示等待、恢复与回滚；`ci` 运行入口检查、语法检查、测试、离线示例与跨进程实验。
启动真实任务 API 前，在当前进程设置 DEEPSEEK_API_KEY、AGENT_API_TOKEN，导入资料并准备检索模型，再运行：

```powershell
.\tools\run.ps1 serve
```

脚本把临时目录、包缓存和模型缓存定位到 D 盘，不自动读取 .env。
需要 Kafka 时另运行 `tools/run.ps1 worker`，与 API 使用同一主机、同一份运行库。
CLI 的提交、状态、恢复与备份命令见运行文档。

## 代码位置

| 目录 | 职责 |
|---|---|
| legal_agent/ | 模型操作、检索推理、状态节点、流程、路由 |
| agent_runtime/ | 持久任务服务与入口适配 |
| agent_tools/ | 确定性计算和隔离代码执行 |
| dataset/ | 原文、向量、语义图、文档与摘要 |
| causal_evidence/ | 文本因果验证、校准接口与 CTVF |
| benchmark_causal/ | 确定性 SCM 与离线评测 |
| model/ | 独立的训练与模型后端代码 |
| tests/、tools/ | 回归测试与工程入口 |

旧 query.py、multiple-search/soul.py 和沙箱导入路径继续兼容；新代码通过包导入。
历史阶段 Worker 仅供兼容排空，默认执行入口使用统一运行时。

## 已验证范围

本地工程验证使用真实 LangGraph、SQLite、HTTP TestClient 与隔离子进程；模型、Kafka 和 Docker 使用替身。
SPACE 后端使用可注入 HTTP 传输做离线生命周期/恢复测试；**没有真实 SPACE 部署联调**。
本次 SPACE 后续优化按要求未运行回归，历史通过记录不代表当前修改已验证。
新增批量编码与快照索引缓存默认关闭，仅做静态检查；未运行测试、数值对照或性能基准。
详细命令和结果见 docs/validation/agent_runtime_2026-10-18.txt 与 docs/SANDBOX_BACKENDS.md。
没有真实模型质量、线上 Kafka/Redis、Docker 构建或 GPU 的验收结果。
当前虚拟环境继承全局包，尚未证明干净环境可完整重装；依赖约束文件只固定有元数据的已验证直接依赖。

CHAIN 分数表示文本支持度与时间有效性，不是真实因果效应；PageIndex 导航摘要不充当原文证据。
检查点回滚不会撤销已经发生的外部操作。本期未注册外部写入工具，不宣称分布式 exactly-once。

早期设计与训练笔记保留于 [历史 README](docs/archive/README_before_modularization.md)，不作为当前能力或效果证据。
