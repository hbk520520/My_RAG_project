# 模块职责与运行入口

项目根目录在 D 盘。运行数据与测试临时文件使用根目录下的 `.runtime/`、`.memory/`、`outputs/`。

| 模块 | 职责 |
|---|---|
| `legal_agent/operators.py` | 模型角色与结构化模型输出 |
| `legal_agent/reasoning.py` | 原文检索、CHAIN 与充分性判断 |
| `legal_agent/state.py` | 图状态 |
| `legal_agent/nodes.py` | 状态转换节点 |
| `legal_agent/workflow.py` | LangGraph 拓扑与依赖注入 |
| `legal_agent/scheduling.py` | DAG 就绪任务、有界分支执行、结果合并和局部重规划 |
| `legal_agent/gateway.py` / `evidence.py` | 统一前置分类、独立于因果分析的原文来源记录 |
| `legal_agent/routing.py` / `cache.py` | 意图路由与兼容的缓存包装 |
| `agent_tools/sandbox/` | 代码执行与容器生命周期 |
| `agent_tools/calculator.py` | 经过校验的 Decimal 四则运算计划 |
| `agent_runtime/contracts.py` / `context.py` | 请求/任务契约、运行预算与工具上下文 |
| `agent_runtime/store.py` / `checkpointer.py` | 租约、检查点、执行台账与待发送消息 |
| `agent_runtime/service.py` | 统一执行、恢复、取消、回滚和备份 |
| `agent_runtime/resources.py` / `models.py` | 同机资源准入、加速器模型拥有者及共享编码器 |
| `agent_runtime/api.py` / `transport.py` / `worker.py` | HTTP 与 Kafka 适配 |
| `dataset/` | 知识存储、文档结构、摘要与检索 |
| `dataset/encoding.py` / `index_cache.py` | 可选批量编码与服务实例内的序列化 CPU 索引缓存；默认关闭 |
| `dataset/material_routing.py` / `material_rules.json` | rule-engine 资料用途与加工深度分流、流式来源报告 |
| `dataset/pdf_intake.py` | 原文页轻量导入、核心长文目录、可选 PDF 检查和待提取边界 |
| `causal_evidence/` | 文本因果支持验证与时间权重 |
| `benchmark_causal/` / `model/` | 独立的评测与训练；在线执行不启动训练 |

历史 `query.py`、`multiple-search/soul.py` 与沙箱导入入口继续可用，实现只维护在包内。
新代码使用包导入；离线流程示例用 `python -m legal_agent.demo`。
入口包装器只负责引导历史脚本，不在包内复制实现。

当前运行、恢复与工具隔离契约见 [AGENT_RUNTIME.md](AGENT_RUNTIME.md)。
可选性能路径、原实现保留方式及尚未执行的验收见 [PERFORMANCE_PATHS.md](PERFORMANCE_PATHS.md)。
在 Windows 用 `tools/run.ps1 ci` 统一配置 D 盘临时目录和缓存，再执行现有工程检查。
