# 当前项目概览

当前架构、模块职责与运行契约分别维护于：

- [模块职责](MODULAR_ARCHITECTURE.md)
- [Agent 运行时](AGENT_RUNTIME.md)
- [项目知识记忆](LONG_TERM_MEMORY.md)
- [文档结构与导航](PAGEINDEX_IMPLEMENTATION.md)
- [文本因果证据](CAUSAL_EVIDENCE.md)

任务由共享 LangGraph 流程推进。CLI、HTTP 和 Kafka 调用同一个 AgentService，
SQLite 保存运行状态与检查点。原文、摘要与图向量继续由独立知识库保存。
训练、确定性 SCM 评测与在线执行分开，不在启动服务时训练或评测模型。

本项目当前面向单机/受信小团队，不提供多机器共享存储、多租户权限或任意外部副作用撤销。
本地离线测试不能证明真实模型质量、检索收益或分布式部署可靠性。
历史设计讨论保留于 [历史 README](archive/README_before_modularization.md)。
