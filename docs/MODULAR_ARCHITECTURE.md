# 模块职责与运行入口

项目根目录在 D 盘。运行数据与测试临时文件使用根目录下的 `.runtime/`、`.memory/`、`outputs/`。

| 模块 | 职责 |
|---|---|
| `legal_agent/operators.py` | 模型角色与结构化模型输出 |
| `legal_agent/reasoning.py` | 原文检索、CHAIN 与充分性判断 |
| `legal_agent/state.py` | 图状态 |
| `legal_agent/nodes.py` | 状态转换节点 |
| `legal_agent/workflow.py` | LangGraph 拓扑与依赖注入 |
| `legal_agent/routing.py` / `cache.py` | 意图路由与兼容的缓存包装 |
| `agent_tools/sandbox/` | 代码执行与容器生命周期 |
| `dataset/` | 知识存储、文档结构、摘要与检索 |
| `causal_evidence/` | 文本因果支持验证与时间权重 |
| `benchmark_causal/` / `model/` | 独立的评测与训练；在线执行不启动训练 |

历史 `query.py`、`multiple-search/soul.py` 与沙箱导入入口继续可用，实现只维护在包内。
新代码使用包导入；离线流程示例用 `python -m legal_agent.demo`。
入口包装器只负责引导历史脚本，不在包内复制实现。

验证：`python tools/ci.py`。模块移动阶段保持原执行行为，恢复、任务管理和工具隔离随后单独验收。
