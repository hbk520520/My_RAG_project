# PA-Tool 工具名称对齐

状态：代码已接入 Replanner 和文档导航，默认不配置映射。未训练、未调用真实模型生成映射，未验证模型质量提升。

依据首尔大学数据科学研究生院 Jonggeun Lee 等人的
[论文 v3](https://arxiv.org/html/2510.07248v3)，实现三步：目标模型生成名称候选、
按字符 Levenshtein 距离计算 peakedness、选择聚集程度最高的名称。
每个组件先在温度 0 下获得参考名称，再按默认温度 0.4 采样 32 次；
距离阈值为 alpha × 候选最大字符长度，默认 alpha=0.2，平分时选离参考名最近者。
重复采样保留频数；重名时较高 peakedness 的组件锁定名称，其他组件选择后续候选。
与未改名字段冲突的候选不采用，候选耗尽则显式报告，不拼接虚构后缀或做模糊执行匹配。
作者仓库当前只有项目页面，未公开可直接调用的算法包；本项目用标准库实现上述命名步骤，无新增依赖。

## 使用

在实际目标模型的配置下，保存候选并构建映射：

```powershell
.\.venv\Scripts\python.exe -m model.tool_schema --role replanner `
  --samples D:/data/replanner-name-samples.json `
  --output 'D:/project(now)/outputs/tool-schema/replanner.json'
```

`--samples` 不调用模型。文件为以 `fields.task_desc`、`tools.GRAPH_TRAVERSAL` 等组件标识为键的 JSON 对象，
每项包含 `reference`（温度 0 名称）和 `samples`（采样名称列表），必须覆盖该角色的全部组件。
组件清单在 `agent_tools/schema_alignment.py` 的 CATALOGS；不能把手写别名当作真实模型的采样结果。

需要真实命名采样时，把 `--samples ...` 换为 `--generate`。
此入口只接受显式 vllm 后端和已存在的本地完整基座，通过现有模型角色接口调用同一基座及 LoRA；
不允许隐式下载，依赖/权重缺失直接报错。默认每个组件调用 33 次：Replanner 198 次，navigator 363 次。
这是推理采样，不是训练；本次没有执行。可用 `--count`、`--temperature`、`--alpha` 调整，名称只接受单个标识符。
Python 调用方也可通过 `sample_names(role, call, ...)` 注入目标模型回调。

在 config.yaml 中启用已有映射，相对路径以配置文件所在目录为基准：

```yaml
tool_schema_alignment:
  profiles:
    replanner: outputs/tool-schema/replanner.json
    navigator: outputs/tool-schema/navigator.json
```

可以只配置其中一个角色。删除该条目即可拔除；默认 `profiles: {}` 保留原来的提示和接口。
映射绑定当前角色的后端、基座与适配器配置；更换模型或 Heretic 行为适配器后重新采样，不能直接复用旧模型映射。
传入的已有样本应来自该目标模型；配置身份不代替真实权重指纹或采样来源验证。

## 实现边界

- Replanner：对齐两个检索引擎名称及 task_queue/task_desc/engine/rationale；其余任务字段保持现状。
- 文档导航：对齐五个动作及六个参数。返回值解码后继续使用原 Action 校验和 DocumentReader 执行。
- 静态说明和 JSON Schema 使用同一映射；案件文本、问题、失败日志、目录及原文内容不做字符串替换。
- 任务队列、工具实现、来源元数据和导航 trace 继续保留原名称，模型历史保留其实际生成的名称。
- 同步与统一 Kafka 运行时共用这两处接口；历史 Redis 阶段 Worker 的独立重规划入口未扩展。
- 映射文件内容参与既有运行配置指纹，避免恢复任务时混用两套命名；未启用时不改变既有指纹。

不修改模型权重，不接入新工具执行框架，不添加裁判模型或额外安全组件。
无需训练不等于无需模型采样；当前没有目标小模型的真实映射或修复效果结论。
工程验证与未执行项见 [验证记录](validation/tool_schema_alignment_2026-11-07.txt)。
