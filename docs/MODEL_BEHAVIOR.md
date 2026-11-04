# 本地模型行为优化

状态：接入组件已实现，默认不执行、不挂载。没有运行真实权重修改或验证质量收益。

`model/behavior.py` 只准备本地配置和显式启动独立进程；复用
[Heretic v2.0.0](https://github.com/p-e-w/heretic/releases/tag/v2.0.0) 的
KeywordRate、KLDivergence、ARA，不复制其算法，不增加在线依赖。
可分别选择减少迎合（nosyco）、减少空泛套话（noslop），或一起处理（both）。
这个最小版本只优化指定表达的出现率，并约束保留样本上的输出分布变化。
关键词减少不等于减少所有冗余或无依据附和，KL 接近也不等于任务能力保持。
默认中文短语只是可替换的起点，不惩罚承认错误、信息缺失或必要的不确定性表达。
不使用拒答抑制目标，最终 Generator API 和既有工作流保持现状。

## 准备与显式运行

使用独立环境安装 `requirements-behavior.txt`，固定 `heretic-llm==2.0.0`。
本组件不会自动安装、下载或升级任何依赖；确需下载时先说明名称、用途、大小和可省略性，取得同意后再做。
不要把其依赖并入 API、Unsloth 或 vLLM 环境。上游许可为 AGPL-3.0-or-later，源码入口见
[上游仓库](https://github.com/p-e-w/heretic/tree/v2.0.0)。

需要准备完整本地模型（包含 config.json、权重及 tokenizer）和三个 UTF-8 文本文件，
每个非空行是一条完整提示，包含角色、任务、输入事实和预期输出格式；不能把 JSONL 对话当作纯文本直接传入。

- good.txt：能诱导模型给出简洁、依据事实的响应，并保留正确纠错、必要解释和合法 JSON 的任务。
- bad.txt：能诱导同类任务出现目标迎合或套话的对比提示；不是错误答案标签。
- evaluation.txt：衡量目标表达出现率的优化样本；参与搜索，**不是留出测试集**。

建议对比提示使用相同任务信息、只改变行为诱导，并用相同任务分布的证据冲突和失败恢复场景。
是否能可靠分离目标行为仍需要实际检查。任务质量验收另用来源/任务隔离的留出集。

以下路径为使用示例，需要替换为实际本地文件；本次未创建这些模型或提示数据。

```powershell
.\.venv\Scripts\python.exe -m model.behavior `
  --model D:/models/replanner-base `
  --good D:/data/behavior/good.txt `
  --bad D:/data/behavior/bad.txt `
  --evaluation D:/data/behavior/evaluation.txt `
  --output 'D:/project(now)/outputs/behavior/replanner-01' `
  --goal both --trials 10
```

默认只生成配置、提示快照和 manifest，不加载 Heretic、Torch 或模型。
`--trials` 必须显式指定；示例的 10 次只是搜索预算，不是经过验证的最优值。
`--markers D:/data/behavior/markers.json` 可用非空 JSON 字符串数组替换全部表达词表。
`--max-tokens` 默认 2048；不要通过截断必要计划来制造“更简洁”的假象。

要实际处理，在上述命令中增加 `--run --heretic-python D:/envs/heretic/Scripts/python.exe`，并使用新的输出目录。
启动时核对上游版本，禁止 Hugging Face 在线获取，所有提示来源均被本地快照替换。
临时文件、检查点与主要模型缓存位于本次输出目录；子进程中的设置不影响父进程。
缺依赖、缺本地模型内容、非零退出或没有导出适配器都会显式失败。

导出位置为 `adapter/`；原模型和旧运行目录不覆盖，不自动上传或启用。
自动导出的是上游排序后 Pareto 候选的索引 0，不称为任务质量最优模型。
manifest 保存上游版本、参数配置哈希、提示哈希和运行状态；模型哈希只覆盖 config.json，
**不包含完整权重指纹**，因此不宣称完整模型可复现性。
即使导出成功也标记 `exported_unvalidated`、`quality_validated=false`。

## 接入与拔除

先在留出任务上对照原模型，检查完整任务成功率、错误预设处理、正确纠错、
证据缺失处理、JSON/任务契约、必要信息完整性及 token 数。工程测试不能替代这些验收。

验收后复用现有 `vllm.lora_modules` 和 `vllm.role_adapters`，无需新增在线插件框架。
在原配置的对应映射中增加/修改以下条目，其余角色保持原配置：

```yaml
vllm:
  lora_modules:
    replanner_behavior: 'D:/project(now)/outputs/behavior/replanner-01/adapter'
  role_adapters:
    replanner: replanner_behavior
```

此片段不是完整配置，不要用它覆盖现有 vllm 配置。推理基座必须与处理时的完整模型一致；
重启推理拥有者后生效。拔除时恢复 `role_adapters.replanner` 的原值并移除新增模块条目，再重启。
不修改共享基座，不自动给 Planner、Reasoner 或 Generator 启用。

当前每次请求只选择一个 LoRA；行为适配器不能与已有 replanner_grpo 自动叠加。
若需要保留已有任务微调能力，应另行准备合并后的完整模型，处理后验证并单独部署；
本组件不负责合并或转换已有适配器，也不改变其他角色的基座。

验证命令、替身测试范围与未执行项见 [验证记录](validation/model_behavior_2026-11-04.txt)。
