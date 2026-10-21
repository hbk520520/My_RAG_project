# 合成数据组件

`model.data.synthetic` 独立于训练消费者：默认离线，只使用标准库与已有 Pydantic。
不加载模型、不读取 API 凭据、不启动训练，也不修改在线 Agent、原始证据库或已有 Evol-Instruct 入口。

## 数据与验证边界

- **已验证集**：虚构环境中的工具、依赖、权限、状态和结果可由程序检查；发布前实际回放参考轨迹。
  `verified` 只表示这些模拟规则内的验证，不表示法律结论正确或模型已具备业务能力。
- **待审核集**：法律适用、因果判断和开放式推理题目，来源可为生成资料或用户提供的种子资料。
  默认输出无答案的题目草稿；注入生成器后可附带 `proposed_label`，仍然是 pending。
  格式和来源 ID 检查不能证明语义正确；组件没有自动晋升或人工审核工作流。
- **可选轨迹集**：独立导出程序参考轨迹、有效但冗余、伪造观测、无效工具操作和错误最终答案。
  它们是 `scripted_reference`，不是模型或真实 Agent 的运行记录。

所有生成任务保留 `synthetic=true`；种子来源另标 `provided_seed`，不把提供的来源本身冒称为合成资料。
完整行含环境和标签，不能直接整行作为训练输入。`input` 才是可公开给求解者的任务部分；
待审核样本不自动进入已验证集或训练集。

## 支持的场景

| 基础任务 | 可执行环境中的行为 |
|---|---|
| lookup | 读取数值或文本状态，不额外推断 |
| composition | 多记录汇总，写入当前任务的内存结果区 |
| dependency | 沿 next_id 发现后续记录，前置读取完成后才能访问下一个节点 |
| parallel | 两个 researcher 读取各自分支，交接给 coordinator 后汇合 |
| recovery | 首次读取暂时失败，随后重试并汇总 |
| constraints | 信息缺失须请求补充；权限不足或预算耗尽须停止 |

协作场景模拟角色分工与交接，独立分支允许不同执行顺序；不启动真实多 Agent 或并发调度器。
所有写入只发生在任务实例的内存字典中，不执行外部 SQL、浏览器操作、任意代码或真实服务写入。

可叠加文档内复述、过期版本、低优先级来源冲突和同级未决冲突；不通过修改正确标签制造噪声。
来源优先级与 active 标记是明确声明的模拟规则，不代表现实中的法律效力判断。
无关天气工具是合法干扰项；组件不实现任意业务文档的相关性判定。

## 采样与切分

任务混合权重、实体／来源组的变体数量、记录数／依赖长度分别配置。
变体数量与复杂度采用有限范围的幂律权重采样，不宣称复现真实业务分布或预训练知识偏好。
同一组的基础事实保持一致；缺失、冲突等变换会记录，变换后重新计算目标行为。

先按稳定的组 ID 哈希确定 train / validation / test，再生成组内变体。
OOD 使用独立实体组、改名工具、改写问题和 sources→rows 的 JSON 返回字段变化。
这只测试当前生成器的特定结构变化，不等于独立生成规则、CSV/HTML 解析或业务分布外泛化。
小规模哈希切分不保证每个集合非空或比例精确；不要混合不同切分配置的数据版本后声称无泄漏。

`group_exposures.jsonl` 分别记录组的任务数量、来源在环境中出现次数和参考轨迹实际读取次数。
这些不是模型训练时的 token 曝光统计。文档内部复述不会冒充多份独立来源。
用户种子资料若共用来源 ID 或完全相同的去首尾空白文本，必须归入同一组；这不是语义近似去重。

参考 [Kim 等的受控训练数据研究](https://aclanthology.org/2026.acl-long.1064/) 的复述、冲突和偏斜分布思路。
默认复述率、冲突率均为零，以便建立干净基线；具体比例是实验变量，不将论文中的某个比例直接当作通用参数。

## CLI：始终离线

在项目根目录运行，输出必须是新目录；Windows 下限制在 D 盘：

```powershell
.\.venv\Scripts\python.exe -m model.data.synthetic --output 'D:\project(now)\outputs\synthetic-v2\my-baseline'
```

可用 `--config D:/.../config.json` 配置，例如：

```json
{
  "seed": 42,
  "groups": 40,
  "ood_groups": 8,
  "max_variants": 4,
  "max_items": 12,
  "family_weights": {"lookup": 1, "composition": 1, "dependency": 1, "parallel": 1, "recovery": 1, "constraints": 1},
  "repetition_rate": 0,
  "conflict_rate": 0,
  "pending_per_group": 1,
  "include_trajectories": false,
  "candidate_call_limit": 0
}
```

`--trajectories` 显式开启轨迹阶段；`--seeds D:/.../seeds.json` 接受以下 JSON 数组，种子仅生成待审样本：

```json
[{"id":"document-group-a","sources":[{"id":"document-a","text":"待使用的来源原文"}]}]
```

## 可插拔生成器

Python API 为 `export_dataset(config, output, provider=None, seed_groups=())`。
`CandidateProvider(name, generate)` 接受模型／实现版本名与 `generate(request) -> dict` 回调。
request 包含任务种类、来源、生成指令和 Draft JSON Schema；返回 query、可选 proposed_answer 与 source_ids。
任何 provider 输出都进入 pending；返回 verification 等额外字段或引用未知来源会报错。

只有显式注入 provider 且 `candidate_call_limit>0` 才调用，CLI 不动态加载外部函数或读取凭据。
先校验种子分组，再生成；可选调用额度优先分配给种子材料，剩余额度处理模拟材料。
额度耗尽后其余候选按预定离线模板生成，provider 字段明确记录来源，不是错误时的静默兜底。
provider 报错直接终止，组件不自行重试、丢弃样本或切换模型；超时和传输策略由调用方实现。
call_limit 限制请求次数，不代表 token 或货币预算。

## 输出与验证

每个 split 分别存放 verified.jsonl、pending.jsonl；显式开启轨迹时另有 trajectories.jsonl 和 ranking.jsonl。
排序对只比较同一任务的非平分候选，不实现评分模型损失、SFT、DPO 或 GRPO。
评分分别记录答案正确、步骤有效、最终内存状态正确；全部满足才给予成功分，再扣固定的任务内调用成本。
初始成本权重为 0.1，版本写入 reward_version；这是可解释的工程基线，不是经过优化的训练奖励。

manifest 保存配置、Python/Pydantic 版本、生成代码哈希、输出文件哈希、实际分布和 provider 调用次数。
临时目录完整写入后才发布，失败不留下可被误认为完整数据集的最终目录，既有输出禁止覆盖。
模拟组按组生成和流式导出；用户种子分组检查需要额外保存来源 ID／文本哈希索引。

测试位于 `tests/test_synthetic_component.py`，实际命令与结果见
[验证记录](validation/synthetic_component_2026-10-21.txt)。
组件验证不等于训练收益；目标任务效果、不同生成规则与模型训练应另行设计对照和消融。
