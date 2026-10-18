# 单机 Agent：运行、恢复与边界

当前以 LangGraph 推进任务，CLI、HTTP 与 Kafka 作业入口调用同一个 `AgentService`。
旧阶段 Worker 留作显式兼容/排空入口；容器默认启动新的 `agent_runtime.worker`。
没有部署任何服务、修改已有知识库结构或调用真实模型。

## 模块与存储

模块职责见 [MODULAR_ARCHITECTURE.md](MODULAR_ARCHITECTURE.md)。
任务权威在 `.runtime/runs.sqlite`：生命周期、租约、运行版本、预算、事件、工具记录及待发送消息。
配对的 `runs-checkpoints.sqlite` 保存实际图检查点。两者都在 D 盘项目目录内，均被 Git 忽略。
知识原文/摘要/向量继续由 `.memory/rag.sqlite` 保存，不使用 Redis TTL 作为恢复依据。

状态：`queued → running → waiting_input / succeeded / failed / cancelled / timed_out`。
每次运行有独立 ID，任务保留 ID 和依赖；只有全部完成才生成成功结果。
同一运行只有一个有效拥有者；运行库按 scheduler.max_running_runs 限制同时运行数，租约 30 秒、每 5 秒续期。
旧执行者的结果和检查点写入均受租约及 epoch 校验；取消或回滚后不能覆盖新状态。
新执行者可以在旧租约失效后恢复，但单机硬件、磁盘及 SQLite 仍是可用性边界。

运行保存原文和向量快照、知识 revision、模型相关配置及提示词哈希。
一次运行期间不刷新成新的知识版本；重启时重建该读取视图。
影响推理的配置或提示词已变化时明确拒绝续跑，需提交新运行；凭据轮换不改变配置哈希。
快照按运行保存，磁盘/读取成本随语料与运行数量增加，尚无吞吐或存储优化收益结论。

## 在 D 盘运行

已验证 Python 3.13.9；容器声明与此对齐到 Python 3.13，尚未实测构建。
使用项目现有 D 盘虚拟环境，不在 C 盘创建项目代码或新增环境。

```powershell
Set-Location 'D:\project(now)'
$env:MY_RAG_CONFIG = 'D:\project(now)\config.yaml'
$env:TEMP = 'D:\project(now)\outputs\temp'
$env:TMP = $env:TEMP
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:DEEPSEEK_API_KEY = '<模型凭据>'
```

`.env.example` 只列出变量；本项目不自动加载 `.env`。请在启动进程的环境中设置。
模型、语料必须自行准备；运行入口不会凭空生成证据。
新环境安装 `requirements-dev.txt`，可使用 `-c requirements-runtime.constraints.txt` 固定已验证的直接依赖版本。
该约束文件不是完整的传递依赖锁，本机继承全局包的环境也不等于独立安装验收。
当前本机没有 FlagEmbedding、redisvl 的发行包元数据，因而未将它们伪装为已安装并验证的版本。
真实 BGE-M3 检索需先安装对应依赖及准备模型；默认本地运行时不要求 RedisVL。

```powershell
.\.venv\Scripts\python.exe -m pip install -e . --no-deps --no-build-isolation --cache-dir outputs/pip-cache
.\.venv\Scripts\python.exe -m agent_runtime run '请分析我的劳动争议问题'
.\.venv\Scripts\python.exe -m agent_runtime status <run_id>
.\.venv\Scripts\python.exe -m agent_runtime events <run_id>
.\.venv\Scripts\python.exe -m agent_runtime cancel <run_id>
.\.venv\Scripts\python.exe -m agent_runtime resume <run_id> '用户补充的事实'
.\.venv\Scripts\python.exe -m agent_runtime retry <run_id>
.\.venv\Scripts\python.exe -m agent_runtime checkpoints <run_id>
.\.venv\Scripts\python.exe -m agent_runtime rewind <run_id> <checkpoint_id>
.\.venv\Scripts\python.exe -m agent_runtime recover
```

安装后也可使用 `.venv\Scripts\legal-agent.exe`。推荐从本项目的可编辑安装运行。
分发 wheel 时需通过 `MY_RAG_CONFIG` 指定保存在 D 盘的配置文件。
默认查询入口的闲聊、简单问答和复杂任务均进入该服务，使用共享前置路由与预算。
保留报告/观察字段，并提供 run_id、run_status；等待输入、失败及超时明确返回各自状态。
runtime.enabled=false 显式保留历史入口，供兼容使用，不具备新运行时保证。
工具策略、并行批次、角色映射与模型拥有者的配置和边界见 [PARALLEL_AGENT.md](PARALLEL_AGENT.md)。

完全离线的恢复示例：`python -m agent_runtime.demo`。它使用真实 SQLite 检查点与确定性工具，不使用模型、Kafka 或 Docker。

## HTTP 与 Kafka

```powershell
$env:AGENT_API_TOKEN = '<小团队共享的随机令牌>'
.\.venv\Scripts\python.exe -m agent_runtime serve --host 127.0.0.1 --port 8000
```

除 `/ready` 外，接口要求 `Authorization: Bearer <token>`。
提供 `/runs` 提交/列表、`/runs/{id}` 状态与结果，以及事件、取消、恢复、检查点和回滚接口。
`/ready` 检查运行库和本地执行线程；不对模型供应商可用性作保证。
默认后台扫描排队/租约失效任务。共享令牌适用于同一受信团队，不提供用户级权限或项目 ACL。
事件接口分页读取；没有引入额外的监控平台，也不把它描述为 Prometheus 服务。

Kafka 模式的 API 使用 `serve --kafka`，同主机另运行 `python -m agent_runtime.worker`。
也可用 `run '问题' --kafka` 提交。配置的 topic 为 `topic.agent.pending` 和 `topic.agent.result`。
消息：`{version:1,event_id,run_id,epoch}`。结果消息通知调用者按 ID/epoch 读取结果，并非复制整份报告。
消息与待执行状态先原子记录到运行库；发送得到 broker 确认后才标记已发送。
重投依靠状态与执行记录去重；不声称跨 Kafka、SQLite、模型供应商实现 exactly-once。
本地模式不需要 Kafka/Redis；Kafka 进程必须访问同主机同一份运行/检查点库。

旧五阶段 topic 不接受新协议。只在明确排空旧任务时设置 `AGENT_RUNTIME=0` 使用历史容器入口。
K8s 文件是旧链路示例，没有单机共享存储条件，不能用于部署新运行时。
本次修复其多任务提前完成和 DLQ 失败仍提交位移的问题，但旧链路不具备新运行时的恢复保证。

## 工具、预算与回滚

默认计算由模型给出结构化四则运算计划，程序以 Decimal 执行、金额保留两位小数，并保留公式与假设。
程序校验算式，不证明法律适用性或输入事实真实；缺失用户事实由 Grader 请求用户补充。
用户事实与原文证据分开保存，不送入 CHAIN 伪装为原文。

`RunRequest.allow_code=true` 才在新运行时启用生成代码。代码仅在 Docker 中运行：断网、非 root、
只读根文件系统、CPU/内存/PID 限额和输出上限；通过 Docker exec 传递，不开放 HTTP 端口。
每次执行使用新 Python 进程，变量不跨计算保留。超时或取消终止容器；Docker 不可用时明确失败。
构建命令：`docker build -f multiple-search/legal_sandbox/Dockerfile -t legal-sandbox:v2 .`。
`run_in_process` 仅为显式注入的可信离线测试函数，不是隔离环境，在线工具不自动调用它。

默认预算为整次运行 300 秒、40 次模型请求、80 个节点步骤、100000 文本 token 上界。
文本上界以输入 UTF-8 字节数加预留输出 token 计算，保守计费；不等于供应商计费 tokens，未含协议开销。
传输最多额外重试一次，也消耗模型调用预算；模型请求默认最多 30 秒并受剩余 deadline 限制。
取消是协作式的：正在进行的模型 HTTP 请求可能等到超时返回，但其结果不能覆盖取消状态。
等待用户输入也受原 deadline 限制，恢复不会重置预算。需要更长时间时提交新运行。

工具完成记录先于图节点检查点提交；已记录的只读工具结果在重放时复用。
若请求在供应商生效后、结果记录前中断，可能产生再次调用和再次计费，不能撤销已经发生的模型调用。
本期未注册外部写入工具；工具台账为中断写入保留 outcome_unknown 状态，禁止盲目重试或回滚。
未来增加外部写入，必须先逐项定义授权、幂等键、结果核对与补偿，不能把检查点恢复当作事务撤销。

## 备份与恢复

```powershell
python -m agent_runtime backup 'D:\project(now)\outputs\backup-001'
python -m agent_runtime restore 'D:\project(now)\outputs\backup-001' 'D:\project(now)\outputs\restored-001'
python -m agent_runtime --db 'D:\project(now)\outputs\restored-001\runs.sqlite' recover
```

备份要求没有运行中的有效租约，并阻止新执行者在复制期间启动。
使用 SQLite backup API 保存运行库和检查点库，最后发布带 SHA256 的 manifest。
恢复校验哈希和数据库完整性，只写入新目录，不覆盖已有库。
备份包含运行固定的证据视图；项目知识库 `.memory/rag.sqlite`、模型文件及凭据需另外备份。
运行和检查点尚未自动清理；长期使用需要监测磁盘，并在另行批准后制定保留/清理策略。

## 验证边界

统一入口：`python tools/ci.py`。新增测试覆盖真实 LangGraph、SQLite、HTTP TestClient 和隔离子进程；
模型、Kafka 与 Docker 使用确定性替身。跨进程用例在工具结果落盘后、图检查点完成前执行 `os._exit(17)`，
模拟进程消失，再恢复租约并检查工具只实际执行一次。
此前运行时版本完整套件通过 531 项测试，五类工程检查均通过；历史命令、输出及限制见
[验证记录](validation/agent_runtime_2026-10-18.txt)。
没有真实模型质量评测、线上 Kafka/Redis 联机测试、Docker 构建或 GPU 验证；没有既定独立 lint/类型检查流程。
