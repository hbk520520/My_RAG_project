# 沙箱后端：Docker（默认）与 SPACE（可选）

本文记录 `agent_tools/sandbox/` 的实际实现边界。它不是 SPACE 的生产部署证明。

2026-10-19 后续优化：缓存恢复、清理状态、总预算及 JSON 输出限额已调整；按本轮要求
没有运行测试或真实 SPACE 联调。下文 44 个用例及 CI 通过记录对应优化前的版本，
不能作为本次修改已通过回归的证据。

## 当前结论

- Docker 仍是默认后端，`DockerSandboxManager` 的导入方式、`run_code_once` 签名和
  `calc_result / error / session_id / via` 返回字段保持兼容。
- SPACE 是默认关闭的可选后端，首期只支持一次性 Python 代码执行：
  一个 `run_code_once` 调用创建一个沙箱、创建一条命令、读取一条事件流，然后删除沙箱。
- 没有跨任务共享工作区，没有暂停/挂起恢复、快照回滚、预热池、文件上传或 E2B 兼容层。
- 未启用 SPACE 时不会发生 SPACE 远程连接、配置探测或额外服务依赖。
- SPACE 配置无效时明确报错，不回退到 Docker 或宿主机执行；`run_code_once` 返回
  `error_kind="configuration"`，节点和 Worker 不会据此改写代码或重复生成程序。
- 工具缓存命中时直接返回已记录的结果，不初始化沙箱管理器或连接 Docker/SPACE。
- 创建、命令和删除请求在发送前写入现有 effects 台账；成功执行后原始请求仍保留，
  清理失败会保留 `session_id / sandbox_id` 和可追踪日志。
- **尚未在真实 Linux/KVM SPACE 部署上联调**。本文中的“通过”仅指离线测试。

## 实际调用链

```text
legal_agent.nodes.node_execute_code
  → agent_runtime.resources.resource("tool.sandbox", tool_slots)
  → agent_tools.sandbox.executor.run_code_once(code, session_id)
  → agent_runtime.context.tool_call("sandbox.execute", ...)
  → RunContext.tool
  → RunStore.begin_effect / record_effect_request / end_effect
  → agent_tools.sandbox.manager.get_sandbox_manager()
       ├─ DockerSandboxManager（默认）
       └─ SpaceSandboxManager（backend=space）
  → SpaceSandboxManager：执行体内清理一次性沙箱，记录删除意图与 operation
  → AgentService.execute / Worker 对残留句柄调用 destroy_session（幂等包装）
```

Docker 管理器的既有语义不变：默认在线路径仍走 Docker；`run_in_process` 仅是可显式注入的
离线测试辅助函数，不是安全边界。SPACE 只在显式配置时替换执行后端，不改变节点、
资源许可或检查点机制。

## SPACE 上游契约（固定版本）

- 上游提交：`57c7c0c47a73373a974c5b542d180599795465da`（2026-10-18；release baseline 0.8.0）
- 接口面：native `/v1` M1；本次适配不覆盖 `platform`、ECW/E2B 或 WebSocket 交互接口。
- 核对方式：用 `git ls-remote https://github.com/perplexityai/SPACE.git HEAD` 确认该提交，
  并以浅克隆中的 `docs/versioning.md`、`docs/api/index.md`、`docs/first-sandbox.md`、
  `examples/first_sandbox.py`、`docs/native-api-hardening.md`、`docs/egress-profiles.md`
  和 `crates/libs/public/api/src/lib.rs` 的实际请求/响应结构为准。
- 使用的受支持操作子集：
  - `POST /v1/sandboxes`（202，返回 `sandbox_id` + `operation_id`）
  - `GET /v1/operations/{operation_id}`（`queued/running/succeeded/failed/canceled/expired`）
  - `GET /v1/sandboxes/{sandbox_id}`（生命周期状态）
  - `GET /v1/sandboxes/{sandbox_id}/processes?limit=1`（guest 进程服务就绪探测）
  - `POST /v1/sandboxes/{sandbox_id}/commands`（201，返回 `session_id`、`process_id`、`attach`）
  - `GET {data_plane}/v1/sessions/{session_id}/events`（SSE；连接即开始执行）
  - `DELETE /v1/sandboxes/{sandbox_id}`（202，返回删除 `operation_id`）
- 管理请求使用管理 bearer token；事件流使用命令返回的 session token。两者不混用。
- 当前 public model/示例把 `process_id` 作为命令响应字段；客户端要求它存在，并要求
  `process_exited.process_id` 与创建命令时返回的值一致。
- 只有声明需要 `Idempotency-Key` 的变更才去重；沙箱生命周期和命令创建的去重窗口为
  24 小时。适配器据此设置 `idempotency_window_seconds` 上限为 86400 秒。
- 模板必须是不可变摘要引用：`<name>@sha256:<12–64 位小写十六进制>`。适配器拒绝
  `seed:latest` 这类可变 tag，避免“同一次运行”在不同时间落到不同模板；部署应提供
  完整 64 位 digest。真实模板摘要属于部署输入，本次没有部署，因此仓库不硬编码某个 digest。

### 模板与 runner

guest 模板必须提供一个可用的 `python3`。适配器把版本化的 runner 源码
（`agent_tools/sandbox/space_runner.py`，`RUNNER_VERSION=legal-sandbox-space-runner/1`）
通过 `python3 -c` 传入，因此不需要人工修改客体环境或预置 `/app/runner.py`。runner 的语义与 Docker 使用的
`runner.py` 一致：优先读取代码里的 `result`，否则使用捕获的 stdout；输出上限由
命令参数传入；结果以单个 JSON 对象写到 stdout。runner 版本会写入副作用意图记录。

模板还必须由部署方保证：非零用户、只读根文件系统、CPU/内存/PID 限制、无额外挂载、
以及所需的网络隔离。适配器不会用编造的 SPACE 参数假装等价于 Docker 配置。

## 执行阶段与成功判据

```text
创建请求被接受（202）
  → 操作完成（operation=succeeded）
  → 沙箱就绪（sandbox=running）
  → 进程服务就绪（processes 列表接口可用）
  → 创建命令（201）
  → 连接 SSE（连接即开始执行）
  → 收到 process_exited
  → 解析 runner JSON
  → 记录删除意图并删除沙箱
  → 轮询删除 operation 成功
```

以下任一情况都不算成功：

- 只有 HTTP 200/201/202；
- 连接关闭但没有 `process_exited`；
- `process_exited` 的 `status` 不是 `exited`；
- `exit_code` 非 0；
- runner JSON 缺失、损坏、为空或包含执行错误；
- 收到 `error` 或 `resync_required` 帧；
- SSE 在帧中间结束、超过输出/帧/流上限。

SSE 解析器支持任意分片、多行 `data:`、CRLF、注释心跳和末尾无换行；事件流不重放，
所以不会把“重连后没收到旧输出”当成成功。

## 超时与限额

| 配置 | 作用 |
|---|---|
| `connect_timeout` | TCP/TLS 建连上限 |
| `request_timeout` | 单次管理请求的读取上限 |
| `operation_timeout` | 创建/就绪/删除各阶段的轮询上限 |
| `stream_timeout` | SSE 单次读取上限 |
| `command_timeout` | SPACE 服务端命令执行上限，按整秒发送 |
| `max_total_seconds` | 单个 `run_code_once` 的客户端总预算 |
| 任务 deadline | 运行库的 deadline 再收紧上述预算，并预留 2 秒 |
| `max_output_bytes` / `max_stderr_bytes` | stdout/stderr 上限 |
| `max_frame_bytes` / `max_stream_bytes` | SSE 单帧/整流上限 |
| `max_request_bytes` | 管理请求体上限（含 runner 与 base64 代码） |

创建、就绪、命令与事件流共用创建时确定的绝对截止时间，执行阶段不重新增加总预算。
传输层保留连接及单次空闲读取上限，并为整个请求/事件流设置截止时间；已连接 socket
到期后通过 shutdown 中断阻塞读取，响应和事件到达后还会再次检查预算。
响应关闭时取消对应定时器。实现不使用 `SIGALRM`；DNS 解析仍受操作系统行为约束。
本轮没有验证实际 Windows/Linux 网络超时行为。

`max_output_bytes` 限制解析后结果的 UTF-8 字节数。stdout 承载 runner JSON，其上限
另行考虑 JSON 转义和错误包络开销，避免中文及控制字符因编码膨胀而被误拒绝。
stderr、SSE 单帧及整流上限仍独立生效；传输大结果时需同时配置足够的帧和整流预算。

## 副作用、恢复与并发

SPACE 的 `sandbox.execute` 以 `read_only=False` 进入现有 effects 台账。Docker 的
既有台账语义保持不变。

发送创建沙箱、创建命令和删除沙箱前，适配器把以下内容写入现有 `effects.result` 的
`_intents` JSON 数组（不新增数据库表/列）：

- 首次请求时间、原始幂等键、HTTP 方法、路径、必要请求体；
- 后端、M1 契约版本、上游提交、部署身份、API/data-plane 源、模板摘要、runner 版本；
- 运行 ID、epoch、任务 ID、revision；
- 响应返回后的 `sandbox_id` / `operation_id` / `session_id` / `process_id`。

首次请求时间是幂等窗口的基准，后续补充响应 ID 不会延长窗口。成功执行也会保留这些
意图；`begin_effect` 返回缓存结果时会剥离私有的 `_intents`，不改变工具返回契约。
删除意图在 DELETE 发送前记录，删除 operation ID 返回后补充。创建响应只拿到
`sandbox_id` 而缺少 `operation_id` 时，适配器仍会记录并尝试清理这个已知沙箱。

不写入管理 token、attach token、连接凭据或重定向地址。命令请求体包含生成代码，
因此台账可能保存代码文本；它不应包含凭据，调用方也不要把凭据写入生成代码。

当请求可能已执行但客户端无法确定结果时（连接中断、响应丢失、流中断、缺少退出事件），
适配器抛出 `SandboxOutcomeUnknown`，执行器转换为现有
`agent_runtime.contracts.OutcomeUnknown`。这会导致运行以 `outcome_unknown` 失败，
不会把错误改写成普通代码错误，也不会触发节点自动改写代码或再次执行。

恢复时不会自动重放。`SpaceSandboxManager.reconcile_intent()` 校验已持久化的原始
请求：部署/模板/API 源是否仍匹配、是否仍在 24 小时幂等窗口内，以及 action 对应的
method/path/body 是否仍符合固定契约；超窗、换部署或请求形状变化时明确拒绝。
它返回原始 key/body 供人工对照，但不会发送请求。凭据轮换不改变运行语义：运行配置
哈希排除了 `api_token`，但旧 attach 凭据在幂等重放后可能失效，不能假设仍可用。

取消和 epoch 失效后，迟到的命令结果不会被提交；清理使用独立的 5 秒窗口，
即使上下文已失效也会尝试释放本任务拥有的沙箱。清理只操作本管理器创建并登记过的
沙箱句柄，不按全局列表或年龄批量删除其他任务的资源。清理前校验句柄中的 run、
sandbox 以及当前上下文的 run/task/project；跨任务或跨项目句柄会被拒绝。清理失败会
保留资源标识并记录错误，不会谎报“已释放”；`run_code_once` 成功时会在结果中附加
`cleanup_error` 并保留 `session_id`，只有清理被确认成功才清空句柄。
未知且没有本地成功删除记录的句柄会报告无法确认清理，`destroy_session` 包装返回 False，
让节点/Worker 保留句柄。进程重启后的所有权核对及实际资源回收仍需要依据持久化意图
单独处理；本次没有新增自动恢复清理器。

## 隔离与凭据边界

Docker 路径的现有控制：`network_mode=none`、只读根文件系统、非 root、`cap_drop=ALL`、
`no-new-privileges`、PID/内存/CPU 限制、输出上限。

SPACE 路径：

- 创建请求显式发送 `network_policy = {"version":"v1","egress":{"default_action":"deny",
  "allow":[],"deny":[]}}`。
- 但这只是请求；**实际 egress 强制由 SPACE 部署和模板保证**。SPACE 的 egress profile
  是允许规则，不能仅凭它推断“已禁止外网”。
- 因此启用 SPACE 还要求配置 `network_isolation` 记录：
  `mode=egress_deny_all`、具体部署标识、核对日期和证据指针。空值会让后端明确拒绝启用。
  这条记录是操作者的核对声明，不是代码对隔离的运行时证明。
- 管理 token 与 attach token 分开使用；事件流 URL 必须匹配配置的 data-plane origin 和
  精确 session 路径，attach 必须声明支持 `authorization_header` 且未过期。传输层不跟随
  重定向、不使用环境代理，避免把 token 发给不受信任地址。
- token 从 `${SPACE_API_TOKEN}` 等现有环境变量入口读取，不写入代码、日志、检查点或
  副作用结果。日志和意图只记录 sandbox/operation/session 等非秘密标识。
- 响应错误按 `code/retryable/request_id` 保留；404 只表示未找到，不作为执行成功或
  清理成功的证据。

## 配置示例

```yaml
sandbox:
  backend: "docker"          # docker | space
  space:
    enabled: false
    api_url: "${SPACE_API_URL}"
    data_plane_url: "${SPACE_DATA_PLANE_URL}"
    api_token: "${SPACE_API_TOKEN}"
    template: "${SPACE_TEMPLATE}"   # name@sha256:<64 hex>
    runtime: "fc"
    api_contract: "m1"
    vcpu: 1                  # 显式 SPACE 单位，不从 Docker quota 推导
    memory_mib: 512
    disk_mib: 1024
    command_timeout: 10      # 服务端执行上限
    stream_timeout: 45
    operation_timeout: 60
    max_total_seconds: 120
    network_isolation:
      mode: "egress_deny_all"
      verified_deployment: ""
      verified_at: ""
      evidence: ""
```

启用时把 `backend` 改为 `space`、`enabled` 改为 `true`，并填写三项核对记录。
配置不完整会抛 `SandboxConfigError`，不会自动回退。

## 优化前的本地验证记录

- `tests/test_space_sandbox.py` 收集 44 个离线用例，覆盖配置/工厂、完整生命周期、
  SSE 分片与多行、异常退出、缺失退出事件、创建/命令响应丢失、真实 SQLite effect
  台账、取消、epoch/任务隔离、一次性句柄拒绝第二次命令、清理失败和标准库传输。
- `.\tools\run.ps1 ci` 在 2026-10-19 全绿（入口导入、compileall、pytest、7 个离线
  demo、P0 checkpoint/resume spike）。详细命令和测试数量见
  `docs/validation/sandbox_space_2026-10-19.txt`。
- 这些结果只证明离线适配器行为，不证明真实 SPACE 部署可用。
- 本轮后续优化未运行这些测试，不能沿用上述通过结果判断当前版本已通过回归。

## 尚未验证

- 没有真实 SPACE API/data-plane 调用，没有真实模板、KVM、egress 或凭据轮换联调。
- 没有固定真实模板摘要；`SPACE_TEMPLATE` 是部署输入，本次没有可核对的生产 digest。
- 没有验证具体部署的 egress profile 或 `network_policy` override 是否真的阻止外网；
  在核对前不应把该后端用于生产。
- 没有测量创建时延、执行开销或资源收益；不宣称性能提升。
- 没有实现暂停/恢复、快照、预热池、文件传输或跨任务工作区。
- 没有声称任务检查点恢复等于虚拟机回滚或外部副作用撤销。
