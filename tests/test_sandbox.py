"""
沙箱执行单元测试
================
覆盖：
  - `legal_sandbox/sandbox_exec.py`：`run_in_process` / `run_code_once` / `destroy_session`
  - `legal_sandbox/sandbox_manager.py`：执行超时、按名字兜底销毁、闲置回收

回归背景（阶段 5 / 6）：
  1. `node_execute_code` 原先只 catch ImportError，而 Docker 抛的是
     `DockerException` —— 一次沙箱异常就把整张 LangGraph 打断。现在统一走
     `run_code_once`：**任何** Docker 侧异常都降级为本进程执行，不再上抛。
  2. `destroy_session` 原先只查 `self.active_containers`，而调用方每次都 new 一个
     manager 实例（映射为空）→ 静默不销毁，容器只增不减（泄漏）。现在会按
     容器名（== session_id）兜底找回。

⚠️ 注意：`run_in_process` 只是"受限 builtins"，**不是安全边界**（见模块头警告）。
   本文件末尾有一条测试专门记录这个已知限制。
"""
import sys
import time
import types

import pytest
import requests

import sandbox_exec
import sandbox_manager
from sandbox_manager import DockerSandboxManager

# ============================================================================
# 假件
# ============================================================================
class FakeManager:
    """替身沙箱调度器：不碰 Docker，只按预设返回结果"""

    def __init__(self, output: str = "", error: str = None,
                 session: str = "sandbox_fake", raise_on_execute=None,
                 raise_on_start=None):
        self.output = output
        self.error = error
        self.session = session
        self._raise_on_execute = raise_on_execute
        self._raise_on_start = raise_on_start
        self.execute_calls = []

    def start_session(self):
        if self._raise_on_start:
            raise self._raise_on_start
        return self.session

    def execute_code(self, session_id, code):
        self.execute_calls.append((session_id, code))
        if self._raise_on_execute:
            raise self._raise_on_execute
        return {"output": self.output, "error": self.error}

    def destroy_session(self, session_id):
        self.destroyed = session_id


@pytest.fixture
def manager(monkeypatch):
    """真实的 DockerSandboxManager，但把 `docker.from_env` 换成假客户端。

    本机一般没有 Docker 守护进程，而 docker SDK 7.x 的 `from_env()` 会真的去
    协商 API 版本并抛 DockerException —— 那样这些测试就全被跳过了，等于没测。
    这里只替换「客户端工厂」，`__init__` 里的**配置读取逻辑**
    （镜像名/内存/CPU/网络/执行超时/闲置回收）照常真实执行。
    """
    def _no_such_container(name):
        raise Exception(f"No such container: {name}")

    fake_client = types.SimpleNamespace(
        containers=types.SimpleNamespace(get=_no_such_container)
    )
    monkeypatch.setattr(sandbox_manager.docker, "from_env", lambda: fake_client)
    return DockerSandboxManager(image_name="legal-sandbox-test:v1")


# ============================================================================
# 1. run_in_process —— 降级执行路径
# ============================================================================
def test_run_in_process_returns_result_variable():
    res = sandbox_exec.run_in_process("base = 8000\nmonths = 5.5\nresult = (months + 1) * base")
    assert res["error"] is None
    assert res["calc_result"] == "52000.0"
    assert res["via"] == "local"
    assert res["session_id"] is None


def test_run_in_process_falls_back_to_stdout():
    res = sandbox_exec.run_in_process("print('直接打印的结果')")
    assert res["error"] is None
    assert res["calc_result"] == "直接打印的结果"


def test_run_in_process_without_result_or_output():
    res = sandbox_exec.run_in_process("x = 1 + 1")
    assert res["error"] is None
    assert "未定义 result" in res["calc_result"]


def test_run_in_process_captures_error_instead_of_raising():
    res = sandbox_exec.run_in_process("raise ValueError('业务异常')")
    assert res["calc_result"] is None
    assert "ValueError" in res["error"]
    assert "业务异常" in res["error"]
    assert res["via"] == "local"


def test_run_in_process_supports_try_except():
    """阶段 8 修复的回归测试

    白名单原先**没有任何异常类**，而 `exec` 的 globals 用
    `{"__builtins__": dict(_SAFE_BUILTINS)}` 整体替换了内置命名空间，
    于是降级路径下任何带 try/except 的生成代码都会
    `NameError: name 'ZeroDivisionError' is not defined`。
    """
    code = (
        "try:\n"
        "    result = 1 / 0\n"
        "except ZeroDivisionError:\n"
        "    result = '除零已捕获'\n"
    )
    res = sandbox_exec.run_in_process(code)
    assert res["error"] is None
    assert res["calc_result"] == "除零已捕获"


def test_run_in_process_common_exceptions_are_available():
    for name in ("ValueError", "TypeError", "KeyError", "IndexError",
                 "ZeroDivisionError", "ArithmeticError", "RuntimeError",
                 "AttributeError", "Exception"):
        res = sandbox_exec.run_in_process(f"result = {name}.__name__")
        assert res["error"] is None, f"{name} 不在白名单里：{res['error']}"
        assert res["calc_result"] == name


def test_run_in_process_restores_sys_stdout_even_on_error():
    """exec 抛异常时也必须把 sys.stdout 还原，否则后续日志全部丢失"""
    original = sys.stdout
    sandbox_exec.run_in_process("raise RuntimeError('boom')")
    assert sys.stdout is original


def test_run_in_process_restores_sys_stdout_on_success():
    original = sys.stdout
    sandbox_exec.run_in_process("print('hi')")
    assert sys.stdout is original


def test_run_in_process_blocks_dangerous_builtins():
    """受限 builtins：open / __import__ / eval 都不可用"""
    for code in ("open('/etc/passwd')",
                 "__import__('os').system('echo hi')",
                 "eval('1+1')"):
        res = sandbox_exec.run_in_process(code)
        assert res["error"] is not None, f"{code!r} 竟然执行成功了"
        assert "NameError" in res["error"]


def test_run_in_process_is_not_a_security_boundary():
    """⚠️ 已知限制（记录用，非缺陷修复目标）

    受限 builtins 能挡住 open()/__import__()，但挡不住**属性链逃逸**：
    `().__class__.__bases__[0].__subclasses__()` 依然可达。
    所以 `sandbox_exec` 模块头明确写着「降级路径绕过容器隔离，生产环境必须
    保证 Docker 沙箱可用」。一旦这条断言失败了（说明逃逸被堵住了），
    可以把模块头那句警告降级。
    """
    res = sandbox_exec.run_in_process(
        "subs = ().__class__.__bases__[0].__subclasses__()\n"
        "result = len(subs) > 0"
    )
    assert res["error"] is None
    assert res["calc_result"] == "True"


# ============================================================================
# 2. run_code_once —— 两条链路共用的入口
# ============================================================================
def test_run_code_once_rejects_empty_code():
    for code in ("", "   ", "\n\t"):
        res = sandbox_exec.run_code_once(code)
        assert res["calc_result"] is None
        assert res["error"] == "没有可执行的代码"
        assert res["via"] is None


def test_run_code_once_uses_docker_manager_on_success():
    fake = FakeManager(output="52000.0")
    res = sandbox_exec.run_code_once("result = 52000.0", manager=fake)
    assert res == {"calc_result": "52000.0", "error": None,
                   "session_id": "sandbox_fake", "via": "docker"}
    assert fake.execute_calls == [("sandbox_fake", "result = 52000.0")]


def test_run_code_once_reuses_given_session_id():
    fake = FakeManager(output="ok")
    res = sandbox_exec.run_code_once("result = 1", session_id="sandbox_existing", manager=fake)
    assert res["session_id"] == "sandbox_existing"
    assert fake.execute_calls[0][0] == "sandbox_existing"


def test_run_code_once_propagates_docker_side_error():
    fake = FakeManager(error="Traceback: ZeroDivisionError")
    res = sandbox_exec.run_code_once("result = 1/0", manager=fake)
    assert res["calc_result"] is None
    assert res["error"] == "Traceback: ZeroDivisionError"
    assert res["via"] == "docker"
    assert res["session_id"] == "sandbox_fake"


def test_run_code_once_strips_surrounding_whitespace_from_output():
    fake = FakeManager(output="\n  52000.0\n")
    assert sandbox_exec.run_code_once("result = 1", manager=fake)["calc_result"] == "52000.0"


def test_run_code_once_degrades_when_manager_unavailable(monkeypatch):
    """阶段 5 修复：拿不到 Docker 调度器时必须降级，不得上抛"""
    def _boom():
        raise RuntimeError("Docker daemon 不可用")

    monkeypatch.setattr(sandbox_exec, "get_sandbox_manager", _boom)
    res = sandbox_exec.run_code_once("result = 6 * 7")
    assert res["via"] == "local"
    assert res["error"] is None
    assert res["calc_result"] == "42"


def test_run_code_once_degrades_when_execute_raises(monkeypatch):
    """阶段 5 修复：DockerException 之类必须在入口处被吞掉

    原先 soul.node_execute_code 只 catch ImportError，Docker 抛的
    DockerException 会直接打断整张图。
    """
    fake = FakeManager(raise_on_execute=RuntimeError("DockerException: 容器已退出"))
    res = sandbox_exec.run_code_once("result = 1 + 1", manager=fake)
    assert res["via"] == "local"
    assert res["calc_result"] == "2"


def test_run_code_once_degrades_when_start_session_raises():
    fake = FakeManager(raise_on_start=RuntimeError("无法获取沙箱端口映射"))
    res = sandbox_exec.run_code_once("result = 'ok'")
    assert res == {"calc_result": "ok", "error": None,
                   "session_id": None, "via": "local"}

    fake2 = FakeManager(raise_on_start=RuntimeError("端口映射失败"))
    res2 = sandbox_exec.run_code_once("result = 'ok'", manager=fake2)
    assert res2["via"] == "local"


# ============================================================================
# 3. destroy_session —— 幂等 + 失败不上抛
# ============================================================================
def test_destroy_session_is_noop_for_empty_id():
    sandbox_exec.destroy_session(None)
    sandbox_exec.destroy_session("")


def test_destroy_session_swallows_manager_errors(monkeypatch):
    class _Bad(FakeManager):
        def destroy_session(self, session_id):
            raise RuntimeError("清理失败")

    sandbox_exec.destroy_session("sandbox_x", manager=_Bad())


def test_destroy_session_delegates_to_manager():
    fake = FakeManager()
    sandbox_exec.destroy_session("sandbox_abc", manager=fake)
    assert fake.destroyed == "sandbox_abc"


# ============================================================================
# 4. sandbox_manager —— 执行超时
# ============================================================================
def test_execution_timeout_comes_from_config(manager):
    from config_loader import cfg
    assert manager.execution_timeout == cfg.get("sandbox", "execution_timeout")
    assert isinstance(manager.execution_timeout, (int, float))
    assert manager.execution_timeout > 0


def test_resource_limits_come_from_config(manager):
    from config_loader import cfg
    assert manager.mem_limit == cfg.get("sandbox", "mem_limit")
    assert manager.cpu_quota == cfg.get("sandbox", "cpu_quota")
    assert manager.network_mode == cfg.get("sandbox", "network_mode")


def test_image_name_can_be_overridden(manager):
    assert DockerSandboxManager(image_name="custom:tag").image_name == "custom:tag"


def test_execute_code_passes_timeout_to_requests(manager, monkeypatch):
    """超时值必须真的传给 requests，否则配置形同虚设"""
    sid = "sandbox_timeout_test"
    manager.active_containers[sid] = {
        "container": None, "url": "http://127.0.0.1:9/execute", "last_active": time.time(),
    }
    seen = {}

    def _fake_post(url, json=None, timeout=None):
        seen["url"] = url
        seen["json"] = json
        seen["timeout"] = timeout
        raise requests.Timeout("read timed out")

    monkeypatch.setattr(sandbox_manager.requests, "post", _fake_post)

    res = manager.execute_code(sid, "result = 1")
    assert seen["timeout"] == manager.execution_timeout
    assert seen["json"] == {"code": "result = 1"}
    assert res["output"] == ""
    assert str(manager.execution_timeout) in res["error"]
    assert "Timeout" in res["error"]


def test_execute_code_maps_communication_error(manager, monkeypatch):
    sid = "sandbox_comm_error"
    manager.active_containers[sid] = {
        "container": None, "url": "http://127.0.0.1:9/execute", "last_active": time.time(),
    }
    monkeypatch.setattr(sandbox_manager.requests, "post",
                        lambda *a, **kw: (_ for _ in ()).throw(ConnectionError("拒绝连接")))

    res = manager.execute_code(sid, "result = 1")
    assert res["output"] == ""
    assert "Sandbox communication error" in res["error"]


def test_execute_code_rejects_unknown_session(manager):
    with pytest.raises(ValueError) as ei:
        manager.execute_code("sandbox_不存在", "result = 1")
    assert "不存在" in str(ei.value)


def test_execute_code_refreshes_last_active(manager, monkeypatch):
    sid = "sandbox_touch"
    old = time.time() - 9999
    manager.active_containers[sid] = {
        "container": None, "url": "http://127.0.0.1:9/execute", "last_active": old,
    }
    monkeypatch.setattr(sandbox_manager.requests, "post",
                        lambda *a, **kw: types.SimpleNamespace(json=lambda: {"output": "ok"}))
    manager.execute_code(sid, "result = 1")
    assert manager.active_containers[sid]["last_active"] > old


# ============================================================================
# 5. sandbox_manager —— 销毁（阶段 5 泄漏修复）
# ============================================================================
def _with_fake_containers(manager, get_impl):
    manager.client = types.SimpleNamespace(containers=types.SimpleNamespace(get=get_impl))
    return manager


def test_destroy_session_removes_container_from_active_map(manager, fake_container):
    sid = "sandbox_tracked"
    manager.active_containers[sid] = {"container": fake_container, "url": "", "last_active": time.time()}
    manager.destroy_session(sid)
    assert fake_container.removed is True
    assert fake_container.force_used is True
    assert sid not in manager.active_containers


def test_destroy_session_falls_back_to_name_lookup(manager, fake_container):
    """关键回归：会话不在本实例的 active_containers 里（调用方新建了 manager），
    也必须按容器名找回并删除，否则容器泄漏。"""
    looked_up = {}

    def _get(name):
        looked_up["name"] = name
        return fake_container

    _with_fake_containers(manager, _get)
    manager.destroy_session("sandbox_orphan")

    assert looked_up["name"] == "sandbox_orphan", "容器名就是 session_id，应据此兜底查找"
    assert fake_container.removed is True


def test_destroy_session_silently_returns_when_container_missing(manager):
    def _get(name):
        raise Exception("No such container: " + name)

    _with_fake_containers(manager, _get)
    manager.destroy_session("sandbox_never_existed")   # 不应抛异常


def test_destroy_session_swallows_remove_error(manager):
    class _BadContainer:
        def remove(self, force=False):
            raise RuntimeError("容器正忙")

    manager.active_containers["sandbox_bad"] = {
        "container": _BadContainer(), "url": "", "last_active": time.time(),
    }
    manager.destroy_session("sandbox_bad")   # 不应抛异常


# ============================================================================
# 6. sandbox_manager —— 闲置回收
# ============================================================================
def test_cleanup_idle_only_evicts_stale_sessions(manager, monkeypatch):
    now = time.time()
    manager.active_containers = {
        "fresh": {"container": None, "url": "", "last_active": now},
        "stale": {"container": None, "url": "", "last_active": now - 1000},
    }
    evicted = []
    monkeypatch.setattr(manager, "destroy_session", lambda sid: evicted.append(sid))

    manager.cleanup_idle(max_idle_seconds=600)
    assert evicted == ["stale"]


def test_cleanup_idle_uses_config_default(manager, monkeypatch):
    from config_loader import cfg
    assert manager.idle_cleanup_seconds == cfg.get("sandbox", "idle_cleanup_seconds")

    now = time.time()
    manager.active_containers = {
        "very_stale": {"container": None, "url": "",
                       "last_active": now - manager.idle_cleanup_seconds - 10},
        "fresh": {"container": None, "url": "", "last_active": now},
    }
    evicted = []
    monkeypatch.setattr(manager, "destroy_session", lambda sid: evicted.append(sid))

    manager.cleanup_idle()
    assert evicted == ["very_stale"]


# ============================================================================
# 7. 共享入口只应有一份
# ============================================================================
def test_sandbox_exec_does_not_reimplement_docker_lifecycle():
    """两条链路必须共用 sandbox_exec，不得各自复制一份 Docker/降级逻辑"""
    import inspect
    src = inspect.getsource(sandbox_exec)
    # 只应通过 sandbox_manager 间接操作容器
    assert "docker.from_env" not in src
    assert "containers.run" not in src
    assert "from sandbox_manager import" in src
