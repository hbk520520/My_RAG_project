"""Bounded code execution. Docker is the default; SPACE is opt-in.

Container/remote calculations are stateless from this project's point of view.
No online tool executes code on the host. ``run_in_process`` remains an explicit
offline test helper and is not a security boundary.
"""
import io
import os
import sys
import traceback
import logging
from typing import Any, Dict, Optional

from .errors import (SandboxConfigError, SandboxError, SandboxOutcomeUnknown,
                     SandboxUnavailableError)
from .manager import DockerSandboxManager, get_sandbox_manager, sandbox_backend

logger = logging.getLogger("SandboxExec")

# 本进程降级执行时允许的内置函数（尽量小）
_SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "sum": sum,
    "round": round, "int": int, "float": float,
    "len": len, "range": range, "list": list,
    "dict": dict, "str": str, "bool": bool,
    "True": True, "False": False, "None": None,
    "print": print, "isinstance": isinstance,
    "enumerate": enumerate, "sorted": sorted, "zip": zip,
}

# 阶段 8：补上常用异常类。
# 原先这份白名单里**一个异常类都没有**，而 exec 的 globals 用
# `{"__builtins__": dict(_SAFE_BUILTINS)}` 整体替换了内置命名空间，
# 于是沙箱里的 `raise ValueError(...)` 和 `try/except ZeroDivisionError`
# 全部变成 `NameError: name 'ValueError' is not defined` —— 降级路径下
# 任何带异常处理的 LLM 生成代码都跑不起来。
# 异常类本身是惰性的，加进来不扩大攻击面。
_SAFE_BUILTINS.update({
    "Exception": Exception, "BaseException": BaseException,
    "ValueError": ValueError, "TypeError": TypeError,
    "KeyError": KeyError, "IndexError": IndexError,
    "AttributeError": AttributeError, "NameError": NameError,
    "ArithmeticError": ArithmeticError, "ZeroDivisionError": ZeroDivisionError,
    "RuntimeError": RuntimeError, "AssertionError": AssertionError,
    "StopIteration": StopIteration,
})


def run_in_process(code: str) -> Dict[str, Any]:
    """
    在当前进程内执行代码（受限 builtins）。

    ⚠️ 安全提示：这是显式测试辅助函数，**不提供隔离**。
    只接受开发者提供的可信测试代码；在线工具不调用它。
    """
    old_stdout = sys.stdout
    captured = io.StringIO()
    sys.stdout = captured
    safe_globals: Dict[str, Any] = {"__builtins__": dict(_SAFE_BUILTINS)}

    error = None
    try:
        exec(code, safe_globals)
    except Exception:
        error = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    output = captured.getvalue().strip()
    if error:
        return {"calc_result": None, "error": error, "session_id": None, "via": "local"}

    result_value = safe_globals.get("result")
    if result_value is None:
        calc = output or "计算完成（代码未定义 result 变量）"
    else:
        calc = str(result_value)
    return {"calc_result": calc, "error": None, "session_id": None, "via": "local"}


def run_code_once(code, session_id=None, manager=None):
    from docker.errors import DockerException
    from agent_runtime.context import current_run, tool_call
    from agent_runtime.contracts import (BudgetExceeded, OutcomeUnknown, RunConflict,
                                         RunStopped)
    if not code or not code.strip():
        return {"calc_result": None, "error": "没有可执行的代码", "session_id": session_id, "via": None}
    if len(code.encode("utf-8")) > 65536:
        return {"calc_result": None, "error": "Code input limit exceeded", "session_id": session_id, "via": None}

    # Selecting a policy is local; constructing a manager may contact Docker.
    # Leave construction inside the tool body so cached results stay usable
    # even when the selected execution service is unavailable.
    try:
        backend = getattr(manager, "via", "docker") if manager is not None else sandbox_backend()
    except SandboxConfigError as exc:
        return {"calc_result": None, "error": str(exc),
                "session_id": session_id, "via": None,
                "error_kind": "configuration"}
    read_only = not getattr(manager, "side_effecting", backend == "space")
    one_shot = bool(getattr(manager, "one_shot", backend == "space"))
    cleanup_error = None
    cleanup_attempted = False
    cleanup_succeeded = False

    def cleanup():
        """Release this call's sandbox without replacing its result.

        Managed executions are cleaned inside the tool body so the ledger
        records the actual cleanup outcome alongside the calculation result.
        """
        nonlocal cleanup_error, cleanup_attempted, cleanup_succeeded
        if cleanup_attempted or not session_id or manager is None:
            return
        cleanup_attempted = True
        try:
            manager.destroy_session(session_id)
            cleanup_succeeded = True
        except Exception as exc:
            # Cleanup is a best-effort boundary: a failure must never replace
            # the execution result.  It is logged with the resource handle and
            # returned as cleanup_error.
            cleanup_error = f"{type(exc).__name__}: {exc}"
            logger.exception("沙箱清理失败：session_id=%s", session_id)

    def execute():
        nonlocal session_id, manager
        result_outcome = None
        try:
            if manager is None:
                try:
                    manager = get_sandbox_manager()
                except SandboxConfigError:
                    raise
                except (DockerException, OSError, RuntimeError) as exc:
                    raise SandboxUnavailableError(str(exc)) from exc
            session_id = session_id or manager.start_session()
            result = manager.execute_code(session_id, code)
            if result.get("error"):
                error = RuntimeError(result["error"])
                error.error_kind = result.get("error_kind") or "execution"
                raise error
            value = result["output"].strip()
            if not value:
                raise ValueError("Sandbox returned an empty result")
            result_outcome = {"calc_result": value, "error": None,
                              "session_id": session_id,
                              "via": backend}
        finally:
            if one_shot or current_run.get() is not None:
                cleanup()
        if cleanup_succeeded:
            result_outcome["session_id"] = None
        if cleanup_error:
            result_outcome["cleanup_error"] = cleanup_error
        return result_outcome

    try:
        outcome = tool_call("sandbox.execute", {"code": code}, execute,
                            read_only=read_only)
    except (OutcomeUnknown, RunStopped, BudgetExceeded, RunConflict):
        # Cancellation, epoch invalidation and ambiguous side effects must
        # reach the service unchanged. They are not ordinary code errors.
        raise
    except SandboxOutcomeUnknown as exc:
        raise OutcomeUnknown(str(exc)) from exc
    except SandboxConfigError as exc:
        outcome = {"calc_result": None, "error": str(exc),
                   "session_id": session_id, "via": backend,
                   "error_kind": "configuration"}
    except SandboxUnavailableError as exc:
        outcome = {"calc_result": None, "error": str(exc),
                   "session_id": session_id, "via": backend,
                   "error_kind": "unavailable"}
    except (DockerException, OSError, RuntimeError, ValueError, SandboxError) as exc:
        outcome = {"calc_result": None, "error": str(exc),
                   "session_id": session_id, "via": backend}
        kind = getattr(exc, "error_kind", None)
        if kind:
            outcome["error_kind"] = kind

    if cleanup_succeeded:
        outcome["session_id"] = None
    if cleanup_error:
        outcome["cleanup_error"] = cleanup_error
    return outcome


def destroy_session(session_id: Optional[str], manager: Optional[DockerSandboxManager] = None) -> bool:
    """Destroy a sandbox resource.

    Returns ``True`` only when the manager reported success.  Cleanup failures
    are logged with the resource identifier and return ``False`` so callers do
    not silently discard ownership of a resource that may still exist.
    """
    if not session_id:
        return True
    try:
        manager = manager or get_sandbox_manager()
        manager.destroy_session(session_id)
        return True
    except Exception:
        # Best-effort cleanup boundary; never override the caller's result.
        logger.exception("沙箱清理失败：session_id=%s，可能残留资源", session_id)
        return False
