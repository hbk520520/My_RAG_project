"""Docker-only online execution. run_in_process is an explicitly injected test helper.

Container calculations are stateless. No tool automatically executes code on the host.
"""
import io
import os
import sys
import traceback
import logging
from typing import Any, Dict, Optional

from .manager import DockerSandboxManager, get_sandbox_manager

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
    if not code or not code.strip():
        return {"calc_result": None, "error": "没有可执行的代码", "session_id": session_id, "via": None}
    if len(code.encode("utf-8")) > 65536:
        return {"calc_result": None, "error": "Code input limit exceeded", "session_id": session_id, "via": None}
    def execute():
        nonlocal session_id, manager
        try:
            manager = manager or get_sandbox_manager()
            session_id = session_id or manager.start_session()
            result = manager.execute_code(session_id, code)
            if result.get("error"):
                raise RuntimeError(result["error"])
            value = result["output"].strip()
            if not value:
                raise ValueError("Sandbox returned an empty result")
            return {"calc_result": value, "error": None, "session_id": session_id, "via": "docker"}
        finally:
            if current_run.get() is not None and session_id and manager:
                manager.destroy_session(session_id)
    try:
        outcome = tool_call("sandbox.execute", {"code": code}, execute)
    except (DockerException, OSError, RuntimeError, ValueError) as exc:
        return {"calc_result": None, "error": str(exc), "session_id": session_id, "via": "docker"}
    if current_run.get() is not None:
        outcome["session_id"] = None
    return outcome


def destroy_session(session_id: Optional[str], manager: Optional[DockerSandboxManager] = None) -> None:
    """销毁沙箱容器，释放资源（幂等）"""
    if not session_id:
        return
    try:
        manager = manager or get_sandbox_manager()
        manager.destroy_session(session_id)
    except Exception as e:
        logger.warning(f"沙箱清理失败({e})，可能残留容器 {session_id}")
