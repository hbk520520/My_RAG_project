"""
沙箱执行统一入口 —— soul.py 与 reasoner_worker 共用
================================================
背景（阶段 5）：原先只有 soul.py 的 LangGraph 路径真正跑沙箱；Reasoner Worker
只设置了 `needs_sandbox_calc = True` 却没有任何消费方，**金额计算环节在 Worker
链路里是断的**。这里把"执行一次 LLM 生成的代码"抽成公共函数，两条链路共用，
避免各自复制一份 Docker/降级逻辑。

执行策略（依次尝试）：
  1. Docker 沙箱容器 —— 带状态，同一 session 复用容器，变量跨次保留
  2. 本进程内 exec（受限 builtins）—— **降级路径，绕过容器隔离**

⚠️ 生产环境必须保证 Docker 沙箱可用；降级路径仅供开发/无 Docker 环境使用。

技术栈: docker (Python SDK) / subprocess 无关（本地 exec 用 exec+io 捕获）
"""
import io
import os
import sys
import traceback
import logging
from typing import Any, Dict, Optional

# 路径引导：本文件在 multiple-search/legal_sandbox/ 下
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (_ROOT_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sandbox_manager import DockerSandboxManager, get_sandbox_manager

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

    ⚠️ 安全提示：这是降级路径，**绕过 Docker 容器隔离**。
    只允许在开发环境使用；生产环境应保证 sandbox 可用。
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


def run_code_once(code: str,
                  session_id: Optional[str] = None,
                  manager: Optional[DockerSandboxManager] = None) -> Dict[str, Any]:
    """
    执行一次代码（不负责重试，重试由调用方按自己的熔断策略决定）。

    :return: {"calc_result": str|None,   # 成功时的输出
              "error": str|None,          # 失败时的错误信息
              "session_id": str|None,     # 下次复用同一会话可保持变量
              "via": "docker"|"local"|None}
    """
    if not code or not code.strip():
        return {"calc_result": None, "error": "没有可执行的代码",
                "session_id": session_id, "via": None}

    # ---- 尝试获取 Docker 调度器 ----
    try:
        manager = manager or get_sandbox_manager()
    except Exception as e:
        logger.warning(f"Docker 沙箱不可用({e})，降级为本进程内执行"
                       f"（绕过容器隔离，请勿在生产启用）")
        return run_in_process(code)

    # ---- 创建/复用会话并执行 ----
    try:
        if not session_id:
            session_id = manager.start_session()
        result = manager.execute_code(session_id, code)
    except Exception as e:
        logger.warning(f"沙箱调度异常({e})，降级为本进程内执行"
                       f"（绕过容器隔离，请勿在生产启用）")
        return run_in_process(code)

    if result.get("error"):
        return {"calc_result": None, "error": result["error"],
                "session_id": session_id, "via": "docker"}

    return {"calc_result": (result.get("output") or "").strip(),
            "error": None, "session_id": session_id, "via": "docker"}


def destroy_session(session_id: Optional[str], manager: Optional[DockerSandboxManager] = None) -> None:
    """销毁沙箱容器，释放资源（幂等）"""
    if not session_id:
        return
    try:
        manager = manager or get_sandbox_manager()
        manager.destroy_session(session_id)
    except Exception as e:
        logger.warning(f"沙箱清理失败({e})，可能残留容器 {session_id}")
