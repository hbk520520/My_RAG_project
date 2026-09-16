"""
沙箱集成示例 —— 演示 `sandbox_exec` 的工具返回契约
=================================================
这个文件的用途是**演示怎么用沙箱**，不是生产代码。真正被 soul.py 与
reasoner_worker 调用的实现是 `sandbox_exec.run_code_once()`。

阶段 5 之后，"执行一次 LLM 生成的代码"只有**一个**入口：`run_code_once()`。
本示例改用该入口（此前自己调 `DockerSandboxManager.start_session()/execute_code()`，
等于绕开统一入口、又复制了一份 Docker/降级逻辑）。

⚠️ P0 修复：原先这里在**模块级**执行 `sandbox_manager = DockerSandboxManager()`，
   结果 `import example_usage` 在没有 Docker 的机器上直接抛
   `docker.errors.DockerException`。这与阶段 2 的"import soul 就持有假 API Key"
   是同一类缺陷 —— **导入不得产生副作用**。现在改为惰性获取。

离线可跑：`python multiple-search/legal_sandbox/example_usage.py`
  无 Docker 时会自动降级为本进程执行（`via="local"`，绕过容器隔离，仅限开发）。
"""
import logging
import os
import sys
from typing import Any, Dict

# 路径引导：本文件在 multiple-search/legal_sandbox/ 下
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
for _p in (_ROOT_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sandbox_exec import destroy_session, run_code_once

logger = logging.getLogger("SandboxExample")

_manager_cache = None


def get_manager():
    """
    **惰性**获取 Docker 沙箱管理器。

    为什么不能写成模块级常量：`DockerSandboxManager()` 会真的与 Docker daemon
    协商 API 版本。在模块级执行 = 任何 `import` 都要求本机有 Docker daemon，
    而没有 Docker 的环境（含 CI）连 import 都会炸。
    """
    global _manager_cache
    if _manager_cache is None:
        from sandbox_manager import get_sandbox_manager
        _manager_cache = get_sandbox_manager()
    return _manager_cache


# ===========================================================================
# 示例节点：写代码 → 执行 → 按结果决定下一步
# ===========================================================================
def node_write_code(state: Dict[str, Any]) -> Dict[str, Any]:
    """生成代码的节点（此处省略具体 LLM 调用）"""
    state["generated_code"] = "# 示例：计算赔偿金\nresult = 10000 * 2"
    state["next_action"] = "execute_code"
    return state


def node_execute_code(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    执行代码的节点。

    走 `run_code_once()` 统一入口，返回契约见 `docs/CONTRACTS.md` 契约 7：
        {"calc_result": str|None, "error": str|None,
         "session_id": str|None, "via": "docker"|"local"|None}
    """
    code = state.get("generated_code", "")
    hop = state.get("current_hop", 0)

    outcome = run_code_once(code, session_id=state.get("sandbox_session_id"))
    state["sandbox_session_id"] = outcome["session_id"]

    if outcome["error"]:
        logger.warning(f"沙箱报错（via={outcome['via']}）: {outcome['error']}")
        state.setdefault("current_context", []).append({
            "hop": hop, "status": "error", "data": outcome["error"],
        })
        state["next_action"] = "write_code"      # 打回重写
    else:
        logger.info(f"[Hop {hop}] 沙箱执行成功（via={outcome['via']}）")
        state.setdefault("current_context", []).append({
            "hop": hop, "status": "success", "data": outcome["calc_result"],
        })
        state["next_action"] = "evaluate"        # 回到主流程
    return state


def terminate_session(state: Dict[str, Any]) -> Dict[str, Any]:
    """在对话结束或异常时清理沙箱（幂等）"""
    session_id = state.get("sandbox_session_id")
    if session_id:
        destroy_session(session_id)
        state["sandbox_session_id"] = None
    return state


# ===========================================================================
# 离线演示
# ===========================================================================
if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    logging.basicConfig(level=logging.INFO)

    print("=" * 66)
    print("沙箱集成示例 —— 成功路径 / 失败路径 / 清理")
    print("=" * 66)

    # ---- 1. 成功路径 ----
    state: Dict[str, Any] = {"current_hop": 1}
    state = node_execute_code(node_write_code(state))
    print(f"\n[成功路径] next_action={state['next_action']}")
    print(f"           结果={state['current_context'][-1]['data']}")
    assert state["next_action"] == "evaluate"
    assert state["current_context"][-1]["data"] == "20000", "10000 * 2 应为 20000"

    # ---- 2. 失败路径：代码报错必须被"回传"而不是被吞掉 ----
    bad: Dict[str, Any] = {"current_hop": 2, "generated_code": "raise ValueError('boom')"}
    bad = node_execute_code(bad)
    print(f"\n[失败路径] next_action={bad['next_action']}")
    print(f"           错误={bad['current_context'][-1]['data'].strip().splitlines()[-1]}")
    assert bad["next_action"] == "write_code", "报错必须打回重写（不许静默吞掉）"

    # ---- 3. 清理 ----
    state = terminate_session(state)
    print(f"\n[清理] sandbox_session_id={state['sandbox_session_id']}")

    print("\n✅ 示例结论：成功/失败/清理三条路径行为符合 docs/CONTRACTS.md 契约 7")
