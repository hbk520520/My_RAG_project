from sandbox_manager import DockerSandboxManager
import logging

logger = logging.getLogger(__name__)

# 全局单例
sandbox_manager = DockerSandboxManager()

def node_write_code(state: dict) -> dict:
    """生成代码的节点（此处省略具体 LLM 调用）"""
    # state 中应包含 current_sub_task
    code = "# 示例：计算赔偿金\nresult = 10000 * 2"
    state["generated_code"] = code
    state["next_action"] = "execute_code"
    return state

def node_execute_code(state: dict) -> dict:
    """执行代码的节点，调用 Docker 沙箱"""
    code = state.get("generated_code", "")
    hop = state.get("current_hop", 0)

    # 1. 获取或创建沙箱会话
    session_id = state.get("sandbox_session_id")
    if not session_id:
        session_id = sandbox_manager.start_session()
        state["sandbox_session_id"] = session_id

    logger.info(f"[Hop {hop}] 在沙箱 {session_id} 中执行代码")

    # 2. 执行并获取结果
    result = sandbox_manager.execute_code(session_id, code)

    # 3. 错误处理：打回重写
    if result.get("error"):
        logger.warning(f"沙箱报错: {result['error']}")
        state["current_context"].append({
            "hop": hop, "status": "error", "data": result["error"]
        })
        state["next_action"] = "write_code"   # 触发代码重写
        return state

    # 4. 成功：将输出存入上下文，继续主流程
    output = result.get("output", "")
    state["current_context"].append({
        "hop": hop, "status": "success", "data": output
    })
    state["next_action"] = "evaluate"   # 回到大脑节点
    return state

def terminate_session(state: dict):
    """在对话结束或异常时清理沙箱"""
    session_id = state.get("sandbox_session_id")
    if session_id:
        sandbox_manager.destroy_session(session_id)
        state["sandbox_session_id"] = None