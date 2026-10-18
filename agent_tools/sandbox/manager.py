"""
Docker 沙箱调度器 —— 创建/使用/销毁隔离执行环境
=============================================
每个会话一个容器，内存上限 512MB、CPU 0.5 核、网络完全断开。
闲置 10 分钟自动回收，防止资源泄漏。

技术栈: docker (Python SDK) / uuid / requests
"""
import os, sys
import docker, requests, time, uuid, logging

# 允许从任意 cwd 导入根目录模块（config_loader）
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

try:
    from config_loader import cfg
except Exception:  # 沙箱镜像内可能没有 config_loader，此时退回内置默认值
    cfg = None

logger = logging.getLogger("SandboxScheduler")


def _cfg_get(*keys, default=None):
    """带兜底的配置读取：脱离 config.yaml 运行时用内置默认值"""
    if cfg is None:
        return default
    return cfg.get(*keys, default=default)


class DockerSandboxManager:
    """管理有状态 Docker 沙箱的生命周期与代码执行

    阶段 2：镜像名、内存/CPU 限额、网络模式、执行超时、闲置回收时间
    全部改读 config.yaml 的 sandbox 段（原先硬编码在方法体里，
    改一个限额必须动代码）。
    """

    def __init__(self, image_name: str = None):
        self.client = docker.from_env()
        # ---- 阶段 2：配置化 ----
        self.image_name = image_name or _cfg_get("sandbox", "image", default="legal-sandbox:v1")
        self.mem_limit = _cfg_get("sandbox", "mem_limit", default="512m")
        self.cpu_quota = _cfg_get("sandbox", "cpu_quota", default=50000)
        self.network_mode = _cfg_get("sandbox", "network_mode", default="none")
        self.execution_timeout = _cfg_get("sandbox", "execution_timeout", default=10)
        self.idle_cleanup_seconds = _cfg_get("sandbox", "idle_cleanup_seconds", default=600)
        # 记录 session_id -> 容器元数据
        self.active_containers = {}

    def start_session(self) -> str:
        """为一个新的对话会话创建沙箱容器，返回 session_id"""
        session_id = f"sandbox_{uuid.uuid4().hex[:8]}"
        logger.info(f"正在创建沙箱容器: {session_id}")

        try:
            container = self.client.containers.run(
                image=self.image_name,
                name=session_id,
                detach=True,
                # 安全防线（阶段 2：限额来自 config.yaml 的 sandbox 段）
                mem_limit=self.mem_limit,
                cpu_quota=self.cpu_quota,          # 最多占用 0.5 个核心
                network_mode=self.network_mode,    # 完全断网
                security_opt=["no-new-privileges:true"],
                ports={'8000/tcp': None}  # 随机映射宿主机端口
            )

            # 等待服务就绪（重试最多3次）
            container.reload()
            host_port = None
            for attempt in range(3):
                container.reload()
                ports_info = container.attrs['NetworkSettings']['Ports']
                if '8000/tcp' in ports_info and ports_info['8000/tcp']:
                    host_port = ports_info['8000/tcp'][0]['HostPort']
                    break
                time.sleep(1)
            if not host_port:
                raise RuntimeError("无法获取沙箱端口映射")

            self.active_containers[session_id] = {
                "container": container,
                "url": f"http://127.0.0.1:{host_port}/execute",
                "last_active": time.time()
            }

            # 额外等待 FastAPI 完全启动
            time.sleep(1)
            return session_id

        except Exception as e:
            logger.error(f"创建沙箱失败: {e}")
            raise

    def execute_code(self, session_id: str, code: str) -> dict:
        """向指定沙箱发送代码并获取执行结果"""
        if session_id not in self.active_containers:
            raise ValueError(f"Session {session_id} 不存在或已销毁")

        meta = self.active_containers[session_id]
        meta["last_active"] = time.time()

        try:
            resp = requests.post(meta["url"], json={"code": code},
                                 timeout=self.execution_timeout)
            return resp.json()
        except requests.Timeout:
            return {"output": "", "error": f"Execution Timeout (>{self.execution_timeout}s)"}
        except Exception as e:
            return {"output": "", "error": f"Sandbox communication error: {str(e)}"}

    def destroy_session(self, session_id: str):
        """
        销毁沙箱容器，释放资源。

        阶段 5 修复：原先只查 self.active_containers，而调用方（soul.py 的
        node_cleanup_sandbox、reasoner_worker）每次都会 **新建** manager 实例，
        新实例的 active_containers 是空的 —— 于是 destroy_session 静默什么都不做，
        容器只增不减（泄漏）。
        这里增加兜底：容器名就是 session_id，可以直接按名字找回并强制删除。
        """
        meta = self.active_containers.pop(session_id, None)
        container = meta["container"] if meta else None

        if container is None:
            try:
                container = self.client.containers.get(session_id)
            except Exception:
                logger.info(f"沙箱 {session_id} 不存在或已销毁，无需清理")
                return

        try:
            container.remove(force=True)
            logger.info(f"沙箱 {session_id} 已销毁")
        except Exception as e:
            logger.error(f"销毁沙箱 {session_id} 时出错: {e}")

    def cleanup_idle(self, max_idle_seconds: int = None):
        """回收闲置超过指定秒数的容器（可放入后台守护线程）"""
        if max_idle_seconds is None:
            max_idle_seconds = self.idle_cleanup_seconds
        now = time.time()
        to_delete = []
        for sid, meta in self.active_containers.items():
            if now - meta["last_active"] > max_idle_seconds:
                to_delete.append(sid)
        for sid in to_delete:
            self.destroy_session(sid)


# ============================================================================
# 进程内单例（阶段 5 新增）
# ============================================================================
_default_manager = None


def get_sandbox_manager() -> DockerSandboxManager:
    """
    获取进程内共享的沙箱调度器。

    单例的意义：会话映射（active_containers）与 docker client 只建一次，
    避免"每次调用 new 一个 manager、彼此看不到对方创建的会话"。
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = DockerSandboxManager()
    return _default_manager