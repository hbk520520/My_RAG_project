import docker
import requests
import time
import uuid
import logging

logger = logging.getLogger("SandboxScheduler")

class DockerSandboxManager:
    """管理有状态 Docker 沙箱的生命周期与代码执行"""

    def __init__(self, image_name: str = "legal-sandbox:v1"):
        self.client = docker.from_env()
        self.image_name = image_name
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
                # 安全防线
                mem_limit="512m",
                cpu_quota=50000,       # 最多占用 0.5 个核心
                network_mode="none",   # 完全断网
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
            resp = requests.post(meta["url"], json={"code": code}, timeout=10)
            return resp.json()
        except requests.Timeout:
            return {"output": "", "error": "Execution Timeout (>10s)"}
        except Exception as e:
            return {"output": "", "error": f"Sandbox communication error: {str(e)}"}

    def destroy_session(self, session_id: str):
        """销毁沙箱容器，释放资源"""
        if session_id in self.active_containers:
            logger.info(f"销毁沙箱: {session_id}")
            container = self.active_containers[session_id]["container"]
            try:
                container.remove(force=True)
            except Exception as e:
                logger.error(f"销毁沙箱 {session_id} 时出错: {e}")
            finally:
                del self.active_containers[session_id]

    def cleanup_idle(self, max_idle_seconds: int = 600):
        """回收闲置超过指定秒数的容器（可放入后台守护线程）"""
        now = time.time()
        to_delete = []
        for sid, meta in self.active_containers.items():
            if now - meta["last_active"] > max_idle_seconds:
                to_delete.append(sid)
        for sid in to_delete:
            self.destroy_session(sid)