"""Network-free, bounded Docker execution. One process per calculation."""
from __future__ import annotations
import base64
import json
import queue
import threading
import time
import uuid
import docker
from config_loader import cfg

class DockerSandboxManager:
    def __init__(self, image_name=None):
        self.client = docker.from_env()
        self.image_name = image_name or cfg.get('sandbox', 'image')
        self.mem_limit = cfg.get('sandbox', 'mem_limit', default='512m')
        self.cpu_quota = cfg.get('sandbox', 'cpu_quota', default=50000)
        self.network_mode = 'none'
        self.execution_timeout = cfg.get('sandbox', 'execution_timeout', default=10)
        self.idle_cleanup_seconds = cfg.get('sandbox', 'idle_cleanup_seconds', default=600)
        self.active_containers = {}

    def start_session(self):
        session_id = 'sandbox_' + uuid.uuid4().hex
        container = self.client.containers.run(
            image=self.image_name, name=session_id, detach=True,
            command=['python', '-c', 'import time; time.sleep(86400)'],
            mem_limit=self.mem_limit, cpu_quota=self.cpu_quota,
            network_mode='none', read_only=True, pids_limit=32,
            tmpfs={'/tmp': 'rw,noexec,nosuid,size=64m'}, cap_drop=['ALL'],
            user='65534:65534', security_opt=['no-new-privileges:true'],
            labels={'managed_by': 'legal-agent', 'created_at': str(time.time())})
        self.active_containers[session_id] = {'container': container, 'last_active': time.time()}
        return session_id

    def execute_code(self, session_id, code):
        from agent_runtime.context import current_run
        if session_id not in self.active_containers:
            raise ValueError(f"Session {session_id} 不存在或已销毁")
        meta = self.active_containers[session_id]
        meta['last_active'] = time.time()
        container = meta['container']
        context = current_run.get()
        result = queue.Queue(maxsize=1)
        encoded = base64.b64encode(code.encode()).decode()
        def execute():
            try:
                result.put((True, container.exec_run(['python', '/app/runner.py', encoded])))
            except Exception as exc:
                result.put((False, exc))
        thread = threading.Thread(target=execute, daemon=True)
        thread.start()
        deadline = time.monotonic() + self.execution_timeout
        try:
            while True:
                if context:
                    context.check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.destroy_session(session_id)
                    return {'output': '', 'error': 'Execution Timeout', 'error_kind': 'timeout'}
                try:
                    ok, payload = result.get(timeout=min(0.1, remaining))
                    break
                except queue.Empty:
                    continue
            if not ok:
                raise payload
            if payload.exit_code:
                return {'output': '', 'error': f'Sandbox exited with code {payload.exit_code}', 'error_kind': 'execution'}
            if len(payload.output) > cfg.get('sandbox', 'max_output_bytes', default=65536) + 4096:
                return {'output': '', 'error': 'Sandbox output limit exceeded', 'error_kind': 'output_limit'}
            return json.loads(payload.output)
        except BaseException:
            self.destroy_session(session_id)
            raise

    def destroy_session(self, session_id):
        meta = self.active_containers.pop(session_id, None)
        try:
            container = meta['container'] if meta else self.client.containers.get(session_id)
            container.remove(force=True)
        except docker.errors.NotFound:
            return

    def cleanup_idle(self, max_idle_seconds=None):
        limit = self.idle_cleanup_seconds if max_idle_seconds is None else max_idle_seconds
        for session_id, meta in list(self.active_containers.items()):
            if time.time() - meta['last_active'] > limit:
                self.destroy_session(session_id)

    def cleanup_orphans(self):
        for container in self.client.containers.list(all=True, filters={'label': 'managed_by=legal-agent'}):
            if time.time() - float(container.labels['created_at']) > self.idle_cleanup_seconds:
                container.remove(force=True)

_default_manager = None

def get_sandbox_manager():
    global _default_manager
    if _default_manager is None:
        _default_manager = DockerSandboxManager()
    return _default_manager
