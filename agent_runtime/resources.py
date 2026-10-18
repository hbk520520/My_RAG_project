"""Host-local admission, shared by API and Kafka processes without schema changes.

OS locks release on process death. Limits describe a deployment, not a GPU model.
Model objects are shared inside a process; device slots are shared across processes.
"""
from contextlib import contextmanager
from functools import lru_cache
import errno
import hashlib
import os
import time
import threading
import atexit

from pydantic import BaseModel, ConfigDict, Field
from config_loader import cfg
from .contracts import RunConflict, local_path


class SchedulerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_parallel_tasks: int = Field(default=2, ge=1, le=64)
    max_running_runs: int = Field(default=2, ge=1, le=64)
    remote_model_slots: int = Field(default=4, ge=1, le=256)
    local_model_slots: int = Field(default=1, ge=1, le=64)
    encoder_slots: int = Field(default=1, ge=1, le=64)
    tool_slots: int = Field(default=1, ge=1, le=64)
    max_waiters: int = Field(default=64, ge=1, le=4096)
    model_cache_entries: int = Field(default=2, ge=1, le=64)
    lock_path: str = ".runtime/resources"


def settings():
    return SchedulerSettings.model_validate(cfg.get("scheduler", default={}))


def _lock(handle, unlock=False):
    if os.name == "nt":
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK if unlock else msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN if unlock else fcntl.LOCK_EX | fcntl.LOCK_NB)


class ResourceManager:
    def __init__(self, policy):
        self.policy = policy
        self.root = local_path(policy.lock_path)

    def _try_slot(self, name, capacity):
        self.root.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(name.encode()).hexdigest()
        for index in range(capacity):
            handle = (self.root / f"{key}-{index}.lock").open("a+b")
            if handle.seek(0, 2) == 0:
                handle.write(b"0")
                handle.flush()
            try:
                _lock(handle)
                return handle
            except OSError as exc:
                handle.close()
                if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                    raise
        return None

    @contextmanager
    def slot(self, name, capacity, *, check=None):
        # Bound active work AND the queue. Never hold a compute slot while waiting
        # for another kind of resource; callers acquire only around the operation.
        if check:
            check()
        queued = self._try_slot("wait:" + name, self.policy.max_waiters)
        if queued is None:
            raise RunConflict(f"Resource queue is full: {name}")
        handle = None
        started = time.perf_counter()
        try:
            while handle is None:
                if check:
                    check()
                handle = self._try_slot(name, capacity)
                if handle is None:
                    time.sleep(0.05)
            _lock(queued, unlock=True)
            queued.close()
            queued = None
            if check:
                check()
            from .context import current_run
            context = current_run.get()
            if context:
                context.store.event(context.run_id, context.epoch, context.token, "resource_admitted",
                                    {"resource": name, "wait_seconds": time.perf_counter() - started,
                                     "task_id": context.task_id})
            yield
        finally:
            for item in (handle, queued):
                if item is not None:
                    _lock(item, unlock=True)
                    item.close()


@lru_cache(maxsize=8)
def _manager(serialized):
    return ResourceManager(SchedulerSettings.model_validate_json(serialized))


def manager():
    return _manager(settings().model_dump_json())


@contextmanager
def resource(name, capacity):
    from .context import current_run
    context = current_run.get()
    with manager().slot(name, capacity, check=context.check if context else None):
        yield


_owner_lock = threading.Lock()
_owners = {}


def accelerator_owner():
    """Keep resident GPU weights in one execution process on this host.

    API in Kafka mode only submits work. An accidental second local inference
    process fails before allocating weights, rather than duplicating GPU memory.
    The owner may itself use multiple configured devices and concurrent runs.
    """
    pool = manager()
    key = (os.getpid(), os.path.normcase(str(pool.root)))
    with _owner_lock:
        if key not in _owners:
            handle = pool._try_slot("resident-accelerator-model-owner", 1)
            if handle is None:
                raise RunConflict("Another process owns local accelerator models; submit through its API/Kafka queue")
            _owners[key] = handle
            atexit.register(handle.close)
