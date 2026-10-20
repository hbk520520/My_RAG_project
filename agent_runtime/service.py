"""One execution service for direct calls, HTTP and Kafka job delivery."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from openai import APIConnectionError, APITimeoutError, RateLimitError
from pydantic import ValidationError

from config_loader import cfg
from .context import RunContext, current_run
from .contracts import BudgetExceeded, OutcomeUnknown, ROOT, RunConflict, RunRequest, RunStatus, RunStopped
from .store import RunStore, dumps


logger = logging.getLogger(__name__)


def sandbox_policy_identity(sandbox):
    """Semantic sandbox policy for a run pin.

    Docker-only runs keep the historical shape: the new backend/space keys are
    omitted, so existing pins remain valid.  SPACE identity includes the
    deployment/template semantics but never the bearer token.
    """
    policy = dict(sandbox or {})
    if policy.get("backend", "docker") == "docker":
        policy.pop("backend", None)
        policy.pop("space", None)
        return policy
    space = policy.get("space")
    if isinstance(space, dict):
        sanitized = dict(space)
        sanitized.pop("api_token", None)
        policy["space"] = sanitized
    return policy


def _execution_hashes():
    # Performance switches do not change the durable snapshot/config format.
    # Keep the historical policy fields so existing run pins remain readable.
    policy = {key: cfg.get(key) for key in ("graph", "embedding", "retriever", "causal_evidence", "pageindex", "agent", "sandbox", "scheduler", "vllm")}
    policy["llm"] = {key: value for key, value in cfg.get("llm").items() if key != "api_key"}
    policy["sandbox"] = sandbox_policy_identity(policy.get("sandbox"))
    config_hash = hashlib.sha256(dumps(policy).encode()).hexdigest()
    import prompts
    prompt_hash = hashlib.sha256(Path(prompts.__file__).read_bytes()).hexdigest()
    return config_hash, prompt_hash


class AgentService:
    def __init__(self, store=None, *, engine_factory=None, operator_factory=None, workflow_factory=None):
        self.store = store or RunStore(cfg.get("runtime", "path", default=".runtime/runs.sqlite"))
        self.checkpoint_path = self.store.path.with_name(self.store.path.stem + "-checkpoints.sqlite")
        self.engine_factory = engine_factory or self._engine
        if operator_factory is None:
            from legal_agent.operators import AgenticNodesOperator
            operator_factory = AgenticNodesOperator
        self.operator_factory = operator_factory
        self.workflow_factory = workflow_factory
        self._operator_lock = threading.Lock()
        self._operator = None
        self._index_cache = None
        if engine_factory is None and cfg.get("performance", "snapshot_index_cache", default=False):
            from dataset.index_cache import SnapshotIndexCache
            self._index_cache = SnapshotIndexCache(
                max_entries=cfg.get("performance", "snapshot_cache_entries", default=2),
                max_bytes=cfg.get("performance", "snapshot_cache_max_bytes", default=268435456))

    def submit(self, request, *, transport="local"):
        request = RunRequest.model_validate(request)
        if transport not in ("local", "kafka"):
            raise ValueError("transport must be local or kafka")
        topic = cfg.get("runtime", "pending_topic", default="topic.agent.pending") if transport == "kafka" else None
        return self.store.create(request, queued_topic=topic)

    def _engine(self, project_id):
        from dataset.graph import LegalDenseGraphBuilder
        from .contracts import local_path
        engine = LegalDenseGraphBuilder.from_config(load_memory=False)
        if cfg.get("long_term_memory", "enabled", default=False):
            engine.enable_memory(local_path(cfg.get("long_term_memory", "path")), project_id,
                                 cfg.get("long_term_memory", "embedding_space_id"))
        return engine

    def _pinned_engine(self, row):
        started = time.perf_counter()
        try:
            if self._index_cache is not None:
                return self._cached_pinned_engine(row)
            return self._reference_pinned_engine(row)
        finally:
            logger.info("knowledge_view run_id=%s cached_path=%s seconds=%.6f",
                        row["run_id"], self._index_cache is not None,
                        time.perf_counter() - started)

    def _reference_pinned_engine(self, row):
        """Original construction/restoration path, retained for comparison."""
        from dataset.long_term_memory import graph_snapshot, restore_graph
        engine = self.engine_factory(row["project_id"])
        pin = self.store.pin(row["run_id"], row["epoch"])
        config_hash, prompt_hash = _execution_hashes()
        if pin is None:
            pin = {"snapshot": graph_snapshot(engine), "revision": engine._memory_revision,
                   "config_hash": config_hash, "prompt_hash": prompt_hash}
            self.store.pin(row["run_id"], row["epoch"], pin)
        else:
            if pin["config_hash"] != config_hash or pin["prompt_hash"] != prompt_hash:
                raise ValueError("Run configuration or prompts changed; submit a new run")
            restore_graph(engine, pin["snapshot"])
        # Frozen run view: retrieval must not refresh into a different corpus revision.
        engine._memory_store = None
        return engine

    def _cached_pinned_engine(self, row):
        """Restore a private run view without first rebuilding the live corpus."""
        from dataset.graph import LegalDenseGraphBuilder
        from dataset.long_term_memory import MemoryStore, graph_snapshot, restore_graph
        from .contracts import local_path

        pin = self.store.pin(row["run_id"], row["epoch"])
        config_hash, prompt_hash = _execution_hashes()
        if pin is not None and (pin["config_hash"] != config_hash or pin["prompt_hash"] != prompt_hash):
            raise ValueError("Run configuration or prompts changed; submit a new run")
        engine = LegalDenseGraphBuilder.from_config(load_memory=False)
        memory_enabled = cfg.get("long_term_memory", "enabled", default=False)
        if memory_enabled:
            space_id = cfg.get("long_term_memory", "embedding_space_id")
            if not space_id or not space_id.strip():
                raise ValueError("Long-term memory requires an explicit embedding_space_id")
            engine._memory_space_id = space_id
        new_pin = pin is None
        if new_pin:
            revision, snapshot = 0, None
            if memory_enabled:
                memory = MemoryStore(local_path(cfg.get("long_term_memory", "path")), row["project_id"])
                revision, snapshot = memory.load()
            pin = {"snapshot": snapshot if snapshot is not None else graph_snapshot(engine),
                   "revision": revision, "config_hash": config_hash, "prompt_hash": prompt_hash}
        restore_graph(engine, pin["snapshot"], index_cache=self._index_cache,
                      cache_namespace=row["project_id"])
        engine._memory_revision = pin["revision"]
        if new_pin:
            self.store.pin(row["run_id"], row["epoch"], pin)
        # The cache retains only immutable bytes. Graph, metadata, document
        # reader and deserialized FAISS index remain private to this run.
        engine._memory_store = None
        return engine

    @contextmanager
    def _workflow(self, row, *, inspection=False):
        from legal_agent.gateway import classify
        connection = sqlite3.connect(self.checkpoint_path, check_same_thread=False, timeout=10)
        try:
            from .checkpointer import FencedSqliteSaver
            context = current_run.get()
            saver = FencedSqliteSaver(connection, context) if context else SqliteSaver(connection)
            if self.workflow_factory:
                graph = self.workflow_factory(saver)
            elif inspection or classify(row["request"]["query"]) == "CHITCHAT":
                from legal_agent.workflow import build_plan_replan_agent
                graph = build_plan_replan_agent(None, None, checkpointer=saver)
            else:
                from legal_agent.reasoning import Reasoner
                from legal_agent.workflow import build_plan_replan_agent
                engine = self._pinned_engine(row)
                from .models import shared_encoder
                from dataset.document_store import DocumentReader
                engine.set_encoder(shared_encoder(engine))
                engine._document_reader = DocumentReader.from_engine(engine)
                with self._operator_lock:
                    if self._operator is None:
                        self._operator = self.operator_factory()
                ops = self._operator
                reasoner = Reasoner(engine, lambda text: engine.encode_text(text)["dense"], ops)
                graph = build_plan_replan_agent(reasoner, ops, checkpointer=saver)
            yield graph
        finally:
            connection.close()

    def _config(self, row, checkpoint=None):
        config = {"configurable": {"thread_id": row["run_id"]},
                  "recursion_limit": row["request"]["budgets"]["node_steps"] + 10}
        checkpoint = checkpoint or row.get("checkpoint")
        if checkpoint:
            config["configurable"]["checkpoint_id"] = checkpoint
        return config

    def execute(self, run_id, *, epoch=None):
        from .resources import settings
        token = self.store.claim(run_id, epoch=epoch,
                                 result_topic=cfg.get("runtime", "result_topic", default="topic.agent.result"),
                                 max_running=settings().max_running_runs)
        if token is None:
            return self.store.get(run_id)
        row = self.store.get(run_id)
        context = RunContext(self.store, run_id, row["epoch"], token)
        result_topic = cfg.get("runtime", "result_topic", default="topic.agent.result") if row["transport"] == "kafka" else None
        stop = threading.Event()

        def renew():
            while not stop.wait(5):
                try:
                    self.store.heartbeat(run_id, row["epoch"], token)
                except (RunStopped, sqlite3.Error):
                    return

        heartbeat = threading.Thread(target=renew, name=f"agent-lease-{run_id[:8]}", daemon=True)
        heartbeat.start()
        mark = current_run.set(context)
        final = {}
        try:
            with self._workflow(row) as graph:
                config = self._config(row)
                snapshot = graph.get_state(config)
                if row["resume"] is not None and any(task.interrupts for task in snapshot.tasks):
                    payload = Command(resume=row["resume"])
                elif snapshot.values:
                    payload = None
                else:
                    request = row["request"]
                    payload = {"user_query": request["query"], "task_queue": [], "past_observations": [],
                               "global_facts": [], "completed_tasks": [], "user_facts": [],
                               "retry_context": {}, "recursion_depth": 0,
                               "prediction_cutoff": request["prediction_cutoff"],
                               "causal_targets": request["causal_targets"]}
                    payload["calculation_mode"] = "code" if request["allow_code"] else "structured"
                for value in graph.stream(payload, config, stream_mode="values", durability="sync"):
                    context.check()
                    final = value
                    latest = graph.get_state({"configurable": {"thread_id": run_id}})
                    self.store.set_checkpoint(run_id, row["epoch"], token,
                                              latest.config["configurable"]["checkpoint_id"])
                latest = graph.get_state({"configurable": {"thread_id": run_id}})
                final = latest.values
                waiting = any(task.interrupts for task in latest.tasks)
                if waiting:
                    status, error_kind = RunStatus.WAITING, None
                elif final.get("execution_status") in ("budget_exhausted", "calculation_failed"):
                    status, error_kind = RunStatus.FAILED, final["execution_status"]
                elif latest.next:
                    raise RuntimeError("Workflow stopped with unfinished nodes")
                elif not final.get("final_report"):
                    raise RuntimeError("Workflow completed without a report")
                else:
                    status, error_kind = RunStatus.SUCCEEDED, None
                context.check()
                self.store.finish(run_id, row["epoch"], token, status, result=final, error_kind=error_kind,
                                  result_topic=result_topic)
        except Exception as exc:
            if isinstance(exc, BudgetExceeded):
                status, kind = RunStatus.TIMED_OUT, "budget"
            elif isinstance(exc, OutcomeUnknown):
                status, kind = RunStatus.FAILED, "outcome_unknown"
            elif isinstance(exc, RunConflict):
                status, kind = RunStatus.FAILED, "resource"
            elif isinstance(exc, (APIConnectionError, APITimeoutError, RateLimitError, ConnectionError)):
                status, kind = RunStatus.FAILED, "transport"
            elif isinstance(exc, (ValidationError, ValueError, json.JSONDecodeError)):
                status, kind = RunStatus.FAILED, "validation"
            elif isinstance(exc, RunStopped):
                status, kind = RunStatus.CANCELLED, "ownership_lost"
            else:
                status, kind = RunStatus.FAILED, "internal"
            self.store.finish(run_id, row["epoch"], token, status, result=final or None,
                              error=f"{type(exc).__name__}: {exc}", error_kind=kind,
                              result_topic=result_topic)
        finally:
            stop.set()
            heartbeat.join(timeout=1)
            current_run.reset(mark)
            session = final.get("sandbox_session_id")
            if session:
                from agent_tools.sandbox.executor import destroy_session
                destroy_session(session)
        return self.store.get(run_id)

    def resume(self, run_id, answer, *, transport="local"):
        if not isinstance(answer, str) or not answer.strip() or len(answer) > 20000:
            raise ValueError("Answer must contain 1–20000 characters")
        topic = cfg.get("runtime", "pending_topic", default="topic.agent.pending") if transport == "kafka" else None
        return self.store.requeue(run_id, answer=answer, topic=topic)

    def checkpoints(self, run_id):
        row = self.store.get(run_id)
        with self._workflow(row, inspection=True) as graph:
            return [{"checkpoint_id": state.config["configurable"]["checkpoint_id"],
                     "next": list(state.next), "created_at": state.created_at}
                    for state in graph.get_state_history({"configurable": {"thread_id": run_id}}, limit=100)]

    def rewind(self, run_id, checkpoint, *, transport="local"):
        row = self.store.get(run_id)
        with self._workflow(row, inspection=True) as graph:
            state = graph.get_state(self._config(row, checkpoint))
            if not state.values:
                raise ValueError("Checkpoint is not part of this run")
        topic = cfg.get("runtime", "pending_topic", default="topic.agent.pending") if transport == "kafka" else None
        return self.store.rewind(run_id, checkpoint, topic=topic)

    def recover(self):
        from .resources import settings
        self.store.expire_waiting(cfg.get("runtime", "result_topic", default="topic.agent.result"))
        pending = self.store.recoverable()[:settings().max_running_runs]
        if not pending:
            return []
        with ThreadPoolExecutor(max_workers=len(pending), thread_name_prefix="agent-run") as pool:
            return list(pool.map(self.execute, pending))

    def backup(self, destination):
        """Publish a manifest only after both runtime databases pass integrity checks."""
        from .contracts import RunConflict, local_path
        target = local_path(destination)
        if target.exists():
            raise ValueError("Backup directory must not already exist")
        # Hold the authority write lock: a new executor cannot start during backup.
        with self.store.connection() as guard:
            guard.execute("BEGIN IMMEDIATE")
            if guard.execute("SELECT 1 FROM runs WHERE status=? AND lease_until>?",
                             (RunStatus.RUNNING, time.time())).fetchone():
                raise RunConflict("Wait for the running task to finish before backup")
            target.mkdir(parents=True)
            self.store.backup(target / self.store.path.name)
            files = [self.store.path.name]
            if self.checkpoint_path.exists():
                source = sqlite3.connect(self.checkpoint_path)
                copied = sqlite3.connect(target / self.checkpoint_path.name)
                try:
                    source.backup(copied)
                    if copied.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                        raise RuntimeError("Checkpoint backup is corrupt")
                finally:
                    copied.close()
                    source.close()
                files.append(self.checkpoint_path.name)
            manifest = {"format": 1, "run_db": self.store.path.name,
                        "files": {name: hashlib.sha256((target / name).read_bytes()).hexdigest()
                                               for name in files}}
            (target / "manifest.json").write_text(dumps(manifest), encoding="utf-8")
        return target
