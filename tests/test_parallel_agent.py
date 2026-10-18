"""Offline concurrency, provenance policy, replay and cross-process admission."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import copy
import json
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from agent_runtime.context import RunContext, current_run, task_context
from agent_runtime.contracts import EvidencePolicy, RunConflict, RunRequest, RunStopped, TaskSpec
from agent_runtime.resources import ResourceManager, SchedulerSettings
from agent_runtime.service import AgentService
from agent_runtime.store import RunStore
from config_loader import cfg
from legal_agent.scheduling import execute_ready, replan_pending, validate_plan


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    import tiktoken
    from tests.test_causal_evidence import ByteEncoding
    monkeypatch.setattr(tiktoken, "get_encoding", lambda name: ByteEncoding())


def task(identity, deps=(), **policy):
    return TaskSpec(task_id=identity, task_desc=identity, deps=list(deps), evidence=EvidencePolicy(**policy)).model_dump()


def state(queue):
    return {"user_query": "调查材料", "task_queue": queue, "completed_tasks": [],
            "global_facts": [], "past_observations": [], "recursion_depth": 0}


def merge(before, delta):
    result = {**before, **delta}
    for key in ("past_observations", "causal_evidence", "source_evidence"):
        result[key] = before.get(key, []) + delta.get(key, [])
    return result


class Worker:
    def __init__(self, call):
        self.call = call
        self.dependencies = []

    def fork(self, policy, dependencies):
        worker = copy(self)
        worker.dependencies = dependencies
        return worker

    def answer(self, query, **kwargs):
        return self.call(query, self.dependencies, kwargs)


def answer(query):
    return {"fact": "fact:" + query, "reasoning": "", "sufficient": True, "grader_status": "sufficient"}


class Ops:
    def __init__(self, queue):
        self.queue = queue

    def generate_plan(self, query):
        return self.queue

    def generate_final_report(self, query, observations):
        return json.dumps(observations, ensure_ascii=False)


def service(tmp_path, queue, call):
    from legal_agent.workflow import build_plan_replan_agent
    return AgentService(RunStore(tmp_path / "runs.sqlite"), workflow_factory=lambda saver:
                        build_plan_replan_agent(Worker(call), Ops(queue), checkpointer=saver))


def test_independent_tasks_overlap_and_dependency_waits_for_both():
    barrier = threading.Barrier(2)
    seen = []
    def call(query, dependencies, kwargs):
        if query in ("a", "b"):
            barrier.wait(timeout=3)
        else:
            assert dependencies == [["fact:a"], ["fact:b"]]
        seen.append(query)
        return answer(query)
    initial = state([task("a"), task("b"), task("c", ("a", "b"))])
    first = merge(initial, execute_ready(initial, Worker(call)))
    assert [t["task_id"] for t in first["completed_tasks"]] == ["a", "b"]
    assert seen.count("c") == 0
    final = merge(first, execute_ready(first, Worker(call)))
    assert [t["task_id"] for t in final["completed_tasks"]] == ["a", "b", "c"]
    assert final["global_facts"] == ["fact:a", "fact:b", "fact:c"]


@pytest.mark.parametrize("queue,completed", [
    ([task("a"), task("a")], []), ([task("a", ("missing",))], []),
    ([task("a", ("b",)), task("b", ("a",))], []), ([task("a")], [task("a")])])
def test_invalid_dependency_plans_fail_before_execution(queue, completed):
    with pytest.raises(ValueError):
        validate_plan(queue, completed)


def test_successful_branch_is_reused_after_sibling_failure_and_restart(tmp_path):
    calls = Counter()
    barrier = threading.Barrier(2)
    lock = threading.Lock()
    def call(query, dependencies, kwargs):
        with lock:
            calls[query] += 1
            attempt = calls[query]
        if attempt == 1:
            barrier.wait(timeout=3)
        if query == "b" and attempt == 1:
            raise ConnectionError("transient failure")
        return answer(query)
    original = service(tmp_path, [task("a"), task("b")], call)
    run = original.submit({"query": "调查材料"})["run_id"]
    assert original.execute(run)["status"] == "failed"
    original.store.requeue(run, retry=True)
    reopened = service(tmp_path, [task("a"), task("b")], call)
    result = reopened.execute(run)
    assert result["status"] == "succeeded", result["error"]
    assert calls == {"a": 1, "b": 2}
    assert result["result"]["global_facts"] == ["fact:a", "fact:b"]


def test_two_user_questions_are_resumed_individually_without_repeating_completed_work(tmp_path):
    calls = Counter()
    def call(query, dependencies, kwargs):
        calls[query] += 1
        if query not in kwargs.get("user_facts", []):
            return {**answer(query), "sufficient": False, "missing_info_kind": "user",
                    "suggestion": "Please answer " + query}
        return answer(query)
    worker = service(tmp_path, [task("a"), task("b")], call)
    run = worker.submit({"query": "调查材料"})["run_id"]
    first = worker.execute(run)
    assert first["status"] == "waiting_input" and first["result"]["pending_task_id"] == "a"
    worker.resume(run, "a")
    second = worker.execute(run)
    assert second["status"] == "waiting_input" and second["result"]["pending_task_id"] == "b"
    before = calls["a"]
    worker.resume(run, "b")
    final = worker.execute(run)
    assert final["status"] == "succeeded", final["error"]
    assert calls["a"] == before
    assert final["result"]["task_user_facts"] == {"a": ["a"], "b": ["b"]}


def test_parallel_replanning_scopes_intermediate_ids_and_retains_dependents():
    class ReplanOps:
        def replan_with_wormhole(self, **kwargs):
            return [task("intermediate"), task("replacement")]
    pending = state([task("a"), task("b"), task("c", ("a", "b"))])
    pending["task_retry_contexts"] = {key: {"status": "irrelevant"} for key in ("a", "b")}
    result = replan_pending(pending, ReplanOps())
    ids = [item["task_id"] for item in result["task_queue"]]
    assert len(set(ids)) == len(ids) == 5
    assert set(result["task_queue"][-1]["deps"]) == {"a", "b"}
    assert result["task_queue"][1]["task_id"] == "a"


def test_branch_effect_identity_and_atomic_budget(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite")
    row = store.create(RunRequest(query="q"))
    token = store.claim(row["run_id"])
    root = RunContext(store, row["run_id"], 0, token)
    root.begin_node("Executor", {"recursion_depth": 0})
    mark = current_run.set(root)
    try:
        for identity, revision in (("a", 0), ("b", 0), ("a", 1)):
            with task_context(identity, revision) as branch:
                assert branch is not root
                assert branch.tool("read", {"same": True}, lambda: identity + str(revision)) == identity + str(revision)
        assert root.task_id == ""
    finally:
        current_run.reset(mark)
    with store.connection() as db:
        assert db.execute("SELECT count(*) FROM effects").fetchone()[0] == 3


def test_resource_limits_span_manager_instances_and_release_after_failure(tmp_path):
    policy = SchedulerSettings(lock_path=str(tmp_path / "locks"))
    a, b = ResourceManager(policy), ResourceManager(policy)
    def stopped():
        raise RunStopped("cancelled")
    with a.slot("device:0", 1):
        count = 0
        def check():
            nonlocal count
            count += 1
            if count > 2:
                stopped()
        with pytest.raises(RunStopped):
            with b.slot("device:0", 1, check=check):
                pytest.fail("oversubscribed")
    with b.slot("device:0", 1):
        pass


def test_os_lock_is_released_after_process_exit(tmp_path, repo_root):
    child = '''
import sys
from config_loader import cfg
from agent_runtime.resources import accelerator_owner
cfg._data["scheduler"]["lock_path"]=sys.argv[1]
accelerator_owner()
print("locked",flush=True)
sys.stdin.readline()
'''
    process = subprocess.Popen([sys.executable, "-B", "-c", child, str(tmp_path / "locks")],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, encoding="utf-8", cwd=repo_root,
                               env={**os.environ, "PYTHONPATH": repo_root, "PYTHONIOENCODING": "utf-8"})
    try:
        assert process.stdout.readline().strip() == "locked"
        manager = ResourceManager(SchedulerSettings(lock_path=str(tmp_path / "locks")))
        assert manager._try_slot("resident-accelerator-model-owner", 1) is None
        process.terminate()
        process.wait(timeout=5)
        with manager.slot("resident-accelerator-model-owner", 1):
            pass
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


def test_multiple_runs_execute_concurrently_with_shared_lease_limit(tmp_path):
    barrier = threading.Barrier(2)
    def call(query, dependencies, kwargs):
        barrier.wait(timeout=3)
        return answer(query)
    runner = service(tmp_path, [task("a")], call)
    runs = [runner.submit({"query": "调查材料"})["run_id"] for _ in range(2)]
    result = runner.recover()
    assert {row["run_id"] for row in result} == set(runs)
    assert all(row["status"] == "succeeded" for row in result), result


def test_shared_encoder_uses_model_identity_instead_of_project_namespace(monkeypatch, offline_engine):
    from agent_runtime.models import shared_encoder
    first = shared_encoder(offline_engine)
    assert first is shared_encoder(offline_engine)
    offline_engine._memory_space_id = "different-version"
    assert first is not shared_encoder(offline_engine)


def test_reasoning_is_separate_from_grading_rationale():
    from legal_agent.reasoning import Reasoner
    calls = []
    ops = SimpleNamespace(grade_facts=lambda *args: {"status": "sufficient", "extracted_facts": ["fact"], "rationale": "grading only"},
                          reason=lambda *args: calls.append(args) or "actual reasoning")
    reasoner = Reasoner(None, None, ops)
    plain = reasoner.fork({"causal": "off"}, [])._answer_from_docs("q", "q", "source")
    assert plain["reasoning"] == "" and plain["grading_rationale"] == "grading only"
    result = reasoner.fork({"analysis": "reasoning"}, [])._answer_from_docs("q", "q", "source")
    assert result["reasoning"] == "actual reasoning" and len(calls) == 1


def test_temporal_policy_requires_inputs_before_retrieval():
    from legal_agent.reasoning import Reasoner
    reasoner = Reasoner(None, None, None).fork({"causal": "temporal"}, [])
    with pytest.raises(ValueError, match="cutoff"):
        reasoner.answer("q", "q")


def test_greeting_uses_shared_gateway_without_loading_models(tmp_path):
    runner = AgentService(RunStore(tmp_path / "runs.sqlite"), engine_factory=lambda _: pytest.fail("loaded engine"),
                          operator_factory=lambda: pytest.fail("loaded model"))
    run = runner.submit({"query": "你好"})["run_id"]
    result = runner.execute(run)
    assert result["status"] == "succeeded" and result["usage"]["model_calls"] == 0


def test_model_role_reaches_local_adapter_and_remote_model(monkeypatch):
    from legal_agent.llm import complete, model_role
    seen = []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])
    local = SimpleNamespace(chat=lambda *args, **kwargs: seen.append(kwargs) or response)
    monkeypatch.setitem(cfg._data["vllm"], "role_adapters", {"reasoner": "reasoner"})
    with model_role("reasoner"):
        assert complete(local, "base", [{"role": "user", "content": "q"}]) == "ok"
    assert seen[-1]["lora_name"] == "reasoner"
    remote = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: seen.append(kwargs) or response)))
    monkeypatch.setitem(cfg._data["llm"], "role_models", {"planner": "planner-model"})
    with model_role("planner"):
        assert complete(remote, "base", [{"role": "user", "content": "q"}]) == "ok"
    assert seen[-1]["model"] == "planner-model"


def test_cancellation_fences_all_late_branch_results(tmp_path):
    entered = threading.Barrier(3)
    release = threading.Event()
    def call(query, dependencies, kwargs):
        entered.wait(timeout=3)
        assert release.wait(timeout=3)
        return answer(query)
    runner = service(tmp_path, [task("a"), task("b")], call)
    run = runner.submit({"query": "调查材料"})["run_id"]
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(runner.execute, run)
        entered.wait(timeout=3)
        runner.store.cancel(run)
        release.set()
        assert future.result(timeout=5)["status"] == "cancelled"
    with runner.store.connection() as db:
        assert db.execute("SELECT count(*) FROM effects WHERE tool='task.execute' AND status='done'").fetchone()[0] == 0


def test_parallel_budget_reservation_never_exceeds_limit(tmp_path):
    from agent_runtime.contracts import Budgets, BudgetExceeded
    store = RunStore(tmp_path / "runs.sqlite")
    row = store.create(RunRequest(query="q", budgets=Budgets(model_calls=1)))
    token = store.claim(row["run_id"])
    def reserve(identity):
        context = RunContext(store, row["run_id"], 0, token, task_id=identity)
        try:
            context.model_budget([{"content": "q"}], 1)
            return True
        except BudgetExceeded:
            return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(reserve, ("a", "b"))) == [False, True]
    assert store.get(row["run_id"])["usage"]["model_calls"] == 1


def test_run_quota_is_shared_across_store_instances(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite")
    runs = [store.create(RunRequest(query="q"))["run_id"] for _ in range(3)]
    a = store.claim(runs[0], max_running=2)
    reopened = RunStore(store.path)
    assert reopened.claim(runs[1], max_running=2)
    assert reopened.claim(runs[2], max_running=2) is None
    store.finish(runs[0], 0, a, "succeeded")
    assert reopened.claim(runs[2], max_running=2)


def test_noncausal_tasks_preserve_original_pages_and_citations(monkeypatch):
    from legal_agent.reasoning import Reasoner
    from legal_agent.workflow import build_plan_replan_agent
    from tests.test_pageindex import imported, TEXT, cited
    from tests.test_chain_integration import make_ops
    graph = imported()
    ops = make_ops()
    ops.generate_plan = lambda query: [TaskSpec(task_id="read", task_desc="读取原文", evidence={"causal": "off"}).model_dump()]
    original = ops._call_messages
    def call(messages, **kwargs):
        if "Causal Extraction:" in messages[0]["content"]:
            pytest.fail("Unnecessary causal extraction")
        if "Read evidence catalog:" in messages[-1]["content"]:
            return cited()
        return original(messages, **kwargs)
    ops._call_messages = call
    result = build_plan_replan_agent(Reasoner(graph, None, ops), ops).invoke(state([]))
    assert result["causal_evidence"] == []
    assert result["source_evidence"][0]["graph"]["source_chunks"][0]["text"] == TEXT
    assert result["citations"][0]["source_file"] == "news.pdf"


def test_durable_router_preserves_chain_and_normalizes_date_inputs(tmp_path, monkeypatch):
    from query import UnifiedQueryRouter_Query
    from tests.test_chain_integration import make_ops, sources, CUTOFF
    from tests.test_causal_evidence import ByteEncoding
    from causal_evidence import graph as module
    # Build through the public fixture function's original factory, then use a
    # real AgentService rather than the historical routing compatibility path.
    engine = sources.__wrapped__()
    ops = make_ops()
    original = module.chunk_record
    monkeypatch.setattr(module, "chunk_record", lambda record, **kwargs: original(record, encoding=ByteEncoding()))
    runner = AgentService(RunStore(tmp_path / "runs.sqlite"), engine_factory=lambda _: engine, operator_factory=lambda: ops)
    router = UnifiedQueryRouter_Query.__new__(UnifiedQueryRouter_Query)
    router._agent_service = runner
    result = router.process("预算削减是否导致资金紧张？", prediction_cutoff=CUTOFF, causal_targets=["资金紧张"])
    assert result["run_status"] == "succeeded", result.get("agent_error")
    assert result["agent_causal_evidence"][0]["weighting_status"] == "scored"
    assert runner.store.get(result["run_id"])["request"]["prediction_cutoff"] == CUTOFF.isoformat()


def test_completed_branch_survives_process_exit_before_coordinator_checkpoint(tmp_path, repo_root):
    helper = tmp_path / "parallel_crash.py"
    helper.write_text('''
import os,sys,time
from pathlib import Path
from agent_runtime.context import current_run
from agent_runtime.service import AgentService
from agent_runtime.store import RunStore
from legal_agent.workflow import build_plan_replan_agent
from tests.test_parallel_agent import Worker,Ops,task,answer
folder=Path(sys.argv[1])
def call(query,dependencies,kwargs):
 with (folder/(query+".calls")).open("a") as log: log.write("call\\n")
 if query=="b" and sys.argv[2]=="crash":
  context=current_run.get()
  until=time.monotonic()+5
  while time.monotonic()<until:
   with context.store.connection() as db:
    done=db.execute("SELECT 1 FROM effects WHERE tool='task.execute' AND status='done'").fetchone()
   if done: os._exit(17)
   time.sleep(.02)
  raise RuntimeError("No branch committed")
 return answer(query)
def workflow(saver):
 return build_plan_replan_agent(Worker(call),Ops([task("a"),task("b")]),checkpointer=saver)
runner=AgentService(RunStore(folder/"runs.sqlite"),workflow_factory=workflow)
identity=folder/"run_id"
if not identity.exists(): identity.write_text(runner.submit({"query":"investigate"})["run_id"])
row=runner.execute(identity.read_text())
assert row["status"]=="succeeded",row
''', encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": repo_root, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    crashed = subprocess.run([sys.executable, "-B", str(helper), str(tmp_path), "crash"],
                             capture_output=True, env=env, timeout=30)
    assert crashed.returncode == 17, crashed.stderr
    store = RunStore(tmp_path / "runs.sqlite")
    run = (tmp_path / "run_id").read_text()
    with store.connection() as db:
        db.execute("UPDATE runs SET lease_until=0 WHERE run_id=?", (run,))
    restored = subprocess.run([sys.executable, "-B", str(helper), str(tmp_path), "restore"],
                              capture_output=True, env=env, timeout=30)
    assert restored.returncode == 0, restored.stderr
    assert (tmp_path / "a.calls").read_text().splitlines() == ["call"]
    assert (tmp_path / "b.calls").read_text().splitlines() == ["call", "call"]


@pytest.mark.parametrize("value", [False, 123, [], "2025-02-30"])
def test_invalid_cutoff_is_a_validation_error(value):
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        RunRequest(query="q", prediction_cutoff=value)


def test_expired_executor_cannot_revive_and_exceed_running_quota(tmp_path):
    store = RunStore(tmp_path / "runs.sqlite")
    a, b = [store.create(RunRequest(query="q"))["run_id"] for _ in range(2)]
    old = store.claim(a)
    with store.connection() as db:
        db.execute("UPDATE runs SET lease_until=0 WHERE run_id=?", (a,))
    assert store.claim(b)
    with pytest.raises(RunStopped, match="expired"):
        store.heartbeat(a, 0, old)
    assert not store.finish(a, 0, old, "succeeded", result={"late": True})
    assert a in store.recoverable()


def test_calibration_identity_tracks_extractor_and_verifier_model_overrides(monkeypatch):
    from causal_evidence.pipeline import extraction_identity
    before = extraction_identity()
    monkeypatch.setitem(cfg._data["llm"], "role_models", {"extractor": "new-extractor"})
    extracted = extraction_identity()
    assert before != extracted
    monkeypatch.setitem(cfg._data["llm"], "role_models", {"extractor": "new-extractor", "verifier": "new-verifier"})
    assert extracted != extraction_identity()
