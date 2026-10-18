"""Bounded DAG execution. Workers return deltas; only the coordinator merges state.

Inspired by TokenSpeed's plan / execution / completion separation. LangGraph keeps
the coordinator checkpoints; the existing effect ledger persists branch outcomes.
"""
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
import hashlib

from config_loader import cfg
from agent_runtime.context import current_run, task_context, tool_call
from agent_runtime.contracts import TaskSpec
from agent_runtime.resources import settings


def tasks(queue):
    return [TaskSpec.model_validate({"task_id": f"task-{i}", **task} if isinstance(task, dict)
                                   else {"task_id": f"task-{i}", "task_desc": task}).model_dump()
            for i, task in enumerate(queue)]


def validate_plan(queue, completed=()):
    remaining = {task["task_id"]: task for task in queue}
    done = {task["task_id"] for task in completed}
    if len(remaining) != len(queue) or done & remaining.keys():
        raise ValueError("Task IDs must be unique across pending and completed tasks")
    known = done | remaining.keys()
    if any(not set(task["deps"]) <= known for task in queue):
        raise ValueError("Unknown task dependency")
    visited = set(done)
    while remaining:
        ready = [key for key, task in remaining.items() if set(task["deps"]) <= visited]
        if not ready:
            raise ValueError("Task dependency cycle")
        for key in ready:
            visited.add(key)
            del remaining[key]


def user_facts_for(state, task):
    scoped = state.get("task_user_facts", {})
    facts = list(scoped.get(task["task_id"], state.get("user_facts", []) if not scoped else []))
    for dependency in task["deps"]:
        facts.extend(scoped.get(dependency, []))
        facts.extend(state.get("task_results", {}).get(dependency, {}).get("user_facts", []))
    return list(dict.fromkeys(facts))


def execute_ready(state, reasoner):
    from .nodes import node_executor
    queue = tasks(state["task_queue"])
    if not queue:
        return {}
    completed = state.get("completed_tasks", [])
    validate_plan(queue, completed)
    depth = state.get("recursion_depth", 0)
    remaining = cfg.get("agent", "max_recursion_depth", default=5) - depth
    if remaining <= 0:
        return {"retry_context": {"status": "force_stop"}, "execution_status": "budget_exhausted"}
    done = {task["task_id"] for task in completed}
    # A question belongs to one task. Do not repeat a completed sibling on resume.
    ready = [task for task in queue if set(task["deps"]) <= done]
    selected = ready[:min(settings().max_parallel_tasks, remaining)]
    parent = current_run.get()

    def execute(task):
        with task_context(task["task_id"], task["revision"]) as context:
            # Inputs are read-only. Copy the container, not every source text and
            # causal audit accumulated by previous tasks.
            branch = dict(state)
            branch["task_queue"] = [task]
            branch["user_facts"] = user_facts_for(state, task)
            branch["retry_context"] = state.get("task_retry_contexts", {}).get(task["task_id"], {})
            dependency_facts = [state.get("task_results", {}).get(dep, {}).get("facts", []) for dep in task["deps"]]
            worker = reasoner.fork(task["evidence"], dependency_facts)
            def call():
                if context:
                    context.store.consume(context.run_id, context.epoch, context.token, node_steps=1)
                return node_executor(branch, worker)
            return tool_call("task.execute", {"task": task, "query": state["user_query"],
                             "user_facts": branch["user_facts"], "dependencies": dependency_facts,
                             "cutoff": state.get("prediction_cutoff"), "targets": state.get("causal_targets"),
                             "retry": branch["retry_context"]}, call)

    # copy_context is distinct for every worker: ContextVar values must not point
    # to a shared mutable RunContext. task_context replaces that value above.
    with ThreadPoolExecutor(max_workers=len(selected), thread_name_prefix="agent-task") as pool:
        futures = [pool.submit(copy_context().run, execute, task) for task in selected]
        outcomes = [future.result() for future in futures]
    if parent:
        parent.check()
    delta = {"task_queue": [], "completed_tasks": list(completed),
             "global_facts": list(state.get("global_facts", [])), "past_observations": [],
             "causal_evidence": [], "task_retry_contexts": dict(state.get("task_retry_contexts", {})),
             "source_evidence": [],
             "task_results": dict(state.get("task_results", {})), "recursion_depth": depth + len(selected),
             "retry_context": {}, "pending_question": None, "pending_task_id": None}
    by_id = dict(zip((task["task_id"] for task in selected), outcomes))
    for task in queue:
        identity = task["task_id"]
        if identity not in by_id:
            delta["task_queue"].append(task)
            continue
        outcome = by_id[identity]
        delta["past_observations"].extend(outcome.get("past_observations", []))
        delta["causal_evidence"].extend(outcome.get("causal_evidence", []))
        delta["source_evidence"].extend(outcome.get("source_evidence", []))
        if outcome.get("pending_question"):
            delta["task_queue"].append(task)
            if delta["pending_question"] is None:
                delta.update(pending_question=outcome["pending_question"], pending_task_id=identity,
                             execution_status="waiting_input")
            continue
        added = outcome.get("global_facts", [])[len(state.get("global_facts", [])):]
        delta["global_facts"].extend(fact for fact in added if fact not in delta["global_facts"])
        if any(t.get("task_id") == identity for t in outcome.get("completed_tasks", [])):
            delta["completed_tasks"].append(task)
            delta["task_results"][identity] = {"facts": added, "revision": task["revision"],
                                              "user_facts": user_facts_for(state, task)}
            delta["task_retry_contexts"].pop(identity, None)
        else:
            delta["task_queue"].extend(outcome.get("task_queue", []))
            delta["task_retry_contexts"][identity] = outcome.get("retry_context", {})
    return delta


def replan_pending(state, ops):
    from .nodes import node_replanner
    if state.get("retry_context", {}).get("status") == "force_stop":
        return {"task_queue": []}
    retries = dict(state.get("task_retry_contexts", {}))
    queue = []
    for task in tasks(state["task_queue"]):
        identity = task["task_id"]
        if identity not in retries:
            queue.append(task)
            continue
        retry = retries[identity]
        needs_replan = retry.get("status") == "irrelevant" or (
            retry.get("status") == "partial" and retry.get("fail_count", 0) > cfg.get("agent", "wormhole_threshold", default=2))
        if needs_replan:
            with task_context(identity, task["revision"]):
                update = node_replanner({**state, "task_queue": [task], "retry_context": retry}, ops)
        else:
            update = {"task_queue": [task], "retry_context": retry}
        replacements = tasks(update.get("task_queue", [task]))
        for replacement in replacements:
            # IDs remain stable for dependents; new intermediate nodes are scoped
            # to the failed branch, not to completion order or another branch.
            replacement["revision"] = task["revision"] + 1
        queue.extend(replacements)
        retries[identity] = update.get("retry_context", {})
    # Run completeness rules once on the merged view, including the last task.
    # Deterministic IDs prevent duplicate additions after recovery/replanning.
    from replanner_rules import apply_hard_rules
    completed = state.get("completed_tasks", [])
    known = {task["task_desc"] for task in queue + completed}
    predecessors = [task["task_id"] for task in completed]
    for addition in apply_hard_rules("\n".join(state.get("past_observations", [])), queue + completed):
        if addition["task_desc"] in known:
            continue
        identity = "rule-" + hashlib.sha256(addition["task_desc"].encode()).hexdigest()[:16]
        queue.append(TaskSpec(task_id=identity, deps=predecessors, evidence={"causal": "off", "analysis": "reasoning"},
                              **addition).model_dump())
        predecessors = [identity]
        known.add(addition["task_desc"])
    validate_plan(queue, state.get("completed_tasks", []))
    return {"task_queue": queue, "task_retry_contexts": retries, "retry_context": {}}
