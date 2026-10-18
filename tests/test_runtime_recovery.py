import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from agent_runtime.service import AgentService
from agent_runtime.store import RunStore
from tests.test_agent_runtime import minimal_graph


def test_process_exit_between_tool_result_and_graph_checkpoint_does_not_repeat_effect(tmp_path, repo_root):
    helper = tmp_path / "crash_worker.py"
    helper.write_text('''
from pathlib import Path
import os, sys
from typing import TypedDict
from langgraph.graph import StateGraph, END
from agent_runtime.context import current_run, tool_call
from agent_runtime.store import RunStore
from agent_runtime.service import AgentService
folder=Path(sys.argv[1])
class State(TypedDict):
    user_query: str
    final_report: str
def workflow(saver):
    def node(state):
        current_run.get().begin_node("Lookup",state)
        def read():
            with (folder/"calls").open("a") as log: log.write("executed\\n")
            return "original source"
        value=tool_call("lookup",{"query":state["user_query"]},read)
        if sys.argv[2]=="crash": os._exit(17)
        return {"final_report":value}
    graph=StateGraph(State)
    graph.add_node("Lookup",node)
    graph.set_entry_point("Lookup")
    graph.add_edge("Lookup",END)
    return graph.compile(checkpointer=saver)
service=AgentService(RunStore(folder/"runs.sqlite"),workflow_factory=workflow)
identity=folder/"run_id"
if not identity.exists(): identity.write_text(service.submit({"query":"task"})["run_id"])
row=service.execute(identity.read_text())
assert row["status"]=="succeeded",row
''', encoding="utf-8")
    env = {**os.environ, "PYTHONPATH": repo_root, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    crashed = subprocess.run([sys.executable, "-B", str(helper), str(tmp_path), "crash"], env=env,
                             capture_output=True, timeout=30)
    assert crashed.returncode == 17, crashed.stderr
    store = RunStore(tmp_path / "runs.sqlite")
    run = (tmp_path / "run_id").read_text()
    assert store.get(run)["status"] == "running"
    # Simulate passage of the lease interval without adding 30 seconds to the test.
    with store.connection() as db:
        db.execute("UPDATE runs SET lease_until=0 WHERE run_id=?", (run,))
    recovered = subprocess.run([sys.executable, "-B", str(helper), str(tmp_path), "recover"], env=env,
                               capture_output=True, timeout=30)
    assert recovered.returncode == 0, recovered.stderr
    assert (tmp_path / "calls").read_text().splitlines() == ["executed"]
    assert store.get(run)["status"] == "succeeded"


def test_runtime_bundle_restores_checkpoints_and_resume(tmp_path):
    from tests.test_agent_runtime import minimal_graph
    factory = lambda saver: minimal_graph(saver, pause=True)
    service = AgentService(RunStore(tmp_path / "live" / "runs.sqlite"), workflow_factory=factory)
    run = service.submit({"query": "task"})["run_id"]
    assert service.execute(run)["status"] == "waiting_input"
    backup = service.backup(tmp_path / "backup")
    manifest = json.loads((backup / "manifest.json").read_text())
    assert len(manifest["files"]) == 2
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((backup / name).read_bytes()).hexdigest() == digest
    from agent_runtime.backup import restore_bundle
    restored_path = restore_bundle(backup, tmp_path / "restored")
    restored = AgentService(RunStore(restored_path), workflow_factory=factory)
    restored.resume(run, "recovered facts")
    result = restored.execute(run)
    assert result["status"] == "succeeded" and result["result"]["final_report"] == "recovered facts"


def test_corrupt_backup_cannot_be_restored_or_overwrite_a_directory(tmp_path):
    from agent_runtime.backup import restore_bundle
    service = AgentService(RunStore(tmp_path / "live" / "runs.sqlite"))
    backup = service.backup(tmp_path / "backup")
    with pytest.raises(ValueError, match="new directory"):
        restore_bundle(backup, tmp_path / "live")
    (backup / "runs.sqlite").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        restore_bundle(backup, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_waiting_task_deadline_is_explicitly_expired(tmp_path):
    factory = lambda saver: minimal_graph(saver, pause=True)
    service = AgentService(RunStore(tmp_path / "runs.sqlite"), workflow_factory=factory)
    run = service.submit({"query": "task"})["run_id"]
    assert service.execute(run)["status"] == "waiting_input"
    with service.store.connection() as db:
        db.execute("UPDATE runs SET deadline=0 WHERE run_id=?", (run,))
    service.recover()
    assert service.store.get(run)["status"] == "timed_out"
