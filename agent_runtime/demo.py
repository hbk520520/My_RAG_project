"""Offline durable execution: pause, reopen, resume and rewind without any model."""
import json
import tempfile
from typing import TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from .context import current_run, tool_call
from .contracts import ROOT
from .service import AgentService
from .store import RunStore


class State(TypedDict):
    user_query: str
    answer: str
    final_report: str


def workflow(saver):
    def read(state):
        current_run.get().begin_node("Read", state)
        return {"answer": tool_call("lookup", {"query": state["user_query"]}, lambda: "original evidence")}
    def finish(state):
        current_run.get().begin_node("Finish", state)
        answer = interrupt({"question": "Supply the missing case fact"})
        return {"final_report": state["answer"] + "; " + answer}
    graph = StateGraph(State)
    graph.add_node("Read", read)
    graph.add_node("Finish", finish)
    graph.set_entry_point("Read")
    graph.add_edge("Read", "Finish")
    graph.add_edge("Finish", END)
    return graph.compile(checkpointer=saver)


def main():
    folder = ROOT / "outputs" / "temp"
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="agent-demo-", dir=folder) as path:
        service = AgentService(RunStore(path + "/runs.sqlite"), workflow_factory=workflow)
        run = service.submit({"query": "offline task"})["run_id"]
        assert service.execute(run)["status"] == "waiting_input"
        reopened = AgentService(RunStore(service.store.path), workflow_factory=workflow)
        reopened.resume(run, "verified by user")
        result = reopened.execute(run)
        assert result["status"] == "succeeded"
        assert "verified by user" in result["result"]["final_report"]
        checkpoint = next(item["checkpoint_id"] for item in reopened.checkpoints(run) if item["next"] == ["Finish"])
        reopened.rewind(run, checkpoint)
        assert reopened.execute(run, epoch=0)["status"] == "queued"
        assert reopened.execute(run, epoch=1)["status"] == "waiting_input"
        reopened.resume(run, "corrected user fact")
        assert reopened.execute(run)["status"] == "succeeded"
        print(json.dumps({"pause": "passed", "reopen_resume": "passed", "rewind": "passed", "stale_epoch": "rejected"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
