"""
P0 技术验证 Spike —— 持久化执行 go / no-go
=========================================
目标：在写任何生产代码之前，先证明「Agent 可以离开进程再回来」这条技术路线成立。

要验证的 4 件事（任一失败 → 整个 P1 方案要改）：

  V1  检查点落盘        invoke 之后状态真的被持久化（不是只活在内存里）
  V2  真挂起            `interrupt()` 后图**停止执行**，进程可以正常退出
  V3  跨进程恢复        **另起一个进程**、重新打开同一个 sqlite，用 `Command(resume=…)`
                        从中断点继续 —— 这是"关掉 Worker 系统挂起、重启后接续"的技术基础
  V4  幂等            同一 `request_id` 重复投递只应产生一次副作用

为什么用「另起进程」而不是「同一个进程里再 invoke 一次」：
  后者只证明内存里有状态，证明不了**持久化**。V3 才真正回答了
  「进程被 kill -9 之后能不能恢复」这个问题。

Kafka 说明：
  本机没有 Kafka，所以 V3 用**内存任务总线**模拟"结果从别的进程回来"这一步。
  总线是可替换的传输层；spike 要验的是 **LangGraph 的挂起/恢复语义**，
  不是 Kafka 的可靠性语义（那是独立且低风险的工作）。

⚠️ 必须记住的一条 LangGraph 契约（spike 会复现它）：
   节点里 `interrupt()` **之前**的代码在恢复时会被**重新执行**。
   所以「下发任务」必须在 interrupt 之前做幂等保护，否则重试会重复派发。
   V4 就是专门验这一条。

运行：
    python spikes/spike_checkpoint_resume.py              # 跑全部 4 项验证
    python spikes/spike_checkpoint_resume.py --phase suspend   # 内部用：只挂起
    python spikes/spike_checkpoint_resume.py --phase resume    # 内部用：只恢复
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from typing import Annotated, Any, Dict, List, TypedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

# 运行产物（幂等台账 + 内存总线）必须被父进程和所有子进程共享，所以用环境变量传递目录。
# ⚠️ 必须**惰性求值**：模块级常量在 import 时就固化了，
# 而驱动是在 import 之后才建临时目录的 —— 写成常量会导致切目录无效。
def state_path(name: str) -> str:
    return os.path.join(os.environ.get("SPIKE_P0_STATE_DIR") or _HERE, name)


TASK_LEDGER_NAME = ".spike_task_ledger.sqlite"
BUS_NAME = ".spike_bus.jsonl"


# ===========================================================================
# 幂等台账：模拟"把任务写进 Kafka"这件事的副作用记录
# ===========================================================================
class TaskLedger:
    """
    外部幂等台账。

    为什么不能用 state 记「已派发」：
      LangGraph **不会提交一个未完成节点的状态增量**。节点在 `interrupt()` 处挂起时，
      它的 return 根本没执行，所以恢复时节点是**从头重跑**的，state 里没有上次的痕迹。
      因此「是否已派发」必须记在 **state 之外**（真实系统里就是 Kafka 生产者 + 去重表）。

    两张表分开记：
      attempts  每次调用都记 → 用来证明"节点确实重跑了"
      dispatched 主键去重   → 用来证明"副作用只发生一次"
    """

    def __init__(self, path: str = None):
        self.path = path or state_path(TASK_LEDGER_NAME)
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with sqlite3.connect(self.path) as con:
            con.execute("CREATE TABLE IF NOT EXISTS dispatched "
                        "(request_id TEXT PRIMARY KEY, at INTEGER)")
            con.execute("CREATE TABLE IF NOT EXISTS attempts "
                        "(request_id TEXT, at INTEGER)")

    def note_attempt(self, request_id: str) -> None:
        with sqlite3.connect(self.path) as con:
            con.execute("INSERT INTO attempts(request_id, at) "
                        "VALUES(?, strftime('%s','now'))", (request_id,))

    def dispatch_once(self, request_id: str) -> bool:
        """首次派发返回 True；重复派发返回 False（不产生第二次副作用）"""
        with sqlite3.connect(self.path) as con:
            try:
                con.execute("INSERT INTO dispatched(request_id, at) "
                            "VALUES(?, strftime('%s','now'))", (request_id,))
                return True
            except sqlite3.IntegrityError:
                return False

    def attempts(self, request_id: str) -> int:
        with sqlite3.connect(self.path) as con:
            return con.execute("SELECT COUNT(*) FROM attempts WHERE request_id=?",
                               (request_id,)).fetchone()[0]

    def dispatched(self, request_id: str) -> int:
        with sqlite3.connect(self.path) as con:
            return con.execute("SELECT COUNT(*) FROM dispatched WHERE request_id=?",
                               (request_id,)).fetchone()[0]


class Cycle(TypedDict, total=False):
    user_query: str
    request_id: str
    task_queue: List[str]
    remote_result: str
    dispatch_count: int
    final_report: str


# ===========================================================================
# 图定义
# ===========================================================================
def node_plan(state: Cycle) -> Annotated[dict, "planner"]:
    return {"task_queue": ["核查劳动关系", "核算赔偿金额"]}


def node_dispatch(state: Cycle) -> dict:
    """
    把任务交给远端（这里是「内存总线」），然后挂起等结果。

    ⚠️ 关键契约（spike 已复现）：`interrupt()` **之前**的代码在恢复时**会重跑**。
    所以「派发」必须靠 **state 之外**的幂等台账，不能靠 state ——
    LangGraph 不提交未完成节点的状态增量，恢复时 state 里根本没有上次的痕迹。
    """
    ledger = TaskLedger()
    ledger.note_attempt(state["request_id"])
    if ledger.dispatch_once(state["request_id"]):
        Bus.publish(state["request_id"], {"task_queue": state["task_queue"]})

    # ---- 挂起：等远端把结果送回来 ----
    resume_value = interrupt({"request_id": state["request_id"],
                              "task_queue": state["task_queue"]})

    return {"remote_result": resume_value}


def node_finish(state: Cycle) -> dict:
    return {"final_report": f"依据远端结果出具意见：{state.get('remote_result')}"}


def build_graph(saver):
    builder = StateGraph(Cycle)
    builder.add_node("Planner", node_plan)
    builder.add_node("Dispatch", node_dispatch)
    builder.add_node("Finish", node_finish)
    builder.add_edge(START, "Planner")
    builder.add_edge("Planner", "Dispatch")
    builder.add_edge("Dispatch", "Finish")
    builder.add_edge("Finish", END)
    return builder.compile(checkpointer=saver)


def make_saver(db_path: str):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return SqliteSaver(conn)


# ===========================================================================
# 「内存任务总线」—— 代替 Kafka 的传输层（可替换）
# ===========================================================================
class Bus:
    """把"下发"与"结果回收"落到 JSONL 文件，模拟跨进程的消息往返"""

    @classmethod
    def path(cls) -> str:
        return state_path(BUS_NAME)

    @classmethod
    def reset(cls) -> None:
        if os.path.exists(cls.path()):
            os.remove(cls.path())

    @classmethod
    def publish(cls, request_id: str, payload: Dict[str, Any]) -> None:
        with open(cls.path(), "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "task", "request_id": request_id,
                                "payload": payload}, ensure_ascii=False) + "\n")

    @classmethod
    def publish_result(cls, request_id: str, result: str) -> None:
        with open(cls.path(), "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "result", "request_id": request_id,
                                "result": result}, ensure_ascii=False) + "\n")

    @classmethod
    def tasks(cls) -> List[Dict[str, Any]]:
        if not os.path.exists(cls.path()):
            return []
        out = []
        with open(cls.path(), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("kind") == "task":
                    out.append(rec)
        return out


# ===========================================================================
# 子进程用：单阶段执行
# ===========================================================================
def phase_suspend(db_path: str, thread_id: str, request_id: str) -> int:
    """阶段一：跑到 interrupt 挂起，然后**正常退出进程**"""
    saver = make_saver(db_path)
    graph = build_graph(saver)
    config = {"configurable": {"thread_id": thread_id}}

    out = graph.invoke({"user_query": "我被违法辞退了，能要多少钱？",
                        "request_id": request_id},
                       config)

    interrupted = "__interrupt__" in out
    print(json.dumps({"phase": "suspend",
                      "interrupted": interrupted,
                      "task_queue": out.get("task_queue"),
                      "has_final_report": bool(out.get("final_report")),
                      "__interrupt__": _ser(out.get("__interrupt__"))},
                     ensure_ascii=False))
    return 0 if interrupted else 3


def phase_resume(db_path: str, thread_id: str, request_id: str, result: str) -> int:
    """阶段二：**另起一个进程**，重新打开同一份 sqlite，从中断点恢复"""
    saver = make_saver(db_path)
    graph = build_graph(saver)
    config = {"configurable": {"thread_id": thread_id}}

    out = graph.invoke(Command(resume=result), config)
    print(json.dumps({"phase": "resume",
                      "final_report": out.get("final_report"),
                      "still_interrupted": "__interrupt__" in out},
                     ensure_ascii=False))
    return 0


def _ser(obj):
    if obj is None:
        return None
    try:
        return [{"value": getattr(i, "value", None)} for i in obj]
    except TypeError:
        return str(obj)


# ===========================================================================
# 驱动：跑全部 4 项验证
# ===========================================================================
class Check:
    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def add(self, vid: str, title: str, ok: bool, detail: str) -> None:
        self.results.append({"id": vid, "title": title, "ok": ok, "detail": detail})

    def report(self) -> int:
        print()
        print("=" * 78)
        print("P0 Spike 结果")
        print("=" * 78)
        for r in self.results:
            print(f"  [{'OK  ' if r['ok'] else 'FAIL'}] {r['id']}  {r['title']}")
            print(f"         {r['detail']}")
        failed = [r for r in self.results if not r["ok"]]
        print()
        if failed:
            print(f"结论：❌ {len(failed)}/{len(self.results)} 项未通过 —— "
                  f"**不要进入 P1**，先改方案")
            print("     退化方案：soul 进程常驻 + 阻塞拉取结果 topic（不依赖 checkpointer）")
            return 1
        print(f"结论：✅ {len(self.results)}/{len(self.results)} 项全部通过 —— "
              f"**可以进入 P1**（LangGraph Checkpointer 路线成立）")
        return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["suspend", "resume"], default=None)
    ap.add_argument("--db", default=None)
    ap.add_argument("--thread", default="spike-thread-0001")
    ap.add_argument("--request-id", default="req-0001")
    ap.add_argument("--result", default="赔偿金 48000 元")
    args = ap.parse_args(argv)

    # ---- 子进程入口 ----
    if args.phase == "suspend":
        return phase_suspend(args.db, args.thread, args.request_id)
    if args.phase == "resume":
        return phase_resume(args.db, args.thread, args.request_id, args.result)

    # ---- 驱动 ----
    check = Check()
    tmpdir = tempfile.mkdtemp(prefix="spike_p0_")
    # 把临时目录告诉**子进程**（它们靠环境变量找台账和总线）
    os.environ["SPIKE_P0_STATE_DIR"] = tmpdir
    db_path = os.path.join(tmpdir, "checkpoints.sqlite")
    Bus.reset()

    print(f"临时目录 {tmpdir}")
    print(f"代码库 {os.path.dirname(_HERE)}")
    print()

    # ---------------- V1 + V2：挂起并落盘（子进程） ----------------
    print("[V1/V2] 子进程 A：跑到 interrupt 挂起，然后退出进程…")
    p1 = subprocess.run([sys.executable, os.path.abspath(__file__),
                         "--phase", "suspend", "--db", db_path,
                         "--thread", args.thread, "--request-id", args.request_id],
                        capture_output=True, text=True, encoding="utf-8")
    info1 = {}
    for line in (p1.stdout or "").splitlines():
        if line.startswith("{"):
            info1 = json.loads(line)
    check.add("V2", "interrupt() 真的挂起（且进程正常退出 = 未阻塞等待）",
              bool(info1.get("interrupted")),
              f"子进程 exit={p1.returncode}；interrupted={info1.get('interrupted')}；"
              f"挂起时 final_report={info1.get('has_final_report')}（应为 False）"
              + ("" if p1.returncode == 0 else f"；stderr={p1.stderr[-160:]}"))

    check.add("V1", "检查点落盘（sqlite 里有内容，不是纯内存）",
              os.path.exists(db_path) and os.path.getsize(db_path) > 0,
              f"{os.path.basename(db_path)} = "
              f"{(os.path.getsize(db_path) if os.path.exists(db_path) else 0)} B")

    tasks = Bus.tasks()
    ledger = TaskLedger()
    check.add("V1b", "任务已下发给「总线」（模拟 Kafka 生产）",
              len(tasks) == 1,
              f"总线上任务数 = {len(tasks)}；attempts={ledger.attempts(args.request_id)}；"
              f"dispatched={ledger.dispatched(args.request_id)}")

    # ---------------- V4：恢复前重复投递，验证幂等 ----------------
    print("[V4] 恢复前，模拟同一条任务重复投递 3 次…")
    for _ in range(3):
        subprocess.run([sys.executable, os.path.abspath(__file__),
                        "--phase", "suspend", "--db", db_path,
                        "--thread", args.thread, "--request-id", args.request_id],
                       capture_output=True, text=True, encoding="utf-8")
    ledger = TaskLedger()
    attempts, dispatched, tasks_now = (ledger.attempts(args.request_id),
                                       ledger.dispatched(args.request_id),
                                       len(Bus.tasks()))
    check.add("V4", "节点确实被重复执行，但副作用只发生一次（幂等）",
              attempts >= 4 and dispatched == 1 and tasks_now == 1,
              f"节点执行 {attempts} 次（含重复投递）、实际派发 {dispatched} 次、"
              f"总线上任务 {tasks_now} 条 —— 期望「执行多次 / 派发一次」")

    # ---------------- V3：另起进程恢复 ----------------
    print("[V3] 子进程 B：另起进程，重新打开同一 sqlite，从中断点恢复…")
    Bus.publish_result(args.request_id, args.result)
    p2 = subprocess.run([sys.executable, os.path.abspath(__file__),
                         "--phase", "resume", "--db", db_path,
                         "--thread", args.thread, "--request-id", args.request_id,
                         "--result", args.result],
                        capture_output=True, text=True, encoding="utf-8")
    info2 = {}
    for line in (p2.stdout or "").splitlines():
        if line.startswith("{"):
            info2 = json.loads(line)

    finished = bool(info2.get("final_report")) and not info2.get("still_interrupted")
    check.add("V3", "跨进程恢复：新进程从中断点继续并跑到 END",
              p2.returncode == 0 and finished,
              f"exit={p2.returncode}；final_report={info2.get('final_report')!r}；"
              f"still_interrupted={info2.get('still_interrupted')}"
              + ("" if p2.returncode == 0 else f"；stderr={p2.stderr[-160:]}"))

    check.add("V3b", "恢复时节点重跑，但幂等仍生效（未重复派发）",
              ledger.dispatched(args.request_id) == 1 and len(Bus.tasks()) == 1,
              f"节点累计执行 {ledger.attempts(args.request_id)} 次、"
              f"实际派发 {ledger.dispatched(args.request_id)} 次、"
              f"总线任务 {len(Bus.tasks())} 条（期望 派发=1 且 总线=1）")

    return check.report()


if __name__ == "__main__":
    raise SystemExit(main())
