"""SQLite run authority, leases, events, effect ledger and transactional outbox.

Graph checkpoints have their own commit boundary. The ledger makes replay explicit;
it does not claim a transaction across a model provider, Docker and SQLite.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3
import time
import uuid

from .contracts import (BudgetExceeded, OutcomeUnknown, RunConflict, RunRequest,
                        RunStatus, RunStopped, local_path)


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


class RunStore:
    def __init__(self, path=".runtime/runs.sqlite"):
        self.path = local_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                  run_id TEXT PRIMARY KEY, project_id TEXT NOT NULL,
                  status TEXT NOT NULL, epoch INTEGER NOT NULL DEFAULT 0,
                  request TEXT NOT NULL, result TEXT, error TEXT, error_kind TEXT,
                  created REAL NOT NULL, updated REAL NOT NULL, deadline REAL NOT NULL,
                  transport TEXT NOT NULL DEFAULT 'local',
                  lease_token TEXT, lease_until REAL, resume TEXT, checkpoint TEXT, usage TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events (
                  seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                  epoch INTEGER NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL, at REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS events_run ON events(run_id, seq);
                CREATE TABLE IF NOT EXISTS effects (
                  call_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, epoch INTEGER NOT NULL,
                  tool TEXT NOT NULL, input_hash TEXT NOT NULL, read_only INTEGER NOT NULL,
                  status TEXT NOT NULL, result TEXT, attempts INTEGER NOT NULL DEFAULT 1);
                CREATE TABLE IF NOT EXISTS outbox (
                  event_id TEXT PRIMARY KEY, topic TEXT NOT NULL, payload TEXT NOT NULL,
                  sent INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS pins (
                  run_id TEXT NOT NULL, epoch INTEGER NOT NULL, data TEXT NOT NULL,
                  PRIMARY KEY(run_id, epoch));
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA synchronous=FULL")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _event(self, db, row, kind, payload):
        db.execute("INSERT INTO events(run_id,epoch,kind,payload,at) VALUES(?,?,?,?,?)",
                   (row["run_id"], row["epoch"], kind, dumps(payload), time.time()))

    def _enqueue(self, db, row, topic):
        event_id = uuid.uuid4().hex
        payload = {"version": 1, "event_id": event_id, "run_id": row["run_id"], "epoch": row["epoch"]}
        db.execute("INSERT INTO outbox VALUES(?,?,?,0)", (event_id, topic, dumps(payload)))

    def create(self, request: RunRequest, *, queued_topic=None):
        now = time.time()
        run_id = uuid.uuid4().hex
        with self.connection() as db:
            db.execute("INSERT INTO runs(run_id,project_id,status,request,created,updated,deadline,usage,transport) VALUES(?,?,?,?,?,?,?,?,?)",
                       (run_id, request.project_id, RunStatus.QUEUED, request.model_dump_json(), now, now,
                        now + request.budgets.seconds, dumps({"model_calls": 0, "text_tokens": 0, "node_steps": 0}),
                        "kafka" if queued_topic else "local"))
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            self._event(db, row, "submitted", {})
            if queued_topic:
                self._enqueue(db, row, queued_topic)
        return self.get(run_id)

    def get(self, run_id):
        with self.connection() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            result = dict(row)
            for field in ("request", "result", "resume", "usage"):
                result[field] = json.loads(result[field]) if result[field] is not None else None
            return result

    def list_runs(self, project_id, limit=50):
        with self.connection() as db:
            return [dict(r) for r in db.execute(
                "SELECT run_id,status,epoch,created,updated,error_kind FROM runs WHERE project_id=? ORDER BY created DESC LIMIT ?",
                (project_id, limit))]

    def claim(self, run_id, *, epoch=None, lease_seconds=30, result_topic="topic.agent.result"):
        now = time.time()
        token = uuid.uuid4().hex
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            if epoch is not None and epoch != row["epoch"]:
                return None
            if row["status"] not in (RunStatus.QUEUED, RunStatus.RUNNING):
                return None
            if row["status"] == RunStatus.RUNNING and (row["lease_until"] or 0) > now:
                return None
            if db.execute("SELECT 1 FROM runs WHERE run_id<>? AND status=? AND lease_until>? LIMIT 1",
                          (run_id, RunStatus.RUNNING, now)).fetchone():
                return None
            if row["deadline"] <= now:
                db.execute("UPDATE runs SET status=?,error_kind='deadline',updated=? WHERE run_id=?",
                           (RunStatus.TIMED_OUT, now, run_id))
                self._event(db, row, "timed_out", {})
                if row["transport"] == "kafka":
                    self._enqueue(db, row, result_topic)
                return None
            db.execute("UPDATE runs SET status=?,lease_token=?,lease_until=?,updated=? WHERE run_id=?",
                       (RunStatus.RUNNING, token, now + lease_seconds, now, run_id))
            self._event(db, row, "claimed", {})
        return token

    def _active(self, db, run_id, epoch, token):
        row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None or row["epoch"] != epoch or row["status"] != RunStatus.RUNNING or row["lease_token"] != token:
            raise RunStopped("Run is no longer owned by this executor")
        if row["deadline"] <= time.time():
            raise BudgetExceeded("Run deadline exceeded")
        return row

    def heartbeat(self, run_id, epoch, token, lease_seconds=30):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._active(db, run_id, epoch, token)
            db.execute("UPDATE runs SET lease_until=? WHERE run_id=?", (time.time() + lease_seconds, run_id))

    def consume(self, run_id, epoch, token, **increments):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._active(db, run_id, epoch, token)
            usage = json.loads(row["usage"])
            budget = json.loads(row["request"])["budgets"]
            for name, amount in increments.items():
                usage[name] += amount
                if usage[name] > budget[name]:
                    raise BudgetExceeded(f"{name} budget exceeded")
            db.execute("UPDATE runs SET usage=? WHERE run_id=?", (dumps(usage), run_id))

    def event(self, run_id, epoch, token, kind, payload):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = self._active(db, run_id, epoch, token)
            self._event(db, row, kind, payload)

    def events(self, run_id, after=0, limit=200):
        with self.connection() as db:
            return [{**dict(r), "payload": json.loads(r["payload"])} for r in db.execute(
                "SELECT * FROM events WHERE run_id=? AND seq>? ORDER BY seq LIMIT ?", (run_id, after, limit))]

    def finish(self, run_id, epoch, token, status, *, result=None, error=None, error_kind=None, result_topic=None):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row["epoch"] != epoch or row["status"] != RunStatus.RUNNING or row["lease_token"] != token:
                return False
            db.execute("UPDATE runs SET status=?,result=?,error=?,error_kind=?,lease_token=NULL,lease_until=NULL,updated=? WHERE run_id=?",
                       (status, dumps(result) if result is not None else None, error, error_kind, time.time(), run_id))
            self._event(db, row, str(status), {"error_kind": error_kind})
            if result_topic:
                self._enqueue(db, row, result_topic)
            return True

    def cancel(self, run_id):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            if row["status"] in (RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.WAITING):
                db.execute("UPDATE runs SET status=?,lease_token=NULL,updated=? WHERE run_id=?",
                           (RunStatus.CANCELLED, time.time(), run_id))
                self._event(db, row, "cancelled", {})
        return self.get(run_id)

    def requeue(self, run_id, *, answer=None, topic=None, retry=False):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            expected = RunStatus.FAILED if retry else RunStatus.WAITING
            if row["status"] != expected:
                raise RunConflict(f"Expected {expected}, got {row['status']}")
            if row["deadline"] <= time.time():
                raise RunConflict("Original run deadline has expired; submit a new run")
            db.execute("UPDATE runs SET status=?,resume=?,error=NULL,error_kind=NULL,updated=? WHERE run_id=?",
                       (RunStatus.QUEUED, dumps(answer) if answer is not None else row["resume"], time.time(), run_id))
            self._event(db, row, "retry" if retry else "resumed", {})
            if topic:
                self._enqueue(db, row, topic)
        return self.get(run_id)

    def begin_effect(self, run_id, epoch, token, call_id, tool, input_hash, read_only):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._active(db, run_id, epoch, token)
            row = db.execute("SELECT * FROM effects WHERE call_id=?", (call_id,)).fetchone()
            if row:
                if row["input_hash"] != input_hash or row["tool"] != tool:
                    raise RunConflict("Call identity reused with different input")
                if row["status"] == "done":
                    return True, json.loads(row["result"])
                if not row["read_only"]:
                    raise OutcomeUnknown("Interrupted write tool needs reconciliation before retry")
                db.execute("UPDATE effects SET attempts=attempts+1,status='started' WHERE call_id=?", (call_id,))
            else:
                db.execute("INSERT INTO effects(call_id,run_id,epoch,tool,input_hash,read_only,status) VALUES(?,?,?,?,?,?,'started')",
                           (call_id, run_id, epoch, tool, input_hash, int(read_only)))
        return False, None

    def end_effect(self, run_id, epoch, token, call_id, result, *, status="done"):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._active(db, run_id, epoch, token)
            cursor = db.execute("UPDATE effects SET status=?,result=? WHERE call_id=?", (status, dumps(result), call_id))
            if cursor.rowcount != 1:
                raise RunConflict("Unknown tool call identity")

    def pending_outbox(self, limit=100):
        with self.connection() as db:
            return [{**dict(r), "payload": json.loads(r["payload"])} for r in db.execute(
                "SELECT * FROM outbox WHERE sent=0 ORDER BY rowid LIMIT ?", (limit,))]

    def acknowledge_outbox(self, event_id):
        with self.connection() as db:
            db.execute("UPDATE outbox SET sent=1 WHERE event_id=?", (event_id,))

    def recoverable(self):
        with self.connection() as db:
            return [r[0] for r in db.execute(
                "SELECT run_id FROM runs WHERE status=? OR (status=? AND lease_until<?)",
                (RunStatus.QUEUED, RunStatus.RUNNING, time.time()))]

    def expire_waiting(self, result_topic="topic.agent.result"):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            for row in db.execute("SELECT * FROM runs WHERE status=? AND deadline<?",
                                  (RunStatus.WAITING, time.time())).fetchall():
                db.execute("UPDATE runs SET status=?,error_kind='deadline',updated=? WHERE run_id=?",
                           (RunStatus.TIMED_OUT, time.time(), row["run_id"]))
                self._event(db, row, "timed_out", {})
                if row["transport"] == "kafka":
                    self._enqueue(db, row, result_topic)

    def pin(self, run_id, epoch, data=None):
        with self.connection() as db:
            if data is not None:
                db.execute("INSERT INTO pins VALUES(?,?,?)", (run_id, epoch, dumps(data)))
            row = db.execute("SELECT data FROM pins WHERE run_id=? AND epoch=?", (run_id, epoch)).fetchone()
            return json.loads(row[0]) if row else None

    def set_checkpoint(self, run_id, epoch, token, checkpoint):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            self._active(db, run_id, epoch, token)
            db.execute("UPDATE runs SET checkpoint=? WHERE run_id=?", (checkpoint, run_id))

    def rewind(self, run_id, checkpoint, *, topic=None):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(run_id)
            if row["status"] in (RunStatus.RUNNING, RunStatus.QUEUED):
                raise RunConflict("Cancel the active run before rewinding")
            if row["deadline"] <= time.time():
                raise RunConflict("Original deadline expired; submit a new run")
            if db.execute("SELECT 1 FROM effects WHERE run_id=? AND read_only=0 LIMIT 1", (run_id,)).fetchone():
                raise OutcomeUnknown("Write effects need explicit reconciliation before rewind")
            db.execute("UPDATE runs SET epoch=epoch+1,status=?,checkpoint=?,resume=NULL,result=NULL,error=NULL,error_kind=NULL,lease_token=NULL,updated=? WHERE run_id=?",
                       (RunStatus.QUEUED, checkpoint, time.time(), run_id))
            old_pin = db.execute("SELECT data FROM pins WHERE run_id=? AND epoch=?", (run_id, row["epoch"])).fetchone()
            if old_pin:
                db.execute("INSERT INTO pins VALUES(?,?,?)", (run_id, row["epoch"] + 1, old_pin[0]))
            new = dict(row)
            new["epoch"] += 1
            self._event(db, new, "rewound", {"checkpoint": checkpoint})
            if topic:
                self._enqueue(db, new, topic)
        return self.get(run_id)

    def backup(self, destination):
        destination = local_path(destination)
        if destination == self.path or destination.exists():
            raise ValueError("Backup destination must be a new file")
        destination.parent.mkdir(parents=True, exist_ok=True)
        target = sqlite3.connect(destination)
        try:
            with self.connection() as source:
                source.backup(target)
            if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("Backup integrity check failed")
        finally:
            target.close()
        return destination
