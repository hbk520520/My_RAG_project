"""Small-team task API. Token authentication is mandatory when serving HTTP."""
from contextlib import asynccontextmanager
import secrets
import threading

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from config_loader import cfg
from .contracts import OutcomeUnknown, RunConflict, RunRequest


class ResumeInput(BaseModel):
    answer: str = Field(min_length=1, max_length=20000)


class RewindInput(BaseModel):
    checkpoint_id: str = Field(min_length=1)


def create_app(service=None, *, token=None, execute_local=True):
    from .service import AgentService
    secret = token or cfg.get("runtime", "api_token")
    if not secret:
        raise ValueError("Set AGENT_API_TOKEN before starting the task API")
    service = service or AgentService()
    stop = threading.Event()

    def work():
        try:
            while not stop.wait(0.5):
                service.recover()
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Local task worker stopped")

    @asynccontextmanager
    async def lifespan(app):
        thread = threading.Thread(target=work, name="agent-local-worker", daemon=True) if execute_local else None
        if thread:
            thread.start()
        app.state.worker = thread
        yield
        stop.set()
        if thread:
            thread.join(timeout=1)

    app = FastAPI(title="Legal Agent tasks", lifespan=lifespan)

    def authorize(authorization: str = Header(default="")):
        if not secrets.compare_digest(authorization, "Bearer " + secret):
            raise HTTPException(401, "Invalid bearer token")

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=404, content={"detail": "Run not found"})

    @app.exception_handler(RunConflict)
    async def conflict(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(OutcomeUnknown)
    async def ambiguous(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/ready")
    def ready():
        try:
            worker = getattr(app.state, "worker", None)
            if worker is not None and not worker.is_alive():
                raise RuntimeError("Local worker is not running")
            with service.store.connection() as db:
                db.execute("SELECT 1").fetchone()
            return {"ready": True}
        except Exception as exc:
            raise HTTPException(503, "Run storage unavailable") from exc

    secured = [Depends(authorize)]

    @app.post("/runs", dependencies=secured, status_code=202)
    def submit(request: RunRequest):
        row = service.submit(request, transport="local" if execute_local else "kafka")
        return {"run_id": row["run_id"], "status": row["status"]}

    @app.get("/runs", dependencies=secured)
    def runs(project_id: str = "legal-rag", limit: int = 50):
        return service.store.list_runs(project_id, min(max(limit, 1), 100))

    @app.get("/runs/{run_id}", dependencies=secured)
    def get(run_id: str):
        row = service.store.get(run_id)
        return {key: value for key, value in row.items() if key not in ("lease_token", "resume")}

    @app.get("/runs/{run_id}/events", dependencies=secured)
    def events(run_id: str, after: int = 0):
        service.store.get(run_id)
        return service.store.events(run_id, after=max(after, 0))

    @app.post("/runs/{run_id}/cancel", dependencies=secured)
    def cancel(run_id: str):
        row = service.store.cancel(run_id)
        return {"status": row["status"]}

    @app.post("/runs/{run_id}/resume", dependencies=secured)
    def resume(run_id: str, body: ResumeInput):
        row = service.resume(run_id, body.answer, transport="local" if execute_local else "kafka")
        return {"status": row["status"]}

    @app.get("/runs/{run_id}/checkpoints", dependencies=secured)
    def checkpoints(run_id: str):
        return service.checkpoints(run_id)

    @app.post("/runs/{run_id}/rewind", dependencies=secured)
    def rewind(run_id: str, body: RewindInput):
        row = service.rewind(run_id, body.checkpoint_id, transport="local" if execute_local else "kafka")
        return {"status": row["status"], "epoch": row["epoch"]}

    return app
