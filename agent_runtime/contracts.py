"""Validated boundaries; internal workflow transitions use ordinary Python data."""
from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from config_loader import cfg

ROOT = Path(cfg.config_path).resolve().parent


def local_path(value: str | Path) -> Path:
    path = Path(value)
    path = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if ROOT.drive and path.drive.casefold() != "d:":
        raise ValueError("Runtime files must stay on the project's D drive")
    return path


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting_input"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class Budgets(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seconds: int = Field(default=300, ge=1, le=86400)
    model_calls: int = Field(default=40, ge=1, le=1000)
    text_tokens: int = Field(default=100000, ge=1)
    node_steps: int = Field(default=80, ge=1, le=1000)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=100000)
    project_id: str = Field(default="legal-rag", pattern=r"^[A-Za-z0-9_-]{1,64}$")
    prediction_cutoff: str | None = None
    causal_targets: list[str] | None = None
    budgets: Budgets = Field(default_factory=Budgets)
    allow_code: bool = False

    @field_validator("query")
    @classmethod
    def meaningful_query(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("query is blank")
        return value

    @field_validator("prediction_cutoff", mode="before")
    @classmethod
    def valid_date(cls, value):
        if value is not None:
            from datetime import date
            if isinstance(value, date):
                value = value.isoformat()
            if not isinstance(value, str):
                raise ValueError("prediction_cutoff must be an ISO date")
            date.fromisoformat(value)
        return value


class EvidencePolicy(BaseModel):
    """Composable tools; none of these choices grants permission to invent evidence."""
    model_config = ConfigDict(extra="forbid")
    navigation: Literal["auto", "vector", "agent"] = "auto"
    causal: Literal["auto", "off", "verify", "temporal"] = "auto"
    analysis: Literal["facts", "reasoning"] = "facts"


class TaskSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(min_length=1)
    task_desc: str = Field(min_length=1)
    deps: list[str] = Field(default_factory=list)
    engine: Literal["GRAPH_TRAVERSAL", "GLOBAL_DENSE_WORMHOLE"] = "GRAPH_TRAVERSAL"
    rationale: str = ""
    evidence: EvidencePolicy = Field(default_factory=EvidencePolicy)
    revision: int = Field(default=0, ge=0)


class Grade(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["sufficient", "partial", "irrelevant"]
    rationale: str = ""
    extracted_facts: list[str] = Field(default_factory=list)
    missing_info: str = ""
    missing_info_kind: Literal["evidence", "user"] = "evidence"


class RunConflict(RuntimeError):
    pass


class RunStopped(RuntimeError):
    pass


class BudgetExceeded(RunStopped):
    pass


class OutcomeUnknown(RuntimeError):
    pass
