"""Configuration and interchange contracts, independent of any training consumer."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

VERSION = "synthetic-tasks-v2"
Family = Literal["lookup", "composition", "dependency", "parallel", "recovery", "constraints"]
Operation = Literal["read", "combine", "handoff", "write", "request", "finish", "distractor"]
Split = Literal["train", "validation", "test", "ood"]
Value = int | str | dict[str, int | str] | None


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Config(Contract):
    seed: int = Field(default=42, ge=0, le=2**32-1)
    groups: int = Field(default=40, ge=1, le=100000)
    ood_groups: int = Field(default=8, ge=0, le=10000)
    max_variants: int = Field(default=4, ge=1, le=16)
    max_items: int = Field(default=12, ge=2, le=24)
    exposure_exponent: float = Field(default=1.4, gt=0, le=5)
    complexity_exponent: float = Field(default=2, gt=0, le=5)
    family_weights: dict[Family, float] = Field(default_factory=lambda: {
        "lookup": 1, "composition": 1, "dependency": 1, "parallel": 1, "recovery": 1, "constraints": 1})
    repetition_rate: float = Field(default=0, ge=0, le=1)
    conflict_rate: float = Field(default=0, ge=0, le=1)
    unresolved_conflict_rate: float = Field(default=0.5, ge=0, le=1)
    train_fraction: float = Field(default=0.8, gt=0, lt=1)
    validation_fraction: float = Field(default=0.1, gt=0, lt=1)
    pending_per_group: int = Field(default=1, ge=0, le=3)
    candidate_call_limit: int = Field(default=0, ge=0, le=1000)
    include_trajectories: bool = False

    @model_validator(mode="after")
    def distributions(self):
        total = sum(self.family_weights.values())
        if not math.isfinite(total) or total <= 0 or any(w < 0 for w in self.family_weights.values()):
            raise ValueError("Family weights need a positive finite total and nonnegative entries")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("Leave a positive test fraction")
        return self


class Evidence(Contract):
    id: str
    facts: dict[str, int | str | None]
    authority: Literal["primary", "reference"] = "primary"
    active: bool = True
    statements: list[str]


class Record(Contract):
    id: str
    sources: list[Evidence]
    requires: list[str] = Field(default_factory=list)
    next_id: str | None = None


class Outcome(Contract):
    status: Literal["completed", "needs_input", "needs_evidence", "blocked"]
    value: Value = None
    reason: str = ""
    evidence_ids: list[str] = Field(default_factory=list)


class Task(Contract):
    id: str
    group_id: str
    split: Split
    family: Family
    query: str
    field: Literal["units", "status"]
    operation: Literal["copy", "sum", "collect"]
    records: list[Record]
    tool_names: dict[Operation, str]
    response_format: Literal["sources", "rows"] = "sources"
    transient_record: str | None = None
    max_calls: int
    write_required: bool = False
    write_allowed: bool = True
    transforms: list[str]

    def public_input(self):
        descriptions = {
            "read": "读取 record_id；返回来源、状态及 next_id；暂时失败可重试一次。",
            "combine": "使用当前角色已读或已交接的数据执行任务指定的 sum/collect；无需参数。",
            "handoff": "把本角色已读的 record_id 交接给 coordinator。",
            "write": "把已计算的 value 写入本任务的内存结果区；必须有写权限。",
            "request": "读完所需记录后请求补充，参数 reason=missing_input 或 source_conflict。",
            "finish": "提交 outcome（status/value/reason/evidence_ids）；不计工具预算。",
            "distractor": "读取无关的天气信息。"}
        roots = [r.id for r in self.records if not r.requires]
        return {"query": self.query, "roots": roots, "field": self.field, "operation": self.operation,
                "outcome_schema": Outcome.model_json_schema(),
                "tools": {name: descriptions[op] for op, name in self.tool_names.items()},
                "response_format": self.response_format, "max_calls": self.max_calls,
                "write_required": self.write_required, "write_allowed": self.write_allowed,
                "assignments": {r.id: "researcher_a" if i % 2 == 0 else "researcher_b" for i, r in enumerate(self.records)}
                    if self.family == "parallel" else {},
                "roles": ["researcher_a", "researcher_b", "coordinator"] if self.family == "parallel" else ["coordinator"]}


class Step(Contract):
    actor: str = "coordinator"
    tool: str
    arguments: dict
    observation: dict


class Trajectory(Contract):
    name: str
    origin: Literal["scripted_reference"] = "scripted_reference"
    steps: list[Step]
    answer: Outcome


class Source(Contract):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class SeedGroup(Contract):
    id: str = Field(min_length=1)
    sources: list[Source] = Field(min_length=1)


class Draft(Contract):
    query: str = Field(min_length=1)
    proposed_answer: str | None = None
    source_ids: list[str] = Field(default_factory=list)
