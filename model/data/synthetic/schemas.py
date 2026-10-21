"""Bounded configuration and serializable contracts for the synthetic environment."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Family = Literal["lookup", "aggregate", "recovery", "missing"]
Split = Literal["train", "validation", "test", "ood"]
VERSION = "synthetic-evidence-v1"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class GenerationConfig(Contract):
    seed: int = Field(default=42, ge=0, le=2**32-1)
    groups: int = Field(default=100, ge=1, le=100000)
    variants: int = Field(default=2, ge=1, le=8)
    ood_groups: int = Field(default=10, ge=0, le=10000)
    max_records: int = Field(default=12, ge=2, le=24)
    complexity_exponent: float = Field(default=2.0, gt=0, le=5)
    topic_count: int = Field(default=16, ge=1, le=64)
    topic_exponent: float = Field(default=1.2, gt=0, le=5)
    family_weights: dict[Family, float] = Field(default_factory=lambda: {
        "lookup": 0.25, "aggregate": 0.35, "recovery": 0.20, "missing": 0.20})
    repetition_rate: float = Field(default=0, ge=0, le=1)
    conflict_rate: float = Field(default=0, ge=0, le=1)
    unresolved_conflict_rate: float = Field(default=0.5, ge=0, le=1)
    train_fraction: float = Field(default=0.8, gt=0, lt=1)
    validation_fraction: float = Field(default=0.1, gt=0, lt=1)

    @model_validator(mode="after")
    def valid_distribution(self):
        import math
        if (not self.family_weights or any(not math.isfinite(w) or w < 0 for w in self.family_weights.values())
                or not math.isfinite(sum(self.family_weights.values())) or sum(self.family_weights.values()) <= 0):
            raise ValueError("Family weights must be finite, nonnegative and have positive total")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("Leave a positive test fraction")
        return self


class Document(Contract):
    id: str
    record_id: str
    amount: int | None = Field(default=None, ge=0, le=100000000)
    authority: Literal["primary", "reference"] = "primary"
    active: bool = True
    statements: list[str] = Field(min_length=1, max_length=2)


class Answer(Contract):
    status: Literal["sufficient", "partial"]
    amount: str | None = None
    evidence_ids: list[str]
    missing_records: list[str] = Field(default_factory=list)
    missing_info_kind: Literal["evidence", "user"] = "evidence"


class Sample(Contract):
    version: Literal["synthetic-evidence-v1"] = VERSION
    synthetic: Literal[True] = True
    sample_id: str
    group_id: str
    split: Split
    family: Family
    topic_rank: int = Field(ge=1)
    query: str
    record_ids: list[str]
    documents: list[Document]
    tools: dict[str, str]
    transient_record: str | None = None
    transforms: list[str]
    expected: Answer


class ToolStep(Contract):
    tool: str
    arguments: dict
    observation: dict


class Trajectory(Contract):
    origin: Literal["scripted_reference"] = "scripted_reference"
    name: str
    steps: list[ToolStep]
    answer: Answer
