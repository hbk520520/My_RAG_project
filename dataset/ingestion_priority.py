"""Explainable intake ordering; never a legal authority or retrieval score."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator


Priority = Literal["P0", "P1", "P2", "P3"]
MaterialRole = Literal["core", "supporting", "reference", "unclassified"]
POLICY_VERSION = "ingestion-priority-v1"


class IngestionMaterial(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    document_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    role: MaterialRole = "unclassified"
    deadline: date | None = None
    expedited: StrictBool = False
    review_required: StrictBool = False
    priority_override: Priority | None = None
    override_reason: str | None = None

    @field_validator("deadline", mode="before")
    @classmethod
    def calendar_deadline(cls, value):
        if value is None or type(value) is date:
            return value
        if not isinstance(value, str):
            raise ValueError("deadline must be a calendar date or YYYY-MM-DD string")
        parsed = date.fromisoformat(value)
        if value != parsed.isoformat():
            raise ValueError("deadline must use YYYY-MM-DD")
        return parsed

    @model_validator(mode="after")
    def require_override_reason(self):
        if self.priority_override is not None and not self.override_reason:
            raise ValueError("A priority override requires a nonempty override_reason")
        if self.priority_override is None and self.override_reason is not None:
            raise ValueError("override_reason requires a priority_override")
        return self


class PriorityAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_version: Literal["ingestion-priority-v1"] = POLICY_VERSION
    assessed_on: date
    urgent_within_days: int
    priority: Priority
    reasons: list[str]
    review_required: bool
    classification_required: bool


def assess_priority(material: IngestionMaterial, *, as_of: date,
                    urgent_within_days: int = 7) -> PriorityAssessment:
    """Use declared intake facts, not filenames, OCR confidence or LLM guesses.

    An explicit date makes deadline decisions reproducible. Overdue material
    remains urgent until its deadline is corrected or explicitly overridden.
    """
    if type(as_of) is not date:
        raise TypeError("as_of must be a calendar date")
    if type(urgent_within_days) is not int or urgent_within_days < 0:
        raise ValueError("urgent_within_days must be a nonnegative integer")
    reasons = ["role:" + material.role]
    urgent = material.expedited
    if material.expedited:
        reasons.append("expedited")
    if material.deadline is not None:
        days = (material.deadline - as_of).days
        if days < 0:
            reasons.append("deadline_overdue")
        elif days <= urgent_within_days:
            reasons.append("deadline_within_window")
        else:
            reasons.append("deadline_outside_window")
        urgent = urgent or days <= urgent_within_days

    if material.priority_override is not None:
        priority = material.priority_override
        reasons.append("manual_override:" + material.override_reason)
    elif urgent:
        priority = "P0"
    else:
        priority = {"core": "P1", "supporting": "P2",
                    "reference": "P3", "unclassified": "P2"}[material.role]
    return PriorityAssessment(
        assessed_on=as_of, urgent_within_days=urgent_within_days,
        priority=priority, reasons=reasons, review_required=material.review_required,
        classification_required=material.role == "unclassified")


def plan_ingestion(materials: list[IngestionMaterial], *, as_of: date,
                   urgent_within_days: int = 7) -> list[dict]:
    """Return all materials, ordered by priority, deadline, then input order.

    No files are opened and no graph, model, worker queue or database is changed.
    Keep this plan separate from source metadata and document fingerprints.
    """
    # Validate policy/date even for an empty manifest.
    if type(as_of) is not date:
        raise TypeError("as_of must be a calendar date")
    if type(urgent_within_days) is not int or urgent_within_days < 0:
        raise ValueError("urgent_within_days must be a nonnegative integer")
    ids = [material.document_id for material in materials]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate document_id in ingestion plan")
    assessed = [(material, assess_priority(material, as_of=as_of,
                                          urgent_within_days=urgent_within_days))
                for material in materials]
    assessed.sort(key=lambda item: (item[1].priority, item[0].deadline or date.max))
    return [{"material": material.model_dump(mode="json"),
             "assessment": assessment.model_dump(mode="json")}
            for material, assessment in assessed]
