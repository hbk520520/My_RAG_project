"""Data contracts: support scores describe text, never statistical causal effects."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SourceLocation(Contract):
    document_id: str | None = Field(default=None, min_length=1)
    document_version: str | None = Field(default=None, min_length=1)
    source_file: str | None = Field(default=None, min_length=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def page_interval(self):
        if self.page_start is not None and self.page_end is not None and self.page_end < self.page_start:
            raise ValueError("Source page interval is reversed")
        return self


class EvidenceRecord(SourceLocation):
    record_id: str = Field(min_length=1)
    source_id: str | None = Field(default=None, min_length=1)
    text: str = Field(min_length=1)
    timestamp: date | None = None
    available_at: date | None = None
    availability_verified: bool = False
    similarity: float | None = Field(default=None, ge=0, le=1)


class TextChunk(SourceLocation):
    chunk_id: str
    record_id: str
    source_id: str | None = None
    text: str
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    timestamp: date | None = None
    available_at: date | None = None
    availability_verified: bool = False
    similarity: float | None = Field(default=None, ge=0, le=1)


class Entity(Contract):
    name: str = Field(min_length=1)


class Proposition(Contract):
    sentence: str = Field(min_length=1)
    entities: list[Entity] = Field(min_length=1)
    evidence_quote: str = Field(min_length=1)


class CausalRelation(Contract):
    cause: str = Field(min_length=1)
    effect: str = Field(min_length=1)
    type: Literal["causes", "enables", "prevents"]
    strength: float = Field(gt=0, le=1)
    proposition_index: int = Field(ge=0)
    evidence_quote: str = Field(min_length=1)


class Extraction(Contract):
    propositions: list[Proposition]
    causal_relations: list[CausalRelation]


class Verdict(Contract):
    candidate_index: int = Field(ge=0)
    classification: Literal["causal", "correlation", "negated", "hypothetical", "unsupported"]
    supported: bool
    evidence_quote: str
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def consistent(self):
        if self.supported and self.classification != "causal":
            raise ValueError("Only an affirmative causal claim can be supported")
        return self


class Verification(Contract):
    verdicts: list[Verdict]


class Hyperedge(Contract):
    id: str
    chunk_id: str
    record_id: str
    source_id: str | None = None
    proposition: str
    entities: list[str]
    evidence_quote: str
    timestamp: date | None = None
    available_at: date | None = None
    availability_verified: bool = False
    similarity: float | None = Field(default=None, ge=0, le=1)


class CausalEdge(Contract):
    id: str
    cause: str
    effect: str
    type: Literal["causes", "enables", "prevents"]
    raw_support: float = Field(gt=0, le=1)
    calibrated_support: float | None = Field(default=None, ge=0, le=1)
    hyperedge_id: str
    evidence_quote: str
    verification_reason: str

    @property
    def support(self) -> float:
        return self.raw_support if self.calibrated_support is None else self.calibrated_support


class Rejection(Contract):
    chunk_id: str
    candidate_index: int | None = None
    proposition_index: int | None = None
    candidate: CausalRelation | None = None
    reason: str


class CausalTemporalGraph(Contract):
    source_chunks: list[TextChunk] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    hyperedges: list[Hyperedge] = Field(default_factory=list)
    causal_edges: list[CausalEdge] = Field(default_factory=list)
    rejections: list[Rejection] = Field(default_factory=list)
    calibration_status: str = "uncalibrated"

    @model_validator(mode="after")
    def graph_references(self):
        hyperedges = {h.id: h for h in self.hyperedges}
        if len(hyperedges) != len(self.hyperedges) or len({c.id for c in self.causal_edges}) != len(self.causal_edges):
            raise ValueError("Duplicate graph identifiers")
        if set(self.entities) != {v for h in self.hyperedges for v in h.entities}:
            raise ValueError("Entity set must equal hyperedge entities")
        for edge in self.causal_edges:
            hyperedge = hyperedges.get(edge.hyperedge_id)
            if hyperedge is None or not {edge.cause, edge.effect}.issubset(hyperedge.entities):
                raise ValueError("Causal endpoints require a supporting hyperedge")
        return self
