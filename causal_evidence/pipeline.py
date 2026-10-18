"""One shared retrieval-to-grading adapter for LangGraph and Kafka Workers."""
from __future__ import annotations

from datetime import date
import json
from pathlib import Path

from pydantic import Field

from .calibration import SupportCalibrator
from .extraction import canonical
from .graph import build_graph, score_graph
from .models import Contract, EvidenceRecord


class EvidenceSettings(Contract):
    enabled: bool = False
    calibration_path: str | None = None
    prediction_cutoff: date | None = None
    targets: list[str] | None = None
    d_max: int = Field(default=3, ge=1)
    eta: float = Field(default=5.0, gt=0)
    missing_time_validity: float = Field(default=0.25, ge=0, le=1)
    max_chains: int = Field(default=200, ge=1)
    max_expansions: int = Field(default=10000, ge=1)
    min_score: float = Field(default=0.005, ge=0, le=1)


def evidence_from_hit(hit: dict, *, similarity: float | None = None) -> EvidenceRecord:
    """Only explicit metadata can assert publication availability; never an LLM guess."""
    metadata = hit.get("metadata") or {}
    return EvidenceRecord(
        record_id=str(metadata.get("record_id", hit["id"])), text=hit["content"],
        source_id=metadata.get("source_id"), timestamp=metadata.get("timestamp"),
        available_at=metadata.get("available_at"),
        availability_verified=metadata.get("availability_verified", False),
        similarity=metadata.get("similarity") if similarity is None else similarity,
    )


def causal_completion(messages):
    """Worker LLM adapter; failures propagate, never turn into an empty graph."""
    from openai import OpenAI
    from config_loader import cfg

    client = OpenAI(api_key=cfg.get("llm", "api_key"), base_url=cfg.get("llm", "base_url"))
    response = client.chat.completions.create(
        model=cfg.get("llm", "judge_model"), messages=messages,
        response_format={"type": "json_object"}, temperature=0,
        max_tokens=cfg.get("llm", "max_tokens_default", default=2048),
    )
    return response.choices[0].message.content


def analyze_evidence(records: list[EvidenceRecord], question: str, call, *,
                     settings: EvidenceSettings, extractor_id: str, encoding=None) -> dict:
    calibrator = None
    if settings.calibration_path:
        path = Path(settings.calibration_path)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        calibrator = SupportCalibrator.model_validate_json(path.read_text(encoding="utf-8"))
    graph = build_graph(records, call, cutoff=settings.prediction_cutoff,
                        calibrator=calibrator, extractor_id=extractor_id, encoding=encoding)
    weighting = None
    status = "not_scored_without_cutoff"
    unresolved_targets = []
    if settings.prediction_cutoff:
        # Conservative anchoring only. Explicit targets are preferred to inferred synonyms.
        targets = settings.targets
        if targets is None:
            targets = [entity for entity in graph.entities if entity in canonical(question)]
        unresolved_targets = [t for t in targets if canonical(t) not in graph.entities]
        anchored = [t for t in targets if canonical(t) in graph.entities]
        if anchored:
            weighting = score_graph(graph, anchored, settings.prediction_cutoff,
                                    d_max=settings.d_max, eta=settings.eta,
                                    missing_time_validity=settings.missing_time_validity,
                                    max_chains=settings.max_chains, max_expansions=settings.max_expansions,
                                    min_score=settings.min_score)
            status = "scored"
        else:
            status = "no_target_anchor"
    serialized = graph.model_dump(mode="json")
    context = ""
    if graph.source_chunks:
        context = json.dumps({
            "notice": "Use original evidence for facts; use only verified_causal_relations for causal claims. "
                      "Scores measure textual support and temporal validity, never effect sizes or forecast probabilities.",
            "admitted_evidence": [{"record_id": c.record_id, "chunk_id": c.chunk_id, "text": c.text}
                                  for c in graph.source_chunks],
            "verified_causal_relations": serialized["causal_edges"], "weighting": weighting,
        }, ensure_ascii=False)
    if not graph.source_chunks:
        status = "no_admitted_evidence"
    return {"graph": serialized, "weighting": weighting, "weighting_status": status,
            "unresolved_targets": unresolved_targets, "context": context}


def analyze_configured(records: list[EvidenceRecord], question: str, call=None, *,
                       prediction_cutoff=None, targets=None) -> dict:
    from config_loader import cfg

    values = dict(cfg.get("causal_evidence", default={}))
    if prediction_cutoff is not None:
        values["prediction_cutoff"] = prediction_cutoff
    if targets is not None:
        values["targets"] = targets
    settings = EvidenceSettings.model_validate(values)
    return analyze_evidence(records, question, call or causal_completion, settings=settings,
                            extractor_id=cfg.get("llm", "judge_model") + ":causal-v1")
