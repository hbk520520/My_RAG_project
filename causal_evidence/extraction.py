"""LLM extraction followed by source checks and a separate verification call."""
from __future__ import annotations

import unicodedata

import prompts

from .calibration import SupportCalibrator
from .models import (CausalEdge, Extraction, Hyperedge, Rejection, TextChunk, Verification)


def canonical(text: str) -> str:
    # Conservative normalization only; do not merge synonyms or infer entity identity.
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def extract_verified(chunk: TextChunk, call, calibrator: SupportCalibrator | None = None):
    extracted = Extraction.model_validate_json(call(
        prompts.build_causal_extraction_messages(chunk.text, Extraction.model_json_schema())))
    hyperedges = []
    by_index = {}
    rejections = []
    for index, proposition in enumerate(extracted.propositions):
        names = [entity.name for entity in proposition.entities]
        if (proposition.evidence_quote not in chunk.text
                or proposition.sentence not in proposition.evidence_quote
                or any(name not in proposition.evidence_quote for name in names)):
            rejections.append(Rejection(chunk_id=chunk.chunk_id, proposition_index=index,
                                        reason="ungrounded_proposition_quote"))
            continue
        edge = Hyperedge(
            id=f"{chunk.chunk_id}:p{index}", chunk_id=chunk.chunk_id, record_id=chunk.record_id,
            source_id=chunk.source_id,
            proposition=proposition.sentence, entities=sorted({canonical(n) for n in names}),
            evidence_quote=proposition.evidence_quote, timestamp=chunk.timestamp,
            available_at=chunk.available_at, availability_verified=chunk.availability_verified,
            similarity=chunk.similarity,
        )
        by_index[index] = edge
        hyperedges.append(edge)
    candidates = []
    for index, relation in enumerate(extracted.causal_relations):
        hyperedge = by_index.get(relation.proposition_index)
        names = {canonical(relation.cause), canonical(relation.effect)}
        reason = None
        if hyperedge is None or not names.issubset(hyperedge.entities):
            reason = "endpoints_not_in_grounded_proposition"
        elif len(names) < 2:
            reason = "self_causal_edge"
        elif (relation.evidence_quote not in chunk.text or relation.cause not in relation.evidence_quote
              or relation.effect not in relation.evidence_quote):
            reason = "ungrounded_relation_quote"
        if reason:
            rejections.append(Rejection(chunk_id=chunk.chunk_id, candidate_index=index,
                                        candidate=relation, reason=reason))
        else:
            candidates.append((index, relation))
    if not candidates:
        return hyperedges, [], rejections
    verdicts = Verification.model_validate_json(call(prompts.build_causal_verification_messages(
        chunk.text, [{"candidate_index": i, **r.model_dump()} for i, r in candidates],
        Verification.model_json_schema(),
    ))).verdicts
    expected = {i for i, _ in candidates}
    if len(verdicts) != len(expected) or {v.candidate_index for v in verdicts} != expected:
        raise ValueError("Verifier must return exactly one verdict per candidate")
    by_verdict = {v.candidate_index: v for v in verdicts}
    edges = []
    for index, relation in candidates:
        verdict = by_verdict[index]
        if not verdict.supported:
            reason = verdict.classification + ": " + verdict.rationale
        elif (not verdict.evidence_quote or verdict.evidence_quote not in chunk.text
              or relation.cause not in verdict.evidence_quote or relation.effect not in verdict.evidence_quote):
            reason = "ungrounded_verifier_quote"
        else:
            edges.append(CausalEdge(
                id=f"{chunk.chunk_id}:c{index}", cause=canonical(relation.cause),
                effect=canonical(relation.effect), type=relation.type, raw_support=relation.strength,
                calibrated_support=calibrator.predict(relation.strength) if calibrator else None,
                hyperedge_id=by_index[relation.proposition_index].id,
                evidence_quote=verdict.evidence_quote, verification_reason=verdict.rationale,
            ))
            continue
        rejections.append(Rejection(chunk_id=chunk.chunk_id, candidate_index=index,
                                    candidate=relation, reason=reason))
    return hyperedges, edges, rejections
