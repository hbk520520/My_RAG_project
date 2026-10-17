"""Programmatic hypergraph assembly and cutoff-conditioned CTVF (CHAIN eqs. 5–12)."""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import date
import math

from .chunking import admissible, chunk_record
from .extraction import canonical, extract_verified
from .models import CausalTemporalGraph, EvidenceRecord


def build_graph(records: list[EvidenceRecord], call, *, cutoff: date | None = None,
                calibrator=None, extractor_id: str = "", encoding=None) -> CausalTemporalGraph:
    if calibrator is not None:
        if calibrator.extractor_id != extractor_id:
            raise ValueError("Calibration extractor does not match the active extractor")
        if cutoff is not None and calibrator.trained_through >= cutoff:
            raise ValueError("Calibration labels must predate the prediction cutoff")
    graph = CausalTemporalGraph(calibration_status="isotonic" if calibrator else "uncalibrated")
    seen = {}
    for record in records:
        if record.record_id in seen:
            if seen[record.record_id] != record:
                raise ValueError(f"Conflicting evidence for record_id={record.record_id}")
            continue
        seen[record.record_id] = record
        # Known publication dates block post-cutoff material before any model sees it.
        if cutoff is not None and record.available_at is not None and record.available_at >= cutoff:
            continue
        if cutoff is not None and record.timestamp is not None and record.timestamp >= cutoff:
            continue
        for chunk in chunk_record(record, encoding=encoding):
            if cutoff is not None and not admissible(chunk, cutoff):
                continue
            hyperedges, edges, rejected = extract_verified(chunk, call, calibrator)
            graph.hyperedges.extend(hyperedges)
            graph.causal_edges.extend(edges)
            graph.rejections.extend(rejected)
    graph.entities = sorted({v for h in graph.hyperedges for v in h.entities})
    return graph


def active_graph(graph: CausalTemporalGraph, cutoff: date) -> CausalTemporalGraph:
    hyperedges = [h for h in graph.hyperedges if admissible(h, cutoff)]
    ids = {h.id for h in hyperedges}
    return CausalTemporalGraph(
        entities=sorted({v for h in hyperedges for v in h.entities}), hyperedges=hyperedges,
        causal_edges=[c for c in graph.causal_edges if c.hyperedge_id in ids],
        rejections=graph.rejections, calibration_status=graph.calibration_status,
    )


def reverse_distances(graph: CausalTemporalGraph, targets: list[str]) -> dict[str, int]:
    roots = {canonical(t) for t in targets}
    if not roots or not roots.issubset(graph.entities):
        raise ValueError("Targets must be nonempty, explicit entities in the active graph")
    reverse = defaultdict(set)
    for edge in graph.causal_edges:
        reverse[edge.effect].add(edge.cause)
    distances = {t: 0 for t in roots}
    queue = deque(sorted(roots))
    while queue:
        effect = queue.popleft()
        for cause in sorted(reverse[effect]):
            if cause not in distances:
                distances[cause] = distances[effect] + 1
                queue.append(cause)
    return distances


def score_graph(graph: CausalTemporalGraph, targets: list[str], cutoff: date, *,
                d_max: int = 3, eta: float = 5.0, missing_time_validity: float = 0.25,
                max_chains: int = 200, max_expansions: int = 10000,
                min_score: float = 0.005) -> dict:
    if (d_max < 1 or not math.isfinite(eta) or eta <= 0 or max_chains < 1
            or max_expansions < 1 or not 0 <= min_score <= 1
            or not 0 <= missing_time_validity <= 1):
        raise ValueError("Invalid CTVF/search configuration")
    active = active_graph(graph, cutoff)
    distances = reverse_distances(active, targets)
    recurrence = defaultdict(set)
    for edge in active.hyperedges:
        recurrence[canonical(edge.proposition)].add(edge.record_id)
    hyperedge_scores = {}
    incident = defaultdict(dict)
    for edge in active.hyperedges:
        distance = min((distances.get(v, math.inf) for v in edge.entities), default=math.inf)
        rho = min(distance / d_max, 1)
        count = len(recurrence[canonical(edge.proposition)])
        root_n = math.sqrt(count)
        # Missing similarity contributes no unmeasured salience bonus.
        salience = (1 / (1 + math.exp(-root_n / eta)) - 0.5) * root_n * (edge.similarity or 0)
        if edge.timestamp is None:
            recency = None
            validity = missing_time_validity
        else:
            span = max(1, (cutoff - edge.timestamp).days)
            recency = span ** (-rho / 2)
            validity = min(recency + salience, 1.0)
        hyperedge_scores[edge.id] = {"distance": distance if math.isfinite(distance) else None,
                                    "recency": recency, "salience": salience,
                                    "support_records": count, "validity": validity}
        # Overlapping chunks of the same source/fact are a single contribution.
        fact_key = (edge.record_id, canonical(edge.proposition))
        for entity in edge.entities:
            previous = incident[entity].get(fact_key)
            incident[entity][fact_key] = validity if previous is None else min(previous, validity)
    entity_scores = {v: sum(values.values()) / len(values) for v, values in incident.items()}
    edge_scores = {}
    outgoing = defaultdict(list)
    best = {}
    for edge in active.causal_edges:
        rho = min((distances.get(edge.cause, math.inf) + distances.get(edge.effect, math.inf))
                  / (2 * d_max), 1)
        omega = ((entity_scores.get(edge.cause, 0.5) + entity_scores.get(edge.effect, 0.5)) / 2) ** rho
        weighted = edge.support * omega
        edge_scores[edge.id] = {"omega": omega, "support": edge.support, "weighted_support": weighted}
        key = (edge.cause, edge.effect, edge.type)
        if key not in best or weighted > best[key][1]:
            best[key] = (edge, weighted)
    for edge, weighted in best.values():
        outgoing[edge.cause].append((edge, weighted))
    roots = {canonical(t) for t in targets}
    queue = deque((v, (v,), (), 1.0, 1) for v in sorted(outgoing) if v not in roots)
    chains = []
    expansions = 0
    while queue and expansions < max_expansions:
        node, visited, edges, score, polarity = queue.popleft()
        for edge, weighted in sorted(outgoing[node], key=lambda item: item[0].id):
            if expansions >= max_expansions:
                break
            expansions += 1
            if edge.effect in visited:
                continue
            value = score * weighted
            if value < min_score:
                continue
            path = edges + (edge.id,)
            sign = polarity * (-1 if edge.type == "prevents" else 1)
            if edge.effect in roots:
                chains.append({"entities": list(visited + (edge.effect,)), "edges": list(path),
                               "score": value, "polarity": sign})
            elif len(path) < d_max:
                queue.append((edge.effect, visited + (edge.effect,), path, value, sign))
    chains.sort(key=lambda chain: (-chain["score"], chain["edges"]))
    return {"cutoff": cutoff.isoformat(), "distances": distances,
            "hyperedges": hyperedge_scores, "entities": entity_scores, "edges": edge_scores,
            "chains": chains[:max_chains], "search_truncated": bool(queue) or expansions >= max_expansions,
            "calibration_status": graph.calibration_status,
            "score_semantics": "textual_support_times_temporal_validity_not_causal_effect_or_forecast_probability"}
