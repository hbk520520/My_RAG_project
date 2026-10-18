"""Offline contract tests. Scripted verifier labels are fixtures, not quality measurements."""
import json
from datetime import date

import pytest
from pydantic import ValidationError

from causal_evidence.calibration import (SupportCalibrator, SupportLabel, calibration_metrics,
                                         fit_calibration)
from causal_evidence.chunking import admissible, chunk_record, parse_timestamp
from causal_evidence.extraction import canonical, extract_verified
from causal_evidence.graph import active_graph, build_graph, reverse_distances, score_graph
from causal_evidence.models import (CausalEdge, CausalTemporalGraph, EvidenceRecord, Hyperedge)


class ByteEncoding:
    """Deterministic tokenizer substitute; deliberately splits multibyte characters."""
    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))

    def decode_single_token_bytes(self, token):
        return bytes([token])


ENCODING = ByteEncoding()
TEXT = "预算削减导致资金紧张。"
CUTOFF = date(2025, 10, 1)


def payload(text=TEXT, cause="预算削减", effect="资金紧张", strength=0.9, kind="causes"):
    return {"propositions": [{"sentence": text, "entities": [{"name": cause}, {"name": effect}],
                               "evidence_quote": text}],
            "causal_relations": [{"cause": cause, "effect": effect, "type": kind,
                                  "strength": strength, "proposition_index": 0, "evidence_quote": text}]}


class ScriptedCalls:
    def __init__(self, extraction=None, classification="causal", quote=TEXT):
        self.extraction = extraction if extraction is not None else payload()
        self.classification = classification
        self.quote = quote
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        if "Causal Extraction:" in messages[0]["content"]:
            return json.dumps(self.extraction, ensure_ascii=False)
        candidates = json.loads(messages[-1]["content"])["candidates"]
        return json.dumps({"verdicts": [{"candidate_index": c["candidate_index"],
            "classification": self.classification, "supported": self.classification == "causal",
            "evidence_quote": self.quote, "rationale": "fixture verdict"} for c in candidates]}, ensure_ascii=False)


def record(**kwargs):
    return EvidenceRecord(record_id="news-1", text=TEXT, timestamp=date(2025, 9, 1), **kwargs)


def one_chunk():
    return chunk_record(record(), encoding=ENCODING)[0]


def chain_graph():
    hyperedges = [Hyperedge(id="h1", chunk_id="x1", record_id="r1", proposition="a causes b",
                            entities=["a", "b"], evidence_quote="a causes b", timestamp=date(2025, 9, 1)),
                  Hyperedge(id="h2", chunk_id="x2", record_id="r2", proposition="b prevents c",
                            entities=["b", "c"], evidence_quote="b prevents c", timestamp=date(2025, 9, 20))]
    edges = [CausalEdge(id="c1", cause="a", effect="b", type="causes", raw_support=0.9,
                        hyperedge_id="h1", evidence_quote="a causes b", verification_reason="fixture"),
             CausalEdge(id="c2", cause="b", effect="c", type="prevents", raw_support=0.8,
                        hyperedge_id="h2", evidence_quote="b prevents c", verification_reason="fixture")]
    return CausalTemporalGraph(entities=["a", "b", "c"], hyperedges=hyperedges, causal_edges=edges)


def test_unicode_token_windows_keep_exact_offsets_and_overlap():
    source = EvidenceRecord(record_id="unicode", text="中文🙂 repeated text " * 100)
    chunks = chunk_record(source, encoding=ENCODING)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text == source.text[chunk.char_start:chunk.char_end]
        assert chunk.token_end - chunk.token_start <= 512
        assert "\ufffd" not in chunk.text
    for left, right in zip(chunks, chunks[1:]):
        assert 0 <= left.token_end - right.token_start <= 64
        assert right.char_start <= left.char_end
    assert chunks[-1].char_end == len(source.text)


@pytest.mark.parametrize("size,overlap", [(0, 0), (64, 64), (20, -1), (20, 30)])
def test_bad_chunk_parameters(size, overlap):
    with pytest.raises(ValueError):
        chunk_record(record(), chunk_size=size, overlap=overlap, encoding=ENCODING)


@pytest.mark.parametrize("text,expected", [("2025年9月消息", date(2025, 9, 30)),
    ("2025-09-12", date(2025, 9, 12)), ("2025年9月12日", date(2025, 9, 12)),
    ("2025-09-12 与 2025-09-13", None), ("2025-02-31", None), ("2025年13月", None), ("昨日", None)])
def test_conservative_timestamp_parsing(text, expected):
    assert parse_timestamp(text) == expected


def test_verified_edges_retain_quote_raw_score_and_hyperedge():
    call = ScriptedCalls()
    graph = build_graph([record()], call, cutoff=CUTOFF, encoding=ENCODING)
    assert graph.entities == sorted(["预算削减", "资金紧张"])
    assert len(graph.hyperedges) == len(graph.causal_edges) == 1
    assert graph.causal_edges[0].raw_support == 0.9
    assert graph.causal_edges[0].calibrated_support is None
    assert graph.causal_edges[0].evidence_quote == TEXT
    assert graph.calibration_status == "uncalibrated"
    assert len(call.calls) == 2


@pytest.mark.parametrize("classification", ["correlation", "negated", "hypothetical", "unsupported"])
def test_noncausal_claim_is_not_a_causal_edge(classification):
    hyperedges, edges, rejected = extract_verified(one_chunk(), ScriptedCalls(classification=classification))
    assert hyperedges and not edges
    assert rejected[0].reason.startswith(classification)


def test_hallucinated_entity_never_reaches_verifier():
    call = ScriptedCalls(payload(cause="不存在的公司"))
    _, edges, rejected = extract_verified(one_chunk(), call)
    assert not edges and rejected
    assert len(call.calls) == 1


def test_verifier_cannot_supply_invented_quote():
    _, edges, rejected = extract_verified(one_chunk(), ScriptedCalls(quote="预算削减导致资金紧张，明日破产。"))
    assert not edges and rejected[0].reason == "ungrounded_verifier_quote"


def test_unknown_endpoint_and_self_edges_rejected():
    data = payload()
    data["causal_relations"][0]["cause"] = "其他"
    _, edges, rejected = extract_verified(one_chunk(), ScriptedCalls(data))
    assert not edges and rejected
    data = payload(cause="资金紧张")
    _, edges, rejected = extract_verified(one_chunk(), ScriptedCalls(data))
    assert not edges and rejected[0].reason == "self_causal_edge"


@pytest.mark.parametrize("value", [0, -0.1, 1.1, float("nan"), float("inf")])
def test_invalid_strength_is_a_loud_schema_error(value):
    with pytest.raises(ValidationError):
        extract_verified(one_chunk(), ScriptedCalls(payload(strength=value)))


def test_malformed_verification_does_not_fall_back_to_extractor():
    call = ScriptedCalls()
    def incomplete(messages):
        return call(messages) if "Extraction:" in messages[0]["content"] else '{"verdicts": []}'
    with pytest.raises(ValueError, match="exactly one"):
        extract_verified(one_chunk(), incomplete)


def test_invalid_json_is_not_swallowed():
    with pytest.raises(ValidationError):
        extract_verified(one_chunk(), lambda messages: "not json")


@pytest.mark.parametrize("when", [date(2025, 10, 1), date(2025, 10, 2)])
def test_post_cutoff_evidence_is_excluded_before_model_call(when):
    call = ScriptedCalls()
    graph = build_graph([EvidenceRecord(record_id="future", text=TEXT, timestamp=when)], call,
                        cutoff=CUTOFF, encoding=ENCODING)
    assert not graph.entities and not call.calls


def test_known_future_publication_blocks_old_event_claim():
    call = ScriptedCalls()
    graph = build_graph([record(available_at=date(2025, 11, 1))], call, cutoff=CUTOFF, encoding=ENCODING)
    assert not graph.entities and not call.calls


def test_undated_evidence_requires_verified_availability():
    source = EvidenceRecord(record_id="undated", text=TEXT, available_at=date(2025, 9, 1))
    assert not admissible(source, CUTOFF)
    assert admissible(source.model_copy(update={"availability_verified": True}), CUTOFF)
    assert not admissible(source.model_copy(update={"available_at": None, "availability_verified": True}), CUTOFF)


def test_duplicates_do_not_repeat_extraction_or_inflate_recurrence():
    call = ScriptedCalls()
    graph = build_graph([record(), record()], call, cutoff=CUTOFF, encoding=ENCODING)
    assert len(call.calls) == 2 and len(graph.hyperedges) == 1
    result = score_graph(graph, ["资金紧张"], CUTOFF)
    assert next(iter(result["hyperedges"].values()))["support_records"] == 1


def test_conflicting_record_ids_fail():
    with pytest.raises(ValueError, match="Conflicting"):
        build_graph([record(), record().model_copy(update={"text": "different"})], ScriptedCalls(),
                    encoding=ENCODING)


def test_bfs_and_chain_scores_follow_direction_and_product():
    graph = chain_graph()
    assert reverse_distances(graph, ["c"]) == {"c": 0, "b": 1, "a": 2}
    result = score_graph(graph, ["c"], CUTOFF, d_max=3)
    first = result["hyperedges"]["h1"]
    assert first["recency"] == pytest.approx(30 ** (-1 / 6))
    full = next(chain for chain in result["chains"] if chain["entities"] == ["a", "b", "c"])
    expected = result["edges"]["c1"]["weighted_support"] * result["edges"]["c2"]["weighted_support"]
    assert full["score"] == pytest.approx(expected)
    assert full["polarity"] == -1
    assert all(chain["entities"][-1] == "c" for chain in result["chains"])


def test_future_edges_cannot_create_shortcuts_in_active_graph():
    graph = chain_graph()
    graph.hyperedges[1].timestamp = date(2025, 11, 1)
    assert len(active_graph(graph, CUTOFF).causal_edges) == 1
    with pytest.raises(ValueError, match="Targets"):
        score_graph(graph, ["c"], CUTOFF)


def test_overlapping_same_fact_does_not_inflate_entity_or_salience_weight():
    graph = chain_graph()
    before = score_graph(graph, ["c"], CUTOFF)
    graph.hyperedges.append(graph.hyperedges[0].model_copy(update={"id": "overlap"}))
    after = score_graph(graph, ["c"], CUTOFF)
    assert after["entities"] == before["entities"]
    assert after["hyperedges"]["overlap"]["support_records"] == 1


def test_missing_time_has_explicit_fallback_only_after_admission():
    graph = chain_graph()
    graph.hyperedges[0].timestamp = None
    graph.hyperedges[0].available_at = date(2025, 9, 1)
    graph.hyperedges[0].availability_verified = True
    result = score_graph(graph, ["c"], CUTOFF)
    assert result["hyperedges"]["h1"]["validity"] == 0.25
    assert result["hyperedges"]["h1"]["recency"] is None


def test_cycles_are_bounded_and_do_not_repeat_nodes():
    graph = chain_graph()
    graph.causal_edges.append(graph.causal_edges[0].model_copy(update={"id": "back", "cause": "b", "effect": "a"}))
    result = score_graph(graph, ["c"], CUTOFF)
    assert all(len(set(c["entities"])) == len(c["entities"]) for c in result["chains"])


def test_calibration_metrics_boundary_scores():
    result = calibration_metrics([0, 1], [0, 1])
    assert result["brier"] == result["ece"] == 0
    with pytest.raises(ValueError):
        calibration_metrics([], [])


def labeled_rows():
    return [SupportLabel(source_id=f"source-{i}", raw_support=score, supported=label)
            for i in range(8) for score, label in [(0.2, 0), (0.8, 1)]]


def test_calibration_uses_disjoint_sources_and_roundtrips():
    model, report = fit_calibration(labeled_rows(), extractor_id="test-v1", trained_through=date(2025, 1, 1))
    assert not set(model.training_sources) & set(model.evaluation_sources)
    restored = SupportCalibrator.model_validate_json(model.model_dump_json())
    assert restored.predict(0.5) == pytest.approx(0.5)
    assert restored.predict(0.2) == 0 and restored.predict(0.8) == 1
    assert report["raw"]["n"] == report["calibrated"]["n"] == 4
    assert report["calibrated"]["brier"] == 0  # synthetic fixture only


def test_calibration_requires_source_groups_and_both_labels():
    with pytest.raises(ValueError, match="four"):
        fit_calibration(labeled_rows()[:2], extractor_id="x", trained_through=CUTOFF)
    with pytest.raises(ValueError, match="both labels"):
        fit_calibration([r for r in labeled_rows() if r.supported], extractor_id="x", trained_through=CUTOFF)


def test_calibration_preserves_raw_support_and_blocks_future_or_wrong_model():
    model, _ = fit_calibration(labeled_rows(), extractor_id="v1", trained_through=date(2025, 1, 1))
    graph = build_graph([record()], ScriptedCalls(), calibrator=model, extractor_id="v1", encoding=ENCODING)
    assert graph.causal_edges[0].raw_support == 0.9
    assert graph.causal_edges[0].calibrated_support == 1
    with pytest.raises(ValueError, match="extractor"):
        build_graph([], ScriptedCalls(), calibrator=model, extractor_id="other", encoding=ENCODING)
    with pytest.raises(ValueError, match="predate"):
        build_graph([], ScriptedCalls(), calibrator=model, extractor_id="v1", cutoff=date(2025, 1, 1), encoding=ENCODING)


def test_conservative_entity_normalization():
    assert canonical("Ａ  Company") == "a company"
    assert canonical("资金紧张") != canonical("资金短缺")


def test_bounded_search_reports_its_limit():
    result = score_graph(chain_graph(), ["c"], CUTOFF, max_expansions=1)
    assert result["search_truncated"]


def test_invalid_graph_reference_is_rejected():
    graph = chain_graph().model_dump()
    graph["causal_edges"][0]["hyperedge_id"] = "missing"
    with pytest.raises(ValidationError, match="hyperedge"):
        CausalTemporalGraph.model_validate(graph)


def test_hallucinated_proposition_with_real_entities_is_rejected():
    data = payload()
    data["propositions"][0]["sentence"] = "预算削减导致资金紧张以及未提及的破产"
    _, edges, rejected = extract_verified(one_chunk(), ScriptedCalls(data))
    assert not edges
    assert rejected[0].reason == "ungrounded_proposition_quote"
    assert rejected[-1].candidate.strength == 0.9
