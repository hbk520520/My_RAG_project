"""PA-Tool algorithm and actual planning/navigation interfaces, without model calls."""
import json

import pytest

from agent_tools.schema_alignment import (
    CATALOGS, SchemaAlignment, build_profile, edit_distance, for_role, rank_candidates, sample_names,
)
from config_loader import cfg


def samples_for(role):
    return {f"{kind}.{name}": {"reference": "aligned_" + name.lower(),
                             "samples": ["aligned_" + name.lower()] * 3}
            for kind in ("fields", "tools") for name in CATALOGS[role][kind]}


def install(role, tmp_path, monkeypatch, records=None):
    profile = build_profile(role, records or samples_for(role))
    path = tmp_path / f"{role}.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.setitem(cfg._data, "tool_schema_alignment", {"profiles": {role: str(path)}})
    return profile, path


def test_peakedness_uses_edit_distance_clusters_and_greedy_reference_tie_break():
    samples = ["rare"] * 4 + ["tool_a"] * 2 + ["tool_b"] * 2 + ["tool_c"] * 2
    ranked = rank_candidates(samples, "tool_b", alpha=0.2)
    assert ranked[0] == {"name": "tool_b", "peakedness": 5, "reference_distance": 0}
    assert next(item for item in ranked if item["name"] == "rare")["peakedness"] == 3
    assert rank_candidates(["ab", "ac"], "ac", alpha=0)[0]["name"] == "ac"
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "tool") == 4


def test_priority_locking_uses_next_candidate_instead_of_overwriting_tools():
    records = samples_for("replanner")
    records["tools.GRAPH_TRAVERSAL"] = {"reference": "retrieve", "samples": ["retrieve"] * 4}
    records["tools.GLOBAL_DENSE_WORMHOLE"] = {
        "reference": "retrieve", "samples": ["retrieve"] * 2 + ["cross_search"]}
    profile = build_profile("replanner", records)
    assert profile["names"]["tools.GRAPH_TRAVERSAL"] == "retrieve"
    assert profile["names"]["tools.GLOBAL_DENSE_WORMHOLE"] == "cross_search"
    records["tools.GLOBAL_DENSE_WORMHOLE"]["samples"] = ["retrieve"]
    with pytest.raises(ValueError, match="No unique candidate"):
        build_profile("replanner", records)


def test_unrenamed_fields_cannot_be_accidentally_reassigned():
    records = samples_for("replanner")
    records["fields.task_desc"] = {"reference": "evidence", "samples": ["evidence", "query"]}
    assert build_profile("replanner", records)["names"]["fields.task_desc"] == "query"


@pytest.mark.parametrize("samples,reference,alpha", [([], "x", .2), (["x"], "x", -1),
    (["x"], "x", 1.1), (["not a name"], "x", .2), (["x"], "{bad}", .2)])
def test_invalid_candidates_are_reported_without_silent_correction(samples, reference, alpha):
    with pytest.raises(ValueError):
        rank_candidates(samples, reference, alpha)


def test_sampling_queries_target_model_with_greedy_and_stochastic_temperatures():
    calls = []
    def model(messages, *, temperature):
        calls.append((messages, temperature))
        return '{"name":"preferred_identifier"}'
    records = sample_names("replanner", model, count=2)
    assert len(records) == 6 and len(calls) == 18
    assert [temperature for _, temperature in calls] == [0, .4, .4] * 6
    assert all(value == {"reference": "preferred_identifier", "samples": ["preferred_identifier"] * 2}
               for value in records.values())


def test_schema_and_decoding_preserve_types_values_and_nested_evidence():
    alignment = SchemaAlignment(build_profile("replanner", samples_for("replanner")))
    schema = alignment.schema()
    assert schema["required"] == ["aligned_task_queue"]
    task = schema["properties"]["aligned_task_queue"]["items"]
    assert task["properties"]["aligned_engine"]["enum"] == [
        "aligned_graph_traversal", "aligned_global_dense_wormhole"]
    assert task["properties"]["aligned_engine"]["default"] == "aligned_graph_traversal"
    assert "task_id" not in task["required"]
    assert task["properties"]["evidence"]["$ref"] == "#/$defs/EvidencePolicy"
    text = "Keep task_desc and GRAPH_TRAVERSAL as literal case evidence"
    decoded = alignment.decode({"aligned_task_queue": [{"aligned_task_desc": text,
        "aligned_engine": "aligned_global_dense_wormhole", "evidence": {"causal": "off"}}]})
    assert decoded == {"task_queue": [{"task_desc": text, "engine": "GLOBAL_DENSE_WORMHOLE",
                                        "evidence": {"causal": "off"}}]}
    with pytest.raises(ValueError, match="Duplicate translated"):
        alignment.decode({"aligned_task_queue": [], "task_queue": []})


def test_replanner_decodes_aligned_fields_before_task_validation(tmp_path, monkeypatch):
    from legal_agent.operators import AgenticNodesOperator
    install("replanner", tmp_path, monkeypatch)
    ops = AgenticNodesOperator.__new__(AgenticNodesOperator)
    calls = []
    def call(role, messages, **kwargs):
        calls.append((role, messages, kwargs))
        return json.dumps({"aligned_task_queue": [{"aligned_task_desc": "检索跨领域依据",
            "aligned_engine": "aligned_global_dense_wormhole", "aligned_rationale": "局部检索无结果",
            "evidence": {"causal": "off"}}]})
    ops._role_messages = call
    result = ops.replan_with_wormhole("GRAPH_TRAVERSAL 是案件原话", ["task_desc 是原始事实"], "engine 原始错误")
    assert result[0]["task_id"] == "replan-0"
    assert result[0]["engine"] == "GLOBAL_DENSE_WORMHOLE"
    assert result[0]["task_desc"] == "检索跨领域依据" and result[0]["evidence"]["causal"] == "off"
    role, messages, kwargs = calls[0]
    assert role == "replanner" and kwargs["require_json"]
    assert "aligned_graph_traversal" in messages[0]["content"]
    assert "engine 原始错误" in messages[0]["content"]
    assert "GRAPH_TRAVERSAL 是案件原话" in messages[1]["content"]
    assert "task_desc 是原始事实" in messages[1]["content"]


def test_no_profile_keeps_the_original_replanner_prompt_and_contract(monkeypatch):
    import prompts
    from legal_agent.operators import AgenticNodesOperator
    monkeypatch.setitem(cfg._data, "tool_schema_alignment", {"profiles": {}})
    ops = AgenticNodesOperator.__new__(AgenticNodesOperator)
    seen = []
    def call(role, messages, **kwargs):
        seen.extend(messages)
        return '{"task_queue":[{"task_desc":"原查询","engine":"GRAPH_TRAVERSAL"}]}'
    ops._role_messages = call
    assert ops.replan_with_wormhole("q", [], "failure")[0]["engine"] == "GRAPH_TRAVERSAL"
    assert seen == prompts.build_replanner_messages("q", [], {"fail_count": 0, "fail_log": "failure"},
        '{"task_queue": [{"task_desc": "...", "engine": "GLOBAL_DENSE_WORMHOLE", "rationale": "..."}]}')


def test_navigator_executes_aliases_and_records_canonical_actions(tmp_path, monkeypatch):
    from dataset.document_navigation import NavigationSettings, retrieve
    from dataset.document_store import DocumentReader
    from tests.test_pageindex import imported
    from tests.test_causal_evidence import ByteEncoding, TEXT
    install("navigator", tmp_path, monkeypatch)
    graph = imported()
    doc = DocumentReader.from_engine(graph).list_documents()[0]
    actions = iter([
        {"aligned_action": "aligned_read_pages", "aligned_document_id": doc["document_id"],
         "aligned_version": doc["version"], "aligned_pages": [1]},
        {"aligned_action": "aligned_finish"},
    ])
    seen = []
    def call(messages):
        seen.append(list(messages))
        return json.dumps(next(actions))
    result = retrieve(graph, "read_pages 是原问题中的文本", lambda: pytest.fail("wrong tool"), call,
                      settings=NavigationSettings(navigation="agent"), encoding=ByteEncoding())
    assert [hit["content"] for hit in result.hits] == [TEXT]
    assert [action["action"] for action in result.trace["actions"]] == ["read_pages", "finish"]
    assert "aligned_read_pages" in seen[0][0]["content"]
    assert "read_pages 是原问题中的文本" in seen[0][1]["content"]
    assert '"aligned_action": "aligned_read_pages"' in seen[1][-2]["content"]
    assert result.trace["model_calls"] == 2


def test_unknown_tools_are_not_fuzzy_matched_to_a_real_action():
    from dataset.document_navigation import Action
    from pydantic import ValidationError
    alignment = SchemaAlignment(build_profile("navigator", samples_for("navigator")))
    with pytest.raises(ValidationError):
        Action.model_validate(alignment.decode({"aligned_action": "read_anything"}))


def test_mapping_is_bound_to_role_model_and_runtime_resume_identity(tmp_path, monkeypatch):
    from agent_runtime.service import _execution_hashes
    monkeypatch.setitem(cfg._data, "tool_schema_alignment", {"profiles": {}})
    baseline = _execution_hashes()
    profile, path = install("replanner", tmp_path, monkeypatch)
    aligned = _execution_hashes()
    assert baseline[0] != aligned[0] and baseline[1] == aligned[1]
    profile["names"]["tools.GRAPH_TRAVERSAL"] = "graph_search"
    path.write_text(json.dumps(profile), encoding="utf-8")
    assert aligned[0] != _execution_hashes()[0]
    assert for_role("replanner").tools["GRAPH_TRAVERSAL"] == "graph_search"
    profile["model"]["model"] = "other-model"
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="currently configured"):
        for_role("replanner")
    monkeypatch.setitem(cfg._data, "tool_schema_alignment", {"profiles": {}})
    assert _execution_hashes() == baseline and for_role("replanner") is None


def test_cli_builds_from_saved_samples_without_loading_a_model(tmp_path, monkeypatch):
    from model.tool_schema import main
    source, target = tmp_path / "samples.json", tmp_path / "profile.json"
    source.write_text(json.dumps(samples_for("replanner")), encoding="utf-8")
    main(["--role", "replanner", "--samples", str(source), "--output", str(target)])
    assert json.loads(target.read_text(encoding="utf-8"))["method"] == "pa-tool-v1"
    with pytest.raises(SystemExit):
        main(["--role", "replanner", "--samples", str(source), "--output", str(target)])
    monkeypatch.setitem(cfg._data["llm"], "backend", "api")
    with pytest.raises(SystemExit):
        main(["--role", "replanner", "--generate", "--output", str(tmp_path / "new.json")])
