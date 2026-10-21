"""Contract and replay checks; these do not measure a trained model."""
import hashlib
import json

import pytest
from pydantic import ValidationError

from model.data.synthetic.candidates import CandidateProvider, pending_candidate
from model.data.synthetic.contracts import Config, Source
from model.data.synthetic.export import export_dataset
from model.data.synthetic.generation import generate_groups, trajectory_candidates
from model.data.synthetic.simulation import Environment, oracle, reference, score


def tasks(**changes):
    config = Config(groups=4, ood_groups=1, max_variants=2, **changes)
    return [task for _, _, group in generate_groups(config) for task in group]


@pytest.mark.parametrize("family", ["lookup", "composition", "dependency", "parallel", "recovery", "constraints"])
def test_all_families_and_ood_have_executable_references(family):
    for task in tasks(family_weights={family: 1}):
        result = score(task, reference(task))
        assert result["final_correct"] and result["steps_correct"] and result["state_correct"]
        if task.split == "ood":
            assert task.response_format == "rows" and task.tool_names["read"] != "read"
        assert "label" not in task.public_input()


def test_dependency_requires_prior_read_and_parallel_requires_handoff():
    task = tasks(family_weights={"dependency": 1})[0]
    environment = Environment(task)
    with pytest.raises(ValueError, match="dependency"):
        environment.execute("coordinator", task.tool_names["read"], {"record_id": task.records[-1].id})
    parallel = tasks(family_weights={"parallel": 1})[0]
    environment = Environment(parallel)
    for record, actor in parallel.public_input()["assignments"].items():
        environment.execute(actor, parallel.tool_names["read"], {"record_id": record})
    with pytest.raises(ValueError, match="delivered"):
        environment.execute("coordinator", parallel.tool_names["combine"], {})
    with pytest.raises(ValueError, match="another role"):
        Environment(parallel).execute("coordinator", parallel.tool_names["read"], {"record_id": parallel.records[0].id})
    gold = reference(parallel)
    pairs = [gold.steps[i:i+2] for i in range(0, 2 * len(parallel.records), 2)]
    gold.steps = [step for pair in reversed(pairs) for step in pair] + gold.steps[2 * len(parallel.records):]
    assert score(parallel, gold)["steps_correct"]


def test_constraints_cover_missing_permissions_and_budget():
    config = Config(groups=40, ood_groups=0, max_variants=2, family_weights={"constraints": 1})
    cases = [t for _, _, group in generate_groups(config) for t in group]
    assert {oracle(t).reason for t in cases} == {"missing_input", "permission_denied", "budget_exhausted"}
    forbidden = next(t for t in cases if not t.write_allowed)
    environment = Environment(forbidden)
    with pytest.raises(ValueError, match="permission"):
        environment.execute("coordinator", forbidden.tool_names["write"], {"value": 1})
    assert environment.store == {}
    exhausted = next(t for t in cases if t.max_calls == 0)
    with pytest.raises(ValueError, match="budget"):
        Environment(exhausted).execute("coordinator", exhausted.tool_names["read"], {"record_id": exhausted.records[0].id})


def test_conflicts_repetition_and_ground_truth_recomputed():
    base = tasks(family_weights={"composition": 1}, repetition_rate=1, conflict_rate=1,
                 unresolved_conflict_rate=0)[0]
    assert all(len(r.sources[0].statements) == 2 for r in base.records[1:])
    expected_sum = sum(next(s.facts["units"] for s in r.sources if s.authority == "primary" and s.active) for r in base.records)
    assert oracle(base).value == expected_sum
    conflict = tasks(family_weights={"lookup": 1}, conflict_rate=1, unresolved_conflict_rate=1)[0]
    assert oracle(conflict).status == "needs_evidence"
    before = reference(base)
    primary = next(s for s in base.records[0].sources if s.authority == "primary" and s.active)
    primary.facts["units"] += 2
    assert not score(base, before)["final_correct"]


def test_final_answer_and_steps_and_state_are_independent():
    task = tasks(family_weights={"composition": 1})[0]
    results = {t.name: score(task, t) for t in trajectory_candidates(task)}
    assert results["reference"]["reward"] > results["redundant"]["reward"] > 0
    assert results["fabricated_observation"]["final_correct"]
    assert not results["fabricated_observation"]["steps_correct"]
    assert results["invalid_tool_use"]["final_correct"]
    assert not results["invalid_tool_use"]["steps_correct"] and results["invalid_tool_use"]["reward"] == 0
    assert not results["wrong_final"]["final_correct"]
    assert results["wrong_final"]["steps_correct"]
    gold = reference(task)
    gold.steps = [s for s in gold.steps if s.tool != task.tool_names["write"]]
    result = score(task, gold)
    assert result["final_correct"] and not result["state_correct"] and result["reward"] == 0


def test_seed_reproducibility_and_real_source_exposures():
    config = Config(groups=24, ood_groups=2, max_variants=8)
    first = list(generate_groups(config))
    assert first == list(generate_groups(config))
    reordered = config.model_copy(update={"family_weights": dict(reversed(list(config.family_weights.items())))})
    assert first == list(generate_groups(reordered))
    assert len({len(group) for _, _, group in first}) > 1
    for identity, split, group in first:
        assert all(t.group_id == identity and t.split == split for t in group)
        known = {}
        for t in group:
            for r in t.records:
                source = r.sources[0]
                value = source.facts
                if t.family != "constraints":
                    assert known.setdefault(source.id, value) == value


@pytest.mark.parametrize("change", [
    {"family_weights": {}}, {"family_weights": {"lookup": -1}},
    {"family_weights": {"lookup": 1e308, "parallel": 1e308}},
    {"max_items": 25}, {"conflict_rate": 1.1}, {"train_fraction": 0.9, "validation_fraction": 0.2},
])
def test_config_boundaries(change):
    with pytest.raises(ValidationError):
        Config(**change)


def test_provider_cannot_promote_or_invent_source():
    sources = [Source(id="source", text="给定材料")]
    provider = CandidateProvider("fake:v1", lambda req: {"query": "问题", "proposed_answer": "答案", "source_ids": ["source"]})
    row = pending_candidate("g", "train", "legal_reasoning", sources, provider)
    assert row["verification"]["status"] == "pending" and row["proposed_label"] == "答案"
    with pytest.raises(ValidationError):
        pending_candidate("g", "train", "legal_reasoning", sources,
                          CandidateProvider("fake", lambda req: {"query": "问题", "verification": "verified"}))
    with pytest.raises(ValueError, match="unknown source"):
        pending_candidate("g", "train", "legal_reasoning", sources,
                          CandidateProvider("fake", lambda req: {"query": "问题", "source_ids": ["invented"]}))


def test_export_separation_budget_hashes_and_no_training_adapters(tmp_path):
    calls = []
    def generate(request):
        calls.append(request)
        return {"query": "开放问题", "proposed_answer": "尚待审核", "source_ids": [request["sources"][0]["id"]]}
    config = Config(groups=4, ood_groups=1, max_variants=2, include_trajectories=True, candidate_call_limit=2)
    output = tmp_path / "dataset"
    manifest = export_dataset(config, output, provider=CandidateProvider("stub:model-v1", generate))
    assert len(calls) == manifest["candidate_provider_calls"] == 2
    assert manifest["training_executed"] is False
    groups = {}; sources = {}
    for name, meta in manifest["files"].items():
        content = (output / name).read_bytes()
        assert hashlib.sha256(content).hexdigest() == meta["sha256"]
        rows = [json.loads(line) for line in content.splitlines()]
        assert len(rows) == meta["rows"]
        for row in rows:
            if ".verified." in name or ".pending." in name:
                assert groups.setdefault(row["group_id"], row["split"]) == row["split"]
                assert "label" not in row["input"] and "expected" not in row["input"]
                expected_status = "verified" if ".verified." in name else "pending"
                assert row["verification"]["status"] == expected_status
            if ".verified." in name:
                for r in row["environment"]["task"]["records"]:
                    for s in r["sources"]:
                        assert sources.setdefault(s["id"], row["split"]) == row["split"]
    with pytest.raises(FileExistsError):
        export_dataset(config, output, provider=CandidateProvider("stub", generate))


def test_no_implicit_provider_calls_and_provider_failure_is_visible(tmp_path):
    def fail(request):
        raise RuntimeError("provider failed")
    config = Config(groups=1, ood_groups=0)
    manifest = export_dataset(config, tmp_path / "offline", provider=CandidateProvider("never-called", fail))
    assert manifest["candidate_provider_calls"] == 0
    enabled = config.model_copy(update={"candidate_call_limit": 1})
    with pytest.raises(ValueError, match="injected provider"):
        export_dataset(enabled, tmp_path / "missing-provider")
    with pytest.raises(RuntimeError, match="provider failed"):
        export_dataset(enabled, tmp_path / "failure", provider=CandidateProvider("fail", fail))
    assert not (tmp_path / "failure").exists()


def test_seed_sources_only_produce_pending_and_cannot_cross_groups(tmp_path):
    seed = {"id": "law-a", "sources": [{"id": "doc-a", "text": "用户提供的种子资料"}]}
    manifest = export_dataset(Config(groups=1, ood_groups=0), tmp_path / "seeded", seed_groups=[seed])
    assert manifest["counts"]["seed_groups"] == 1
    with pytest.raises(ValueError, match="same group"):
        export_dataset(Config(groups=1, ood_groups=0), tmp_path / "bad-seeds", seed_groups=[
            seed, {"id": "law-b", "sources": [{"id": "doc-a", "text": "另一版本"}]}])
    assert not (tmp_path / "bad-seeds").exists()


def test_seed_validation_precedes_provider_and_seed_material_has_priority(tmp_path):
    calls = []
    def generate(request):
        calls.append(request)
        return {"query": "待审问题"}
    provider = CandidateProvider("stub", generate)
    config = Config(groups=1, ood_groups=0, candidate_call_limit=1)
    seed = {"id": "a", "sources": [{"id": "source-a", "text": "种子材料"}]}
    export_dataset(config, tmp_path / "prioritized", provider=provider, seed_groups=[seed])
    assert calls[0]["sources"][0]["id"] == "source-a"
    calls.clear()
    with pytest.raises(ValueError, match="same group"):
        export_dataset(config, tmp_path / "bad", provider=provider, seed_groups=[
            seed, {"id": "b", "sources": [{"id": "source-b", "text": "种子材料"}]}])
    assert calls == []


def test_cli_offline_with_optional_seeds(tmp_path, capsys):
    from model.data.synthetic.__main__ import main
    config = tmp_path / "config.json"
    config.write_text('{"groups":1,"ood_groups":0,"max_variants":1}', encoding="utf-8")
    seeds = tmp_path / "seeds.json"
    seeds.write_text('[{"id":"external","sources":[{"id":"source","text":"种子内容"}]}]', encoding="utf-8")
    output = tmp_path / "cli"
    main(["--output", str(output), "--config", str(config), "--seeds", str(seeds), "--trajectories"])
    result = json.loads(capsys.readouterr().out)
    assert result["candidate_provider_calls"] == 0 and result["training_executed"] is False
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["counts"]["seed_groups"] == 1
