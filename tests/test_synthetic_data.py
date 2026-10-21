"""Synthetic labels and trajectory replay are separate from model-quality evaluation."""
import hashlib
import json

import pytest
from pydantic import ValidationError

from model.data.synthetic.environment import Environment, resolve, score_trajectory
from model.data.synthetic.export import export_dataset
from model.data.synthetic.generator import generate, grader_record, preference_pair, trajectories
from model.data.synthetic.schemas import GenerationConfig, Sample


def one(**kwargs):
    return next(generate(GenerationConfig(groups=1, ood_groups=0, variants=1, **kwargs)))


@pytest.mark.parametrize("config", [
    {"family_weights": {}}, {"family_weights": {"aggregate": -1}},
    {"family_weights": {"aggregate": float("inf")}}, {"max_records": 25},
    {"family_weights": {"aggregate": 1e308, "lookup": 1e308}},
    {"train_fraction": 0.9, "validation_fraction": 0.2}, {"conflict_rate": 1.1},
])
def test_invalid_config(config):
    with pytest.raises(ValidationError):
        GenerationConfig(**config)


def test_reproducible_grouped_split_and_ood():
    config = GenerationConfig(groups=30, ood_groups=3, variants=3)
    samples = list(generate(config))
    assert samples == list(generate(config))
    reordered = config.model_copy(update={"family_weights": dict(reversed(list(config.family_weights.items())))})
    assert samples == list(generate(reordered))
    seen = {}
    for sample in samples:
        assert seen.setdefault(sample.group_id, sample.split) == sample.split
        if sample.split == "ood":
            assert "renamed_tools" in sample.transforms
            assert sample.tools["read"] != "read_record"
    # Changing the number of variants does not change other groups' first variants.
    assert [s for s in samples if s.sample_id.endswith("-v0")] == list(generate(config.model_copy(update={"variants": 1})))


def test_conflict_repetition_and_correct_arithmetic():
    sample = one(family_weights={"lookup": 1}, repetition_rate=1,
                 conflict_rate=1, unresolved_conflict_rate=0)
    source = next(d for d in sample.documents if d.authority == "primary")
    assert len(source.statements) == 2
    assert sample.expected.amount == f"{source.amount // 100}.{source.amount % 100:02d}"
    assert sample.expected.evidence_ids == [source.id]
    conflict = one(family_weights={"aggregate": 1}, conflict_rate=1, unresolved_conflict_rate=1)
    assert conflict.expected.status == "partial"
    assert conflict.expected.amount is None
    assert conflict.expected.missing_info_kind == "evidence"
    missing = one(family_weights={"missing": 1})
    assert missing.expected.missing_info_kind == "user"
    assert grader_record(missing)["status"] == "partial"


def test_recovery_and_each_step_independent_of_final_answer():
    sample = one(family_weights={"recovery": 1})
    candidates = list(trajectories(sample))
    scores = {t.name: score_trajectory(sample, t) for t in candidates}
    assert candidates[0].steps[0].observation == {"status": "retryable_error"}
    assert scores["reference"]["reward"] > scores["redundant"]["reward"] > 0
    assert scores["fabricated_observation"]["final_correct"]
    assert not scores["fabricated_observation"]["steps_correct"]
    assert scores["fabricated_observation"]["reward"] == 0
    assert scores["wrong_final"]["steps_correct"]
    assert not scores["wrong_final"]["final_correct"]
    environment = Environment(sample)
    with pytest.raises(ValueError, match="resolved successful reads"):
        environment.execute(sample.tools["sum"], {"record_ids": sample.record_ids})
    with pytest.raises(ValueError, match="Unknown"):
        environment.execute("external_write", {})


def test_ground_truth_is_not_taken_from_stored_expected():
    sample = one()
    trajectory = next(trajectories(sample))
    sample.expected = sample.expected.model_copy(update={"amount": "-1"})
    assert score_trajectory(sample, trajectory)["final_correct"]
    assert resolve(sample.documents, sample.record_ids) != sample.expected


@pytest.mark.parametrize("family", ["lookup", "aggregate", "recovery", "missing"])
def test_preference_negatives_require_reading_evidence(family):
    sample = one(family_weights={family: 1})
    chosen, rejected = preference_pair(sample)
    assert chosen != rejected
    assert rejected["extracted_facts"] != chosen["extracted_facts"]
    assert rejected["rationale"] == "资料覆盖完整，可执行确定性求和。"
    if family != "missing":
        assert rejected["status"] == chosen["status"] == "sufficient"


def test_export_manifest_hashes_training_contracts_and_preservation(tmp_path):
    from agent_runtime.contracts import Grade
    output = tmp_path / "new"
    config = GenerationConfig(groups=4, ood_groups=1, variants=2, conflict_rate=1)
    manifest = export_dataset(config, output)
    assert manifest["training_executed"] is False
    for name, info in manifest["files"].items():
        raw = (output / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == info["sha256"]
        rows = [json.loads(line) for line in raw.splitlines()]
        assert len(rows) == info["rows"]
        if ".tasks." in name:
            for row in rows:
                Sample.model_validate(row)
        if ".grader_sft." in name:
            for row in rows:
                Grade.model_validate_json(row["messages"][-1]["content"])
                assert "expected" not in row["messages"][-2]["content"]
        if ".grader_dpo." in name:
            for row in rows:
                Grade.model_validate_json(row["chosen"])
                Grade.model_validate_json(row["rejected"])
    with pytest.raises(FileExistsError):
        export_dataset(config, output)
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest


def test_failed_export_does_not_publish_dataset(tmp_path, monkeypatch):
    import model.data.synthetic.export as module
    def fail(_):
        yield one()
        raise ValueError("generation failed")
    monkeypatch.setattr(module, "generate", fail)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="generation failed"):
        export_dataset(GenerationConfig(groups=1), output)
    assert not output.exists()
