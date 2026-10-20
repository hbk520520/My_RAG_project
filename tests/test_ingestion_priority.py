"""Intake policy boundaries and read-only planning; no OCR or model calls."""
from datetime import date, datetime, timedelta
import json

import pytest
from pydantic import ValidationError

from dataset.ingestion_priority import IngestionMaterial, assess_priority, plan_ingestion


AS_OF = date(2026, 10, 20)


def material(document_id="doc", **values):
    return IngestionMaterial(document_id=document_id, source_file=f"{document_id}.pdf", **values)


@pytest.mark.parametrize("role,priority", [
    ("core", "P1"), ("supporting", "P2"), ("reference", "P3"), ("unclassified", "P2")])
def test_declared_role_sets_default_priority(role, priority):
    result = assess_priority(material(role=role), as_of=AS_OF)
    assert result.priority == priority
    assert result.classification_required == (role == "unclassified")


@pytest.mark.parametrize("days,priority,reason", [
    (-1, "P0", "deadline_overdue"), (0, "P0", "deadline_within_window"),
    (7, "P0", "deadline_within_window"), (8, "P2", "deadline_outside_window")])
def test_deadline_window_is_inclusive_and_keeps_overdue_work_visible(days, priority, reason):
    result = assess_priority(material(deadline=AS_OF + timedelta(days=days)), as_of=AS_OF)
    assert result.priority == priority and reason in result.reasons
    assert result.assessed_on == AS_OF and result.urgent_within_days == 7


def test_window_can_be_changed_without_changing_policy_code():
    item = material(role="reference", deadline=AS_OF + timedelta(days=1))
    assert assess_priority(item, as_of=AS_OF, urgent_within_days=0).priority == "P3"
    assert assess_priority(item, as_of=AS_OF, urgent_within_days=1).priority == "P0"


def test_review_requirement_is_independent_of_priority():
    result = assess_priority(material(role="reference", review_required=True), as_of=AS_OF)
    assert result.priority == "P3" and result.review_required
    assert assess_priority(material(expedited=True), as_of=AS_OF).priority == "P0"


def test_manual_override_keeps_deadline_and_reason_for_audit():
    result = assess_priority(material(role="core", deadline=AS_OF, priority_override="P3",
                                     override_reason=" Owner deferred this duplicate copy "), as_of=AS_OF)
    assert result.priority == "P3"
    assert result.reasons == ["role:core", "deadline_within_window",
                              "manual_override:Owner deferred this duplicate copy"]


@pytest.mark.parametrize("values", [
    {"priority_override": "P0"}, {"priority_override": "P0", "override_reason": "  "},
    {"override_reason": "no override"}, {"role": "official"}, {"expedited": "false"},
    {"review_required": 1}, {"authority_score": 1}, {"priority_override": "P4"},
    {"deadline": 0}, {"deadline": "20261020"}, {"deadline": "2026-02-30"},
    {"deadline": datetime(2026, 10, 20)}])
def test_invalid_intake_facts_are_not_silently_coerced_or_ignored(values):
    with pytest.raises(ValidationError):
        material(**values)


@pytest.mark.parametrize("field", ["document_id", "source_file"])
def test_identifiers_must_not_be_blank(field):
    values = {"document_id": "doc", "source_file": "doc.pdf", field: "  "}
    with pytest.raises(ValidationError):
        IngestionMaterial(**values)


def test_manifest_calendar_dates_are_preserved_as_dates():
    item = material(deadline="2026-10-23")
    assert item.deadline == date(2026, 10, 23)
    assert assess_priority(item, as_of=AS_OF).priority == "P0"


def test_plan_preserves_all_materials_and_stable_ties_without_mutating_inputs():
    items = [material("reference", role="reference"), material("unknown"),
             material("core", role="core"), material("support", role="supporting"),
             material("later", deadline=AS_OF + timedelta(days=5)),
             material("overdue", deadline=AS_OF - timedelta(days=1))]
    before = [item.model_dump(mode="json") for item in items]
    plan = plan_ingestion(items, as_of=AS_OF)
    assert [entry["material"]["document_id"] for entry in plan] == [
        "overdue", "later", "core", "unknown", "support", "reference"]
    assert [item.model_dump(mode="json") for item in items] == before
    assert plan == plan_ingestion(items, as_of=AS_OF)


def test_duplicate_document_ids_are_rejected():
    with pytest.raises(ValueError, match="Duplicate document_id"):
        plan_ingestion([material(), material()], as_of=AS_OF)


@pytest.mark.parametrize("window", [-1, True, 1.5])
def test_empty_plans_still_validate_the_window(window):
    with pytest.raises(ValueError, match="nonnegative integer"):
        plan_ingestion([], as_of=AS_OF, urgent_within_days=window)


def test_assessment_date_is_explicit_and_empty_plan_is_supported():
    assert plan_ingestion([], as_of=AS_OF) == []
    with pytest.raises(TypeError, match="calendar date"):
        plan_ingestion([], as_of=datetime(2026, 10, 20))


def test_cli_plans_without_reading_pdf_or_opening_memory(tmp_path, monkeypatch, capsys):
    from dataset import long_term_memory
    from tools.long_term_memory import main

    def forbidden_store(*args, **kwargs):
        raise AssertionError("Planning must not open the knowledge database")
    monkeypatch.setattr(long_term_memory, "MemoryStore", forbidden_store)
    manifest = tmp_path / "intake.json"
    manifest.write_text(json.dumps([material(role="core").model_dump(mode="json")]), encoding="utf-8")
    assert main(["prioritize", "--manifest", str(manifest), "--as-of", AS_OF.isoformat()]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result[0]["assessment"]["priority"] == "P1"
    assert result[0]["material"]["source_file"] == "doc.pdf"


def test_cli_rejects_non_list_manifest(tmp_path):
    from tools.long_term_memory import main
    manifest = tmp_path / "intake.json"
    manifest.write_text('{"documents": []}', encoding="utf-8")
    with pytest.raises(ValidationError):
        main(["prioritize", "--manifest", str(manifest), "--as-of", AS_OF.isoformat()])
