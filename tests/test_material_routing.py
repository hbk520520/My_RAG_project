"""Policy boundaries, streamed reports and actual PDF import gates; offline only."""
import hashlib
import json
import sys
from types import SimpleNamespace

import pymupdf
import pytest
from pydantic import ValidationError

from dataset.material_routing import MaterialFacts, route_material, route_record, write_routing_report
from dataset.pdf_intake import import_document, inspect_with_pdf_inspector


@pytest.mark.parametrize("values,category,path", [
    ({}, "unknown", "deferred"),
    ({"kind": "reference", "needs_ocr": True}, "reference", "light"),
    ({"kind": "case"}, "case", "light"),
    ({"kind": "case", "page_count": 8}, "case", "light"),
    ({"kind": "case", "page_count": 9}, "case", "deep"),
    ({"kind": "normative", "structured": True, "page_count": 100}, "normative", "light"),
    ({"kind": "normative", "complex_layout": True}, "normative", "deep"),
    ({"kind": "case", "needs_ocr": True}, "case", "deep"),
    ({"kind": "case", "derived": True}, "derived", "archive"),
])
def test_business_use_and_extraction_need_jointly_select_depth(values, category, path):
    decision = route_material(MaterialFacts(**values))
    assert (decision["category"], decision["path"]) == (category, path)
    assert decision["rule_id"] and decision["policy_version"] == "material-routing-v1"


@pytest.mark.parametrize("values", [{"kind": "official"}, {"needs_ocr": "false"},
                                         {"page_count": -1}, {"page_count": True}, {"importance": 1}])
def test_invalid_policy_inputs_fail_explicitly(values):
    with pytest.raises(ValidationError):
        MaterialFacts(**values)


def test_existing_law_schema_is_core_but_does_not_need_reprocessing():
    record = {"law_name": "Example", "article_no": "第一条", "text": "Original text",
              "source": "upstream@commit", "valid_to": "2000-01-01"}
    before = dict(record)
    decision = route_record(record)
    assert decision["category"] == "normative" and decision["path"] == "light"
    assert decision["review_flags"] == ["missing_valid_from"]
    assert record == before  # Historical versions and original text remain intact.


def test_unknown_and_derived_records_do_not_become_original_evidence():
    assert route_record({"text": "Some text"})["path"] == "deferred"
    assert route_record({"derived": True, "text": "Generated answer"})["path"] == "archive"


def test_compact_report_keeps_source_locators_without_copying_the_corpus(tmp_path):
    source = tmp_path / "corpus.jsonl"
    raw = json.dumps({"law_name": "Law", "article_no": "1", "text": "ORIGINAL BODY"}) + "\n"
    source.write_text(raw, encoding="utf-8")
    output = tmp_path / "report.jsonl"
    result = write_routing_report(source, output)
    entry = json.loads(output.read_text(encoding="utf-8"))
    assert result["paths"] == {"light": 1}
    assert entry["source_file"] == str(source.resolve()) and entry["line"] == 1
    assert entry["record_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert "ORIGINAL BODY" not in output.read_text(encoding="utf-8")
    assert source.read_text(encoding="utf-8") == raw


def test_failed_report_leaves_previous_output_intact_and_cleans_staging(tmp_path):
    source, output = tmp_path / "input.jsonl", tmp_path / "report.jsonl"
    source.write_text('{}\nnot-json\n', encoding="utf-8")
    output.write_text("previous report", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        write_routing_report(source, output)
    assert output.read_text() == "previous report"
    assert not list(tmp_path.glob(".routing-*.tmp"))
    with pytest.raises(ValueError, match="differ"):
        write_routing_report(source, source)


def pdf(path, texts):
    with pymupdf.open() as document:
        for text in texts:
            page = document.new_page()
            if text:
                page.insert_text((72, 72), text)
        document.save(path)


def forbidden(*args, **kwargs):
    raise AssertionError("Heavy processing must not run for this material")


@pytest.mark.parametrize("kind,status", [("auto", "deferred"), ("derived", "archive")])
def test_deferred_material_does_not_open_pdf_or_construct_engine(tmp_path, kind, status):
    path = tmp_path / "unread.pdf"
    path.write_bytes(b"not parsed during classification")
    result = import_document(path, kind=kind, engine_factory=forbidden, inspector=forbidden, document_id="doc")
    assert result["status"] == status and not result["imported"]


@pytest.mark.parametrize("kind,route", [("case", "deep"), ("reference", "light")])
def test_mixed_text_and_unreadable_pages_are_never_admitted_as_complete(tmp_path, kind, route):
    path = tmp_path / "mixed.pdf"
    pdf(path, ["Original first page", ""])
    result = import_document(path, kind=kind, engine_factory=forbidden, document_id="doc")
    assert result["status"] == "pending_extraction" and not result["imported"]
    assert result["inspection"]["pages_needing_ocr"] == [2]
    assert result["routing"]["path"] == route


def test_reference_import_keeps_only_source_pages_and_never_generates_summaries(tmp_path):
    from tests.test_long_term_memory import engine
    graph = engine()
    path = tmp_path / "reference.pdf"
    pdf(path, ["Source " + str(i) for i in range(9)])
    original = path.read_bytes()
    result = import_document(path, kind="reference", engine_factory=lambda: graph,
                             summary_fn=forbidden, document_id="doc")
    assert result["imported"] and result["routing"]["path"] == "light"
    assert graph.graph.vcount() == 9 and set(graph.graph.vs["type"]) == {"Raw"}
    metadata = graph.graph.vs[0]["metadata"]
    assert metadata["page_start"] == 1 and metadata["original_source_file"] == str(path.resolve())
    assert metadata["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert path.read_bytes() == original


def test_long_core_document_reuses_existing_tree_builder(tmp_path):
    from tests.test_long_term_memory import engine
    graph = engine()
    path = tmp_path / "case.pdf"
    pdf(path, ["Case original " + str(i) for i in range(9)])
    calls = []
    result = import_document(path, kind="case", engine_factory=lambda: graph, document_id="doc",
                             summary_fn=lambda texts: calls.append(texts) or "navigation summary")
    assert result["routing"]["path"] == "deep" and result["imported"]
    assert "Summary" in graph.graph.vs["type"] and calls


def test_optional_inspector_cannot_override_native_missing_pages(tmp_path):
    path = tmp_path / "mixed.pdf"
    pdf(path, ["Original", ""])
    result = import_document(path, kind="case", engine_factory=forbidden, document_id="doc",
        inspector=lambda path: {"page_count": 2, "backend": "fake", "pages_needing_ocr": [],
                                "complex_layout": False, "encoding_issues": False})
    assert result["status"] == "pending_extraction"
    assert result["inspection"]["pages_needing_ocr"] == [2]


def test_adapter_uses_detect_only_with_one_based_pages(monkeypatch):
    fake = SimpleNamespace(detect_pdf=lambda path: SimpleNamespace(
        page_count=2, pages_needing_ocr=[2], is_complex_layout=False,
        has_encoding_issues=False, pdf_type="mixed"), process_pdf_with_ocr=forbidden)
    monkeypatch.setitem(sys.modules, "pdf_inspector", fake)
    assert inspect_with_pdf_inspector("source.pdf")["pages_needing_ocr"] == [2]
    fake.detect_pdf = lambda path: SimpleNamespace(page_count=2, pages_needing_ocr=[0])
    with pytest.raises(ValueError, match="physical page"):
        inspect_with_pdf_inspector("source.pdf")


def test_cli_report_does_not_create_knowledge_database(tmp_path, capsys):
    from tools.long_term_memory import main
    source, output, memory = tmp_path / "input.jsonl", tmp_path / "out.jsonl", tmp_path / "memory.sqlite"
    source.write_text('{"material_kind":"reference","text":"Original"}\n', encoding="utf-8")
    assert main(["--path", str(memory), "route-corpus", "--input", str(source), "--output", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["paths"] == {"light": 1}
    assert not memory.exists()


def test_cli_pending_pdf_does_not_initialize_encoder_or_database(tmp_path, capsys):
    from tools.long_term_memory import main
    path, memory = tmp_path / "scan.pdf", tmp_path / "memory.sqlite"
    pdf(path, [""])
    assert main(["--path", str(memory), "import-document", "--pdf", str(path),
                 "--document-id", "scan", "--material-kind", "case"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pending_extraction"
    assert not memory.exists()
