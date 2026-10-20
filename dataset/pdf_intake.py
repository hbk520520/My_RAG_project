"""Native page extraction and gated import; no OCR runtime or LLM is started here."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .material_routing import MaterialFacts, route_material


def inspect_with_pdf_inspector(path: str | Path) -> dict:
    """Optional detect-only API; PdfResult uses 1-based physical page numbers."""
    import pdf_inspector
    result = pdf_inspector.detect_pdf(str(path))
    pages = list(result.pages_needing_ocr)
    if any(type(page) is not int or not 1 <= page <= result.page_count for page in pages):
        raise ValueError("pdf-inspector returned invalid physical page numbers")
    return {"backend": "pdf-inspector", "page_count": result.page_count,
            "pages_needing_ocr": pages, "complex_layout": result.is_complex_layout,
            "encoding_issues": result.has_encoding_issues, "pdf_type": result.pdf_type}


def import_document(path: str | Path, *, kind: str = "auto", engine_factory,
                    inspector=None, summary_fn=None, document_id: str) -> dict:
    """Use original-page indexing for light material; build hierarchy only for deep.

    Pending/excluded material never constructs a graph or loads an encoder.
    The caller must retain the original PDF; this function never moves it.
    """
    if not document_id.strip():
        raise ValueError("document_id must not be empty")
    initial = route_material(MaterialFacts(kind=kind))
    path = Path(path).resolve()
    # Hash incrementally; do not retain another complete PDF buffer.
    with path.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    base = {"document_id": document_id, "source_file": str(path), "source_sha256": checksum}
    if initial["path"] in ("archive", "deferred"):
        return {**base, "routing": initial, "status": initial["path"], "imported": False}

    # Keep only compact inspection metadata before opening the native document;
    # do not hold two parsers' document objects at the same time.
    external = inspector(path) if inspector is not None else None
    import pymupdf
    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise ValueError("Encrypted PDF requires explicit decryption before intake")
        if not len(document):
            raise ValueError("PDF has no pages")
        # Existing extraction contract: physical page order, no rasterization.
        pages = [page.get_text("text") for page in document]
        unreadable = [index for index, text in enumerate(pages, 1)
                      if not text.strip() or "\ufffd" in text]
        inspection = {"backend": "pymupdf", "page_count": len(pages),
                      "pages_needing_ocr": unreadable, "complex_layout": False,
                      "encoding_issues": bool(any("\ufffd" in text for text in pages))}
        if external is not None:
            if external["page_count"] != len(pages):
                raise ValueError("PDF inspectors disagree on physical page count")
            inspection = {**external, "pages_needing_ocr": sorted(set(unreadable) | set(external["pages_needing_ocr"]))}
        if inspection["encoding_issues"] and not inspection["pages_needing_ocr"]:
            # A global font warning without page attribution cannot certify completeness.
            inspection["pages_needing_ocr"] = list(range(1, len(pages) + 1))
        decision = route_material(MaterialFacts(kind=kind, page_count=len(pages),
                                  needs_ocr=bool(inspection["pages_needing_ocr"]),
                                  complex_layout=inspection["complex_layout"]))
        if inspection["pages_needing_ocr"] or (decision["path"] == "deep" and inspection["complex_layout"]):
            return {**base, "routing": decision, "inspection": inspection,
                    "status": "pending_extraction", "imported": False}
        if decision["path"] == "deep":
            from .document_index import build_pdf_tree
            tree = build_pdf_tree(document, title=path.stem)
        else:
            tree = {"structure": [], "toc_source": "flat"}

    from .page_index_memory import import_tree
    engine = engine_factory()
    imported = import_tree(engine, pages, tree, document_id=document_id, source_file=path.name,
                           metadata={"original_source_file": str(path), "source_sha256": checksum},
                           summary_fn=summary_fn if decision["path"] == "deep" else None)
    return {**base, **imported, "routing": decision, "inspection": inspection,
            "status": "imported", "imported": True}
