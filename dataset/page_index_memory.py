"""Import PageIndex-exported trees as navigation, with original PDF pages as evidence.

This is an independent format adapter, not a vendored PageIndex SDK or optimizer.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from .summary_tree import render_section


class Section(BaseModel):
    node_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    start_index: int = Field(ge=1)
    end_index: int = Field(ge=1)
    summary: str = ""
    key_items: list[str] = Field(default_factory=list)
    toc_source: str | None = None
    nodes: list[Section] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_range(self):
        if self.end_index < self.start_index:
            raise ValueError("Section end precedes start")
        if any(c.start_index < self.start_index or c.end_index > self.end_index for c in self.nodes):
            raise ValueError("Child section lies outside its parent's page interval")
        return self


def import_tree(engine, pages: list[str], tree: dict | list, *, document_id: str,
                source_file: str, metadata: dict | None = None, summary_fn=None) -> dict:
    if not document_id.strip() or not pages or not any(page.strip() for page in pages):
        raise ValueError("Require document_id and text-based PDF pages; scanned PDFs need OCR first")
    roots = [Section.model_validate(node) for node in (tree if isinstance(tree, list) else tree["structure"])]
    flattened = []
    def visit(node, parent=None):
        flattened.append((node, parent))
        for child in node.nodes:
            visit(child, node.node_id)
    for root in roots:
        visit(root)
    if len({node.node_id for node, _ in flattened}) != len(flattened):
        raise ValueError("Duplicate PageIndex section identifiers")
    if any(node.end_index > len(pages) for node, _ in flattened):
        raise ValueError("PageIndex interval extends beyond the supplied PDF")
    fingerprint = hashlib.sha256(json.dumps({"pages": pages, "tree": tree, "metadata": metadata},
                                             ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    existing = [v for v in engine.graph.vs if v["metadata"].get("document_id") == document_id]
    if existing:
        if any(v["metadata"].get("document_fingerprint") != fingerprint for v in existing):
            raise ValueError("Document changed; import under an explicit new document version")
        return {"document_id": document_id, "nodes": [int(v["name"]) for v in existing], "reused": True}
    first_id = engine._generate_unique_id()
    source_ids = {page: first_id + page - 1 for page in range(1, len(pages) + 1)}
    section_ids = {node.node_id: first_id + len(source_ids) + i for i, (node, _) in enumerate(flattened)}
    common = {**(metadata or {}), "document_id": document_id,
              "source_file": str(source_file), "source_id": document_id, "document_fingerprint": fingerprint,
              "document_page_count": len(pages),
              "toc_source": tree.get("toc_source", "imported") if isinstance(tree, dict) else "imported"}
    # The complete document is committed once; model/encoding failures roll back local staging.
    with engine.memory_transaction():
        for page, node_id in source_ids.items():
            engine.add_node(node_id, pages[page - 1], "Raw", {
                **common, "record_id": f"{document_id}:page:{page}",
                "page_start": page, "page_end": page, "retrieval_role": "evidence"})
        summaries = {}
        for node, parent in reversed(flattened):
            covered = {i for child in node.nodes for i in range(child.start_index, child.end_index + 1)}
            selected = [summaries[child.node_id] for child in node.nodes
                        if any(pages[i-1].strip() for i in range(child.start_index, child.end_index+1))]
            selected += [pages[i-1] for i in range(node.start_index, node.end_index+1)
                         if i not in covered and pages[i-1].strip()]
            summary = node.summary
            if not summary and summary_fn is not None and selected:
                summary = summary_fn(selected)
                if not summary or not str(summary).strip():
                    raise ValueError("Summary generator returned empty text")
            summaries[node.node_id] = render_section(node.title, summary, node.key_items)
            engine.add_node(section_ids[node.node_id], summaries[node.node_id], "Summary", {
                **common, "record_id": f"{document_id}:section:{node.node_id}",
                "pageindex_node_id": node.node_id, "title": node.title,
                "summary": summary, "key_items": node.key_items, "toc_source": node.toc_source or common["toc_source"],
                "page_start": node.start_index, "page_end": node.end_index,
                "retrieval_role": "navigation", "derived": True,
                "source_node_ids": [source_ids[p] for p in source_ids if node.start_index <= p <= node.end_index],
                "parent_id": section_ids[parent] if parent is not None else None,
            })
        # Parent links are document structure, not a semantic-nearest-cluster guess.
        for page, node_id in source_ids.items():
            candidates = [node for node, _ in flattened if node.start_index <= page <= node.end_index]
            if candidates:
                owner = min(candidates, key=lambda node: node.end_index-node.start_index)
                parent_id = section_ids[owner.node_id]
            else:
                parent_id = None
            engine.graph.vs.find(name=str(node_id))["metadata"]["parent_id"] = parent_id
        for node_id in section_ids.values():
            engine.graph.vs.find(name=str(node_id))["metadata"]["dirty"] = False
            engine.dirty_summary_ids.discard(node_id)
    return {"document_id": document_id, "nodes": list(source_ids.values()) + list(section_ids.values()), "reused": False}


def import_pdf(engine, pdf_path: str | Path, tree: dict | list, **kwargs):
    import pymupdf as fitz
    with fitz.open(pdf_path) as document:
        pages = [page.get_text("text") for page in document]
    return import_tree(engine, pages, tree, source_file=Path(pdf_path).name, **kwargs)
