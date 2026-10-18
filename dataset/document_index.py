"""Conservative PDF structure extraction using existing PyMuPDF, without model calls.

Physical pages are 1-based. Bookmarks define the frame; explicit layout headings
may fill its sections. Diagnostics describe rejected structure rather than hiding it.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
import re
from statistics import median

from .page_index_memory import Section


NUMBERED = re.compile(r"^(?:第[一二三四五六七八九十百千万\d]+[编章节]|\d+(?:\.\d+)*[.、\s]|(?:Chapter|Section)\s+\d+)", re.I)


def build_pdf_tree(document, *, title: str) -> dict:
    page_count = len(document)
    if not page_count:
        raise ValueError("PDF has no pages")
    diagnostics = []
    bookmarks = document.get_toc(simple=True)
    previous_page, previous_level = 0, 0
    for level, heading, page in bookmarks:
        if (not str(heading).strip() or not 1 <= page <= page_count or page < previous_page
                or level < 1 or level > previous_level + 1):
            diagnostics.append({"source": "bookmarks", "reason": "invalid title, physical page, order or level"})
            bookmarks = []
            break
        previous_page, previous_level = page, level

    lines, sizes = [], []
    for page_number, page in enumerate(document, 1):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line["spans"]
                text = "".join(span["text"] for span in spans).strip()
                if text:
                    size = max(span["size"] for span in spans)
                    sizes.extend([size] * min(len(text), 100))
                    lines.append((page_number, text, size))
    body_size = median(sizes) if sizes else 0
    repeated = Counter(text for _, text, _ in set(lines))
    headings = []
    for page, text, size in lines:
        if len(text) > 100 or repeated[text] > 1:
            continue
        if NUMBERED.match(text) or (size >= body_size * 1.2 and size > body_size):
            prefix = re.match(r"^(\d+(?:\.\d+)*)", text)
            level = prefix.group(1).count(".") + 1 if prefix else 1
            if re.match(r"^第.+章", text):
                level = 2
            if re.match(r"^第.+节", text):
                level = 3
            headings.append((page, text, level))

    root = {"node_id": "root", "title": title, "start_index": 1, "end_index": page_count, "nodes": []}
    serial = 0
    def insert(entries, parent, source):
        nonlocal serial
        stack = [(0, parent)]
        for page, heading, level in entries:
            while stack[-1][0] >= level:
                stack.pop()
            serial += 1
            node = {"node_id": f"section-{serial}", "title": heading, "start_index": page,
                    "end_index": parent["end_index"], "nodes": [], "toc_source": source}
            stack[-1][1]["nodes"].append(node)
            stack.append((level, node))
        def close(node):
            for i, child in enumerate(node["nodes"]):
                child["end_index"] = (max(child["start_index"], node["nodes"][i+1]["start_index"] - 1)
                                      if i + 1 < len(node["nodes"]) else node["end_index"])
                close(child)
        close(parent)

    if bookmarks:
        insert([(page, str(heading).strip(), level) for level, heading, page in bookmarks], root, "bookmarks")
    bookmark_nodes = []
    def collect(node):
        for child in node["nodes"]:
            bookmark_nodes.append(child)
            collect(child)
    collect(root)
    known = {(node["start_index"], re.sub(r"\s+", "", node["title"]).casefold()) for node in bookmark_nodes}
    groups = {}
    owners = {"root": root, **{node["node_id"]: node for node in bookmark_nodes}}
    for page, text, level in headings:
        if (page, re.sub(r"\s+", "", text).casefold()) in known:
            continue
        candidates = [node for node in bookmark_nodes if node["start_index"] <= page <= node["end_index"]]
        owner = min(candidates, key=lambda node: node["end_index"] - node["start_index"]) if candidates else root
        # Do not reparent an existing bookmark subtree using a weaker layout signal.
        if owner["nodes"]:
            diagnostics.append({"source": "detected", "page": page, "title": text,
                                "reason": "ambiguous coverage of an existing bookmark subtree"})
            continue
        groups.setdefault(owner["node_id"], []).append((page, text, level))
        known.add((page, re.sub(r"\s+", "", text).casefold()))
    for owner_id, entries in groups.items():
        insert(entries, owners[owner_id], "detected")
    toc_source = "hybrid" if bookmarks and groups else "bookmarks" if bookmarks else "detected" if groups else "flat"
    root["toc_source"] = toc_source
    Section.model_validate(root)
    return {"structure": [root], "toc_source": toc_source, "diagnostics": diagnostics}


def read_pdf_structure(path: str | Path) -> tuple[list[str], dict]:
    import pymupdf
    with pymupdf.open(path) as document:
        pages = [page.get_text("text") for page in document]
        tree = build_pdf_tree(document, title=Path(path).stem)
    return pages, tree


def diagnose_cost(tree: dict | list, *, routing_pages: float = 1.0) -> list[dict]:
    """Worst-branch page heuristic only; this is not latency or measured token cost."""
    if routing_pages < 0:
        raise ValueError("Routing cost must be nonnegative")
    roots = [Section.model_validate(node) for node in (tree if isinstance(tree, list) else tree["structure"])]
    result = []
    def visit(node):
        children = [visit(child) for child in node.nodes]
        span = node.end_index - node.start_index + 1
        covered = {page for child in node.nodes for page in range(child.start_index, child.end_index + 1)}
        residual = span - len(covered)
        cost = routing_pages + max([residual, *children]) if children else span
        result.append({"node_id": node.node_id, "flat_pages": span, "tree_pages": cost,
                       "residual_pages": residual, "suggest_merge": bool(children) and span <= cost})
        return cost
    for root in roots:
        visit(root)
    return result
