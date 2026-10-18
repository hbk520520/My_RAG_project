"""Validate citations against evidence actually read, retaining the document version.

Exact quotes and locations are checked. Semantic entailment is a separate evaluation.
"""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    evidence_id: str = Field(pattern=r"^E[1-9][0-9]*$")
    quote: str = Field(min_length=1)


class CitedAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: str = Field(min_length=1)
    citations: list[Citation] = Field(min_length=1)


def evidence_catalog(analyses: list[dict]) -> list[dict]:
    sources = {}
    for analysis in analyses:
        for chunk in analysis["graph"]["source_chunks"]:
            if not chunk["text"].strip():
                continue
            key = chunk["chunk_id"]
            if key in sources and sources[key] != chunk:
                raise ValueError("Conflicting evidence for the same chunk ID")
            sources[key] = chunk
    return [{"evidence_id": f"E{index}", **chunk}
            for index, (_, chunk) in enumerate(sorted(sources.items()), 1)]


def requires_citations(analyses: list[dict]) -> bool:
    from config_loader import cfg
    return bool(cfg.get("pageindex", "enabled", default=False) and any(
        item.get("document_version") for item in evidence_catalog(analyses)))


def generate_cited_answer(messages: list[dict], analyses: list[dict], call) -> dict:
    catalog = evidence_catalog(analyses)
    if not catalog:
        raise ValueError("Cannot generate a cited answer without read evidence")
    instruction = ("Return JSON matching this schema. Put [E1], [E2], etc. directly after the claims they support. "
                   "Use only IDs from the evidence catalog actually read; every cited ID needs an exact nonempty "
                   "contiguous quote from that item. Do not cite navigation summaries. Do not invent locations.\n"
                   + json.dumps(CitedAnswer.model_json_schema(), ensure_ascii=False))
    request = [dict(message) for message in messages]
    request[0]["content"] += "\n\n" + instruction
    request[-1]["content"] += "\n\nRead evidence catalog:\n" + json.dumps(catalog, ensure_ascii=False)
    parsed = CitedAnswer.model_validate_json(call(request))
    by_id = {item["evidence_id"]: item for item in catalog}
    used = list(dict.fromkeys(re.findall(r"\[(E[0-9]+)\]", parsed.answer)))
    quoted = {citation.evidence_id: citation.quote for citation in parsed.citations}
    if not used or len(quoted) != len(parsed.citations) or set(used) != set(quoted) or not set(used) <= set(by_id):
        raise ValueError("Answer cites unknown, duplicate, missing or unused evidence IDs")
    resolved = []
    for index, evidence_id in enumerate(used, 1):
        source = by_id[evidence_id]
        quote = quoted[evidence_id]
        if not quote.strip() or quote not in source["text"]:
            raise ValueError("Citation quote is not present in the read original evidence")
        resolved.append({"index": index, "evidence_id": evidence_id, "quote": quote,
                         **{key: source.get(key) for key in (
                             "record_id", "chunk_id", "document_id", "document_version", "source_file", "page_start", "page_end")}})
    answer = parsed.answer
    for item in resolved:
        answer = answer.replace(f"[{item['evidence_id']}]", f"[{item['index']}]")
    footer = []
    for item in resolved:
        name = item["source_file"] or item["record_id"]
        page = item["page_start"]
        location = (f"，第 {page} 页" if page == item["page_end"] else f"，第 {page}–{item['page_end']} 页") if page else ""
        footer.append(f"[{item['index']}] {name}{location}：{item['quote']}")
    return {"answer": answer + "\n\n来源：\n" + "\n".join(footer), "citations": resolved}
