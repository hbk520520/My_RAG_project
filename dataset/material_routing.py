"""Small rule-engine adapter for processing depth, separate from evidence scores."""
from __future__ import annotations

from functools import lru_cache
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path
import tempfile
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt


MaterialKind = Literal["auto", "normative", "case", "reference", "derived", "unknown"]


class MaterialFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: MaterialKind = "auto"
    derived: StrictBool = False
    law_schema: StrictBool = False
    structured: StrictBool = False
    page_count: StrictInt = Field(default=0, ge=0)
    needs_ocr: StrictBool = False
    complex_layout: StrictBool = False


@lru_cache(maxsize=1)
def _policy():
    import rule_engine
    policy = json.loads(files("dataset").joinpath("material_rules.json").read_text(encoding="utf-8"))
    types = {name: rule_engine.DataType.BOOLEAN for name in (
        "derived", "law_schema", "structured", "needs_ocr", "complex_layout")}
    types.update(kind=rule_engine.DataType.STRING, page_count=rule_engine.DataType.FLOAT)
    context = rule_engine.Context(type_resolver=rule_engine.type_resolver_from_dict(types))
    rules = [(entry, rule_engine.Rule(entry["when"], context=context)) for entry in policy["rules"]]
    return policy["version"], rules


def route_material(facts: MaterialFacts) -> dict:
    version, rules = _policy()
    values = facts.model_dump()
    for entry, rule in rules:
        if rule.matches(values):
            return {"policy_version": version, "rule_id": entry["id"],
                    "category": entry["category"], "path": entry["path"],
                    "reason": entry["reason"]}
    raise ValueError("Material routing policy has no matching rule")


def route_record(record: dict) -> dict:
    """Recognize the existing law corpus schema without reading or rewriting text."""
    if not isinstance(record, dict):
        raise TypeError("Each corpus line must be a JSON object")
    law_schema = all(isinstance(record.get(key), str) and record[key].strip()
                     for key in ("law_name", "article_no", "text"))
    facts = MaterialFacts(kind=record.get("material_kind", "auto"),
                          derived=record.get("derived", False), law_schema=bool(law_schema),
                          structured=bool(law_schema))
    decision = route_material(facts)
    decision["review_flags"] = (["missing_" + key for key in ("source", "valid_from", "valid_to")
                                 if not record.get(key)] if decision["category"] == "normative" else [])
    return decision


def route_corpus(path: str | Path):
    """Stream compact source locators and decisions, never a second corpus copy."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            decision = route_record(record)
            yield {"source_file": str(path), "line": line_number,
                   "record_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                   "routing": decision}


def write_routing_report(source: str | Path, output: str | Path) -> dict:
    """Publish a compact report atomically; the original corpus is never replaced."""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("Routing output must differ from the original corpus")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {}
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                     prefix=".routing-", suffix=".tmp", delete=False) as stream:
        staging = Path(stream.name)
        try:
            for entry in route_corpus(source):
                key = entry["routing"]["path"]
                counts[key] = counts.get(key, 0) + 1
                stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            staging.unlink()
            raise
    try:
        os.replace(staging, output)
    finally:
        if staging.exists():
            staging.unlink()
    return {"output": str(output), "records": sum(counts.values()), "paths": counts}
