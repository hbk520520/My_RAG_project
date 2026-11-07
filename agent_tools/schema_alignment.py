"""PA-Tool naming and reversible model-facing schemas; execution names stay canonical."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re


CATALOGS = {
    "replanner": {
        "selector": "engine",
        "fields": {
            "task_queue": "Ordered list of replacement tasks for the failed subtask.",
            "task_desc": "Specific retrieval query or operation to perform for one replacement task.",
            "engine": "Retrieval tool selected for this task.",
            "rationale": "Reason this retrieval tool addresses the observed failure.",
        },
        "tools": {
            "GRAPH_TRAVERSAL": "Retrieve evidence and expand neighboring clues within the current legal graph.",
            "GLOBAL_DENSE_WORMHOLE": "Search the full evidence vector index across legal domains when local graph retrieval is insufficient.",
        },
    },
    "navigator": {
        "selector": "action",
        "fields": {
            "action": "Document navigation tool to invoke next.",
            "document_id": "Identifier of the document from the supplied catalog.",
            "version": "Exact document version from the supplied catalog.",
            "node_id": "Section identifier for descending into the document structure.",
            "offset": "Zero-based pagination offset for a catalog or section list.",
            "pages": "List of one-based physical page numbers whose original text should be read.",
        },
        "tools": {
            "list_documents": "List available documents, with pagination.",
            "read_structure": "Read one level of document sections, optionally below a section identifier.",
            "read_pages": "Read original text from the specified physical document pages.",
            "vector_search": "Retrieve original evidence through the existing vector search baseline.",
            "finish": "Finish evidence collection using the pages already read.",
        },
    },
}


def schema_for(role):
    if role == "navigator":
        from dataset.document_navigation import Action
        return Action.model_json_schema()
    if role != "replanner":
        raise ValueError(f"No tool schema for role: {role}")
    from agent_runtime.contracts import TaskSpec
    task = TaskSpec.model_json_schema()
    definitions = task.pop("$defs", {})
    task["required"].remove("task_id")  # Existing caller supplies a missing ID.
    return {"type": "object", "properties": {"task_queue": {
        "type": "array", "minItems": 1, "items": task}},
        "required": ["task_queue"], "$defs": definitions}


def model_identity(role):
    from config_loader import cfg
    backend = cfg.get("llm", "backend", default="api")
    if backend == "vllm":
        adapter = cfg.get("vllm", "role_adapters", role)
        return {"backend": backend, "model": cfg.get("vllm", "base_model"),
                "adapter": adapter, "adapter_path": cfg.get("vllm", "lora_modules", adapter) if adapter else None}
    return {"backend": backend, "model": cfg.get("llm", "role_models", role,
                                                 default=cfg.get("llm", "judge_model"))}


def edit_distance(left, right):
    """Exact character-level Levenshtein distance, with one row of storage."""
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = current
    return row[-1]


def rank_candidates(samples, reference, alpha=0.2):
    if not samples or not 0 <= alpha <= 1:
        raise ValueError("Need candidate samples and alpha in [0, 1]")
    for name in [reference, *samples]:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Expected a single identifier, got {name!r}")
    counts = Counter(samples)
    threshold = alpha * max(map(len, samples))
    ranked = [{"name": name, "peakedness": sum(
        count for other, count in counts.items() if edit_distance(name, other) <= threshold) - 1,
        "reference_distance": edit_distance(name, reference)} for name in counts]
    # Stable sample order resolves any remaining ties without changing the paper's criteria.
    return sorted(ranked, key=lambda item: (-item["peakedness"], item["reference_distance"]))


def sample_names(role, call, *, count=32, temperature=0.4):
    """call(messages, *, temperature) must query the SAME model that will use the tools."""
    if count < 1 or temperature <= 0:
        raise ValueError("count and sampling temperature must be positive")
    records = {}
    for kind in ("fields", "tools"):
        for original, description in CATALOGS[role][kind].items():
            messages = [{"role": "system", "content":
                "Name a tool component from its functionality. Return only JSON with a single name field. "
                "Use one concise, familiar snake_case identifier; do not add explanations."},
                {"role": "user", "content": f"Component kind: {kind}\nFunctionality: {description}"}]
            def generate(t):
                return json.loads(call(messages, temperature=t))["name"]
            records[f"{kind}.{original}"] = {
                "reference": generate(0), "samples": [generate(temperature) for _ in range(count)]}
    return records


def build_profile(role, records, *, alpha=0.2):
    catalog = CATALOGS[role]
    expected = {f"{kind}.{name}" for kind in ("fields", "tools") for name in catalog[kind]}
    if set(records) != expected:
        raise ValueError("Candidate records must cover exactly the selected role's components")
    rankings = {key: rank_candidates(value["samples"], value["reference"], alpha)
                for key, value in records.items()}
    selected = {}
    for kind in ("fields", "tools"):
        pending = {name: list(rankings[f"{kind}.{name}"]) for name in catalog[kind]}
        reserved = _canonical_names(role, kind)
        used = set()
        while pending:
            for name, choices in pending.items():
                # Canonical names belonging to other fields remain reserved: existing
                # execution fields must not be confused with model-facing aliases.
                choices[:] = [item for item in choices if item["name"] not in used
                              and (item["name"] == name or item["name"] not in reserved)]
                if not choices:
                    raise ValueError(f"No unique candidate left for {kind}.{name}; collect more samples")
            winner = min(pending, key=lambda name: (
                -pending[name][0]["peakedness"], pending[name][0]["reference_distance"]))
            chosen = pending.pop(winner)[0]["name"]
            selected[f"{kind}.{winner}"] = chosen
            used.add(chosen)
    return {"method": "pa-tool-v1", "role": role, "model": model_identity(role),
            "alpha": alpha, "names": selected, "candidates": records, "rankings": rankings}


def _canonical_names(role, kind):
    names = set(CATALOGS[role][kind])
    if kind == "fields":
        def visit(node):
            if isinstance(node, dict):
                names.update(node.get("properties", {}))
                for value in node.values():
                    visit(value)
            elif isinstance(node, list):
                for value in node:
                    visit(value)
        visit(schema_for(role))
    return names


class SchemaAlignment:
    def __init__(self, profile):
        self.role = profile["role"]
        if profile["method"] != "pa-tool-v1" or profile["model"] != model_identity(self.role):
            raise ValueError("Tool schema profile must match the currently configured role model")
        self.fields = {key: profile["names"][f"fields.{key}"] for key in CATALOGS[self.role]["fields"]}
        self.tools = {key: profile["names"][f"tools.{key}"] for key in CATALOGS[self.role]["tools"]}
        for kind, mapping in (("fields", self.fields), ("tools", self.tools)):
            reserved = _canonical_names(self.role, kind)
            if len(set(mapping.values())) != len(mapping) or any(
                    value != key and value in reserved for key, value in mapping.items()):
                raise ValueError("Tool schema aliases must be unambiguous")
        self.selector = CATALOGS[self.role]["selector"]

    def instructions(self, text):
        names = {**self.fields, **self.tools}
        pattern = r"(?<![A-Za-z0-9_])(" + "|".join(map(re.escape, names)) + r")(?![A-Za-z0-9_])"
        return re.sub(pattern, lambda match: names[match[0]], text)

    def schema(self):
        def visit(node, field=None):
            if isinstance(node, list):
                return [visit(value) for value in node]
            if not isinstance(node, dict):
                return node
            result = {}
            for key, value in node.items():
                if key == "properties":
                    result[key] = {self.fields.get(name, name): visit(child, name) for name, child in value.items()}
                elif key == "required":
                    result[key] = [self.fields.get(name, name) for name in value]
                elif field == self.selector and key == "enum":
                    result[key] = [self.tools[name] for name in value]
                elif field == self.selector and key == "default":
                    result[key] = self.tools[value]
                else:
                    result[key] = visit(value)
            return result
        return visit(schema_for(self.role))

    def decode(self, value):
        fields = {alias: original for original, alias in self.fields.items()}
        tools = {alias: original for original, alias in self.tools.items()}
        def visit(item):
            if isinstance(item, list):
                return [visit(child) for child in item]
            if not isinstance(item, dict):
                return item
            result = {}
            for key, child in item.items():
                original = fields.get(key, key)
                if original in result:
                    raise ValueError(f"Duplicate translated parameter: {original}")
                result[original] = tools.get(child, child) if original == self.selector and isinstance(child, str) else visit(child)
            return result
        return visit(value)


def profile_paths():
    from config_loader import cfg
    return {role: (Path(cfg.config_path).parent / path).resolve()
            for role, path in cfg.get("tool_schema_alignment", "profiles", default={}).items()}


def for_role(role):
    path = profile_paths().get(role)
    if path is None:
        return None
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile["role"] != role:
        raise ValueError("Tool schema profile belongs to a different role")
    return SchemaAlignment(profile)


def profile_hashes():
    return {role: hashlib.sha256(path.read_bytes()).hexdigest() for role, path in profile_paths().items()}
