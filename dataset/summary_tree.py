"""Dependency order and section inputs shared by import and incremental rewrites."""
from __future__ import annotations


def postorder(children: dict[int, list[int]], dirty) -> list[int]:
    selected = set(dirty) & children.keys()
    while True:
        parents = {parent for parent, descendants in children.items() if selected.intersection(descendants)}
        expanded = selected | parents
        if expanded == selected:
            break
        selected = expanded
    result, visiting, done = [], set(), set()
    def visit(node):
        if node in visiting:
            raise ValueError("Cycle in summary dependencies")
        if node in done:
            return
        visiting.add(node)
        for child in sorted(set(children[node]) & selected):
            visit(child)
        visiting.remove(node)
        done.add(node)
        result.append(node)
    for node in sorted(selected):
        visit(node)
    return result


def render_section(title: str, summary: str, key_items: list[str]) -> str:
    return "\n".join(part for part in [title, summary, *key_items] if part)


def summary_dependencies(graph) -> dict[int, list[int]]:
    summaries = {int(v["name"]): v for v in graph.vs if v["type"] == "Summary"
                 and v["metadata"].get("status") != "tombstone"}
    return {node: [child for child, vertex in summaries.items()
                   if vertex["metadata"].get("parent_id") == node] for node in summaries}


def section_inputs(graph, section_id: int) -> list[str]:
    section = graph.vs.find(name=str(section_id))
    children = [v for v in graph.vs if v["type"] == "Summary"
                and v["metadata"].get("parent_id") == section_id
                and v["metadata"].get("status") != "tombstone"]
    children.sort(key=lambda v: (v["metadata"]["page_start"], v["name"]))
    covered, texts = set(), []
    active = {int(v["name"]): v for v in graph.vs if v["metadata"].get("status") != "tombstone"}
    for child in children:
        sources = set(child["metadata"]["source_node_ids"])
        covered.update(sources)
        if any(source in active and active[source]["content"].strip() for source in sources):
            texts.append(child["content"])
    for source in section["metadata"]["source_node_ids"]:
        if source not in covered and source in active and active[source]["content"].strip():
            texts.append(active[source]["content"])
    return texts
