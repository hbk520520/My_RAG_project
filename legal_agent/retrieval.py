"""Shared deterministic vector recall and one-hop graph expansion."""
from dataset.graph import LegalDenseGraphBuilder, vname, as_num_id
def _vector_recall(engine: LegalDenseGraphBuilder, query: str, k: int, query_vector=None) -> list:
    """稠密向量召回，返回 [{id, content, score}]（已过滤 tombstone）"""
    vector = engine.encode_text(query)["dense"] if query_vector is None else query_vector
    q_vec = vector.reshape(1, -1).astype("float32")

    k = int(min(max(k, 1), engine.index.ntotal))
    sims, ids = engine.index.search(q_vec, k)

    hits = []
    for score, nid in zip(sims[0], ids[0]):
        if nid == -1:
            continue
        try:
            v = engine.graph.vs.find(name=vname(nid))
        except ValueError:
            continue
        if v["metadata"].get("status") == "tombstone":
            continue
        hits.append({"id": int(nid), "content": v["content"], "score": float(score),
                     "metadata": v["metadata"]})
    return hits


def _graph_expand(engine: LegalDenseGraphBuilder, seeds: list, limit: int) -> list:
    """沿语义边从种子节点扩展一跳，返回新增的 [{id, content, score}]"""
    # 阶段 10：`seen` 与返回的 id 必须是**数值 ID**（外部接口口径），
    # 而 igraph 顶点名是 str(int) —— 混用会让去重判断恒为 False，同一邻居被反复加入。
    seen = {int(h["id"]) for h in seeds}
    expanded = []
    for seed in seeds:
        try:
            v = engine.graph.vs.find(name=vname(seed["id"]))
        except ValueError:
            continue
        for nb in v.neighbors():
            nb_id = as_num_id(nb["name"])
            if nb_id in seen:
                continue
            if nb["metadata"].get("status") == "tombstone":
                continue
            seen.add(nb_id)
            # 邻居得分按一跳衰减，便于 Grader 分辨直接命中与扩展命中
            expanded.append({"id": nb_id, "content": nb["content"],
                             "score": seed["score"] * 0.9, "metadata": nb["metadata"]})
            if len(expanded) >= limit:
                return expanded
    return expanded
