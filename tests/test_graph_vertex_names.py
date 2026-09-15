"""
顶点名类型契约测试（阶段 10）
============================
背景：`LegalDenseGraphBuilder` 原先把 **int** 节点 ID 直接当 igraph 顶点名用，
触发两件事：

  1. 每次 `add_vertex` 都打 `DeprecationWarning: You are using integers as vertex
     names ... Future versions from igraph 0.11.0 will disallow integers as vertex
     names.` —— 一旦真被禁止，整个图引擎全线报错。
  2. 更隐蔽的：`build_initial_graph_batch` 用 `n["id"]` 当顶点名，**类型由语料决定**。
     若语料给的是字符串 ID，而增量 `add_node()` 用 `_generate_unique_id()` 返回 int，
     同一个图里就会混入两种类型的名字。实测混用后 `vs.find(name=...)` 会抛
     `ValueError` —— 而所有调用点都写成 `except ValueError: continue`，
     于是变成**静默漏节点**。

本模块的约定（由本文件锁定）：
    **图内顶点名恒为 `str(int)`；内部一切逻辑仍用 int ID。**
    边界转换只有 `vname()` / `as_num_id()` 两个函数。
"""
import warnings

import numpy as np
import pytest

from dataset.graph import (
    LegalDenseGraphBuilder,
    StubEncoder,
    as_num_id,
    make_offline_engine,
    vname,
)


def _nodes(*ids, node_type="article"):
    return [{"id": i, "content": f"节点内容 {i}", "type": node_type} for i in ids]


# ============================================================================
# 1. 两个边界转换函数
# ============================================================================
@pytest.mark.parametrize("value", [0, 1, 101, np.int64(7), np.int32(8), "9", " 10 "])
def test_vname_always_returns_str(value):
    assert isinstance(vname(value), str)


@pytest.mark.parametrize("value,expected", [
    (101, 101),
    (np.int64(101), 101),
    ("101", 101),
    (" 101 ", 101),
])
def test_as_num_id_accepts_numeric_forms(value, expected):
    assert as_num_id(value) == expected
    assert isinstance(as_num_id(value), int)


def test_vname_and_as_num_id_roundtrip():
    for nid in (1, 42, 99999):
        assert as_num_id(vname(nid)) == nid


@pytest.mark.parametrize("bad", ["doc-1", "abc", "", "1.5", None])
def test_as_num_id_rejects_non_numeric_loudly(bad):
    """非整数 ID 必须报可定位的错误，而不是等 numpy 抛难懂的信息"""
    with pytest.raises(ValueError) as ei:
        as_num_id(bad)
    assert "不是整数" in str(ei.value)


# ============================================================================
# 2. 图内顶点名恒为字符串
# ============================================================================
def test_build_initial_graph_batch_stores_string_names():
    engine = make_offline_engine()
    engine.build_initial_graph_batch(_nodes(101, 102, 103))

    names = list(engine.graph.vs["name"])
    assert names == ["101", "102", "103"]
    assert all(isinstance(n, str) for n in names)


def test_add_node_stores_string_name():
    engine = make_offline_engine()
    engine.add_node(node_id=7, content="增量节点", node_type="Raw")
    assert engine.graph.vs["name"] == ["7"]


def test_generate_summary_node_stores_string_name():
    engine = make_offline_engine()
    engine.build_initial_graph_batch(_nodes(1, 2))
    sid = engine.generate_summary_node(
        ["a", "b"], llm_generate_fn=lambda texts: "合并摘要", child_ids=[1, 2]
    )
    assert engine.graph.vs.find(name=vname(sid))["type"] == "Summary"
    assert all(isinstance(n, str) for n in engine.graph.vs["name"])


def test_no_integer_vertex_name_warning_anywhere():
    """核心回归：常见操作全程不得出现 igraph 的整数顶点名告警"""
    engine = make_offline_engine()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        engine.build_initial_graph_batch(_nodes(101, 102, 103, 200))
        engine.add_node(node_id=301, content="增量 301", node_type="Raw")
        engine.generate_summary_node(
            ["a", "b"], llm_generate_fn=lambda t: "摘要", child_ids=[101, 102]
        )
        engine.tombstone_node(101)
        engine.mount_to_parent(301)

    offenders = [str(w.message) for w in caught
                 if "integers as vertex names" in str(w.message)]
    assert offenders == [], f"仍在用整数顶点名：{offenders}"


# ============================================================================
# 3. 按数值 ID 查找永远可行（这是所有调用方的前提）
# ============================================================================
def test_every_vertex_is_findable_by_numeric_id():
    engine = make_offline_engine()
    engine.build_initial_graph_batch(_nodes(101, 102, 103))
    engine.add_node(node_id=201, content="增量", node_type="Summary")

    for v in engine.graph.vs:
        nid = as_num_id(v["name"])
        assert engine.graph.vs.find(name=vname(nid)).index == v.index


def test_shared_helper_is_used_consistently():
    """调用方必须用 vname()，不能自己 str()/int() 各写一套"""
    engine = make_offline_engine()
    engine.build_initial_graph_batch(_nodes(101))
    # str() 与 vname() 目前等价，但这里是显式的契约声明点
    assert engine.graph.vs.find(name=vname(101))["content"] == "节点内容 101"
    with pytest.raises(ValueError, match="no such vertex|不存在"):
        engine.graph.vs.find(name=101)          # 整数名再也不能匹配


# ============================================================================
# 4. 语料 ID 混合类型不再制造"半盲图"
# ============================================================================
def test_mixed_int_and_str_corpus_ids_are_normalized():
    """修复前：int 101 + str "102" 会让一半顶点查不到（然后被静默跳过）"""
    engine = make_offline_engine()
    engine.build_initial_graph_batch([
        {"id": 101, "content": "整数 ID 的节点", "type": "article"},
        {"id": "102", "content": "字符串 ID 的节点", "type": "article"},
    ])

    assert engine.graph.vs["name"] == ["101", "102"]
    # 两个都必须能按数值 ID 查到
    for nid in (101, 102):
        assert engine.graph.vs.find(name=vname(nid))["content"]


def test_corpus_with_non_numeric_id_fails_fast():
    engine = make_offline_engine()
    with pytest.raises(ValueError) as ei:
        engine.build_initial_graph_batch(
            [{"id": "doc-1", "content": "x", "type": "article"}]
        )
    assert "不是整数" in str(ei.value)


# ============================================================================
# 5. FAISS 主键与 igraph 顶点名是同一套 ID 的两种表示
# ============================================================================
def test_faiss_ids_match_vertex_names():
    engine = make_offline_engine()
    engine.build_initial_graph_batch(_nodes(101, 102, 103, 104))

    vec = engine.encode_text("任意查询")["dense"].reshape(1, -1).astype("float32")
    _, ids = engine.index.search(vec, engine.index.ntotal)
    faiss_ids = {int(i) for i in ids[0] if i != -1}
    graph_ids = {as_num_id(v["name"]) for v in engine.graph.vs}

    assert faiss_ids == graph_ids == {101, 102, 103, 104}


def test_generate_unique_id_avoids_collision_with_string_names():
    engine = make_offline_engine()
    engine.build_initial_graph_batch(_nodes(101, 102))
    assert engine._generate_unique_id() == 103


# ============================================================================
# 6. metadata 里统一存数值 ID
# ============================================================================
def test_mount_to_parent_stores_numeric_parent_id():
    """parent_id 必须是 int —— 否则 `_propagate_dirty` 沿链上行时会类型错位"""
    engine = make_offline_engine()
    engine.build_initial_graph_batch([
        {"id": 1, "content": "Summary 节点：劳动法", "type": "Summary"},
        {"id": 2, "content": "用人单位单方解除需法定理由", "type": "article"},
    ])
    engine.connect_thresh = -1.0          # 放宽阈值，确保必然挂上

    engine.mount_to_parent(2)
    parent_id = engine.graph.vs.find(name=vname(2))["metadata"]["parent_id"]

    assert isinstance(parent_id, int), f"parent_id 是 {type(parent_id).__name__}"
    assert parent_id == 1
    # 顶点名是字符串，但 parent_id 是数值 ID —— 两者都能用过 vname 查回来
    assert engine.graph.vs.find(name=vname(parent_id))["type"] == "Summary"


def test_count_active_children_works_with_string_names():
    engine = make_offline_engine()
    engine.build_initial_graph_batch([
        {"id": 1, "content": "Summary 节点", "type": "Summary"},
        {"id": 2, "content": "子节点 A", "type": "article"},
        {"id": 3, "content": "子节点 B", "type": "article"},
    ])
    engine.connect_thresh = -1.0
    engine.mount_to_parent(2)
    engine.mount_to_parent(3)

    # 父节点自身不计入，两个子节点应被统计到
    assert engine._count_active_children(1) == 2

    engine.tombstone_node(2)
    assert engine._count_active_children(1) == 1


def test_tombstone_and_recalc_still_work_after_name_change():
    """软删除 → 脏传播 → 摘要重算 这条链在字符串顶点名下必须完整可用"""
    engine = make_offline_engine()
    engine.build_initial_graph_batch([
        {"id": 1, "content": "Summary 节点：量刑", "type": "Summary"},
        {"id": 2, "content": "旧法条：故意杀人", "type": "article"},
        {"id": 3, "content": "新法条：故意杀人（含死刑）", "type": "article"},
    ])
    engine.connect_thresh = -1.0
    engine.mount_to_parent(2)
    engine.mount_to_parent(3)

    assert engine.tombstone_node(2) is True
    assert 1 in engine.dirty_summary_ids

    # 注意：2 已被软删，重算只应采纳仍 active 的子节点（3）
    engine.nightly_recalc_summaries(lambda texts: "摘要已更新")

    assert engine.graph.vs.find(name=vname(1))["content"] == "摘要已更新"
    assert engine.graph.vs.find(name=vname(1))["metadata"]["dirty"] is False


def test_summary_skipped_when_all_children_tombstoned():
    """唯一子节点被软删后，摘要不应被重算 —— 否则会把已失效内容再固化一遍"""
    engine = make_offline_engine()
    engine.build_initial_graph_batch([
        {"id": 1, "content": "Summary 节点：量刑", "type": "Summary"},
        {"id": 2, "content": "旧法条：故意杀人", "type": "article"},
    ])
    engine.connect_thresh = -1.0
    engine.mount_to_parent(2)
    engine.tombstone_node(2)

    engine.nightly_recalc_summaries(lambda texts: "不该被写入")
    assert engine.graph.vs.find(name=vname(1))["content"] == "Summary 节点：量刑"
