"""
记忆 ↔ 图引擎桥接集成测试
==========================
覆盖 `dataset/memory_graph_bridge.py`：GMM 增量挂载 + 图引擎双写。

阶段 8：这个模块原先的 `__main__` 是**唯一**使用它的地方，而且那个 demo 直接
`LegalDenseGraphBuilder(alpha_dense=0.3)` 构造引擎，`add_node` 里会懒加载 BGE-M3
→ 无网环境下卡在 huggingface.co 重试到超时，示例根本跑不完。现在 demo 与测试
统一走 `graph.make_offline_engine()`（注入 `StubEncoder`），整条链路**完全离线**可跑。

阶段 9：把"同一个节点必须只有一个向量"从**约定**变成**结构上不可能违反** ——
`add_node()` 新增 `dense`/`sparse` 参数，桥接器把自己算好的向量透传下去，
不再让图引擎背着 GMM 重新编码一遍。
"""
import logging

import numpy as np
import pytest

from dataset.graph import (
    LegalDenseGraphBuilder, StubEncoder, as_num_id, make_offline_engine, vname,
)
from dataset.IncrementalMemoryManager import IncrementalMemoryManager, MockGraphDB
from dataset.memory_graph_bridge import MemoryGraphBridge


# ============================================================================
# 1. 离线编码器
# ============================================================================
def test_stub_encoder_matches_bgem3_output_shape():
    enc = StubEncoder(dim=64)
    out = enc.encode(["甲", "乙", "丙"])
    assert set(out) == {"dense_vecs", "lexical_weights"}
    assert out["dense_vecs"].shape == (3, 64)
    assert out["dense_vecs"].dtype == np.float32
    assert len(out["lexical_weights"]) == 3


def test_stub_encoder_vectors_are_normalized():
    vecs = StubEncoder(dim=32).encode(["a", "b"])["dense_vecs"]
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)


def test_stub_encoder_is_reproducible():
    a = StubEncoder(dim=16, seed=7).encode(["x"])["dense_vecs"]
    b = StubEncoder(dim=16, seed=7).encode(["x"])["dense_vecs"]
    assert np.array_equal(a, b)


def test_stub_encoder_is_deterministic_per_text():
    """同一文本必须永远得到同一向量（两次编码不能漂移）"""
    enc = StubEncoder(dim=16)
    a = enc.encode(["同一个句子"])["dense_vecs"]
    b = enc.encode(["同一个句子"])["dense_vecs"]
    assert np.array_equal(a, b)

    c = StubEncoder(dim=16).encode(["同一个句子"])["dense_vecs"]
    assert np.array_equal(a, c), "不同实例也必须一致"


def test_stub_encoder_gives_different_vectors_for_different_texts():
    vecs = StubEncoder(dim=64).encode(["甲法条", "乙法条"])["dense_vecs"]
    assert not np.allclose(vecs[0], vecs[1])


def test_stub_encoder_handles_empty_batch():
    out = StubEncoder(dim=8).encode([])
    assert out["dense_vecs"].shape == (0, 8)
    assert out["lexical_weights"] == []


def test_make_offline_engine_never_loads_bge_m3():
    """关键点：构造后不应触发模型下载（能秒回就说明没下载）"""
    engine = make_offline_engine()
    assert engine._encoder is not None, "未注入编码器 -> 下次 encode 会去下载 BGE-M3"
    assert isinstance(engine.encoder, StubEncoder)
    assert engine.dim == engine.encoder.dim


def test_make_offline_engine_uses_config_dimension():
    from config_loader import cfg
    engine = make_offline_engine()
    assert engine.dim == cfg.get("graph", "embedding_dim")


# ============================================================================
# 2. 双写：GMM + 图引擎
# ============================================================================
@pytest.fixture
def bridge():
    db = MockGraphDB()
    gmm = IncrementalMemoryManager(db, {})
    engine = make_offline_engine()
    rng = np.random.RandomState(0)

    def embed(text):
        v = rng.randn(engine.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    return MemoryGraphBridge(memory_manager=gmm, graph_engine=engine, embedding_fn=embed), db, engine


def test_inject_knowledge_writes_to_both_subsystems(bridge):
    b, db, engine = bridge
    node_id = b.inject_knowledge("试用期辞退需支付赔偿金")

    # GMM 侧
    assert node_id in db.nodes
    assert db.nodes[node_id]["type"] == "leaf"
    assert b.mm.get_node_text(node_id) == "试用期辞退需支付赔偿金"
    assert len(b.mm.summary_embeddings) == 1

    # 图引擎侧
    assert engine.graph.vcount() == 1
    assert engine.index.ntotal == 1
    assert engine.graph.vs.find(name=vname(node_id))["content"] == "试用期辞退需支付赔偿金"


def test_bridge_reuses_same_id_across_subsystems(bridge):
    b, db, engine = bridge
    first = b.inject_knowledge("第一条")
    second = b.inject_knowledge("第二条")

    # 两个子系统必须用同一套 ID（桥接的意义就在于此）
    assert first != second
    assert {first, second} <= set(db.nodes)
    assert {as_num_id(v["name"]) for v in engine.graph.vs} == {first, second}

    # GMM 侧除了两个叶子，还应有簇节点（数量取决于是否判为孤儿，故不写死）
    leaves = {nid for nid, p in db.nodes.items() if p["type"] == "leaf"}
    clusters = {nid for nid, p in db.nodes.items() if p["type"] == "summary"}
    assert leaves == {first, second}
    assert clusters and not (leaves & clusters)


def test_dirty_summary_is_produced_by_leaf_mount(bridge):
    """回归：首个叶子必须挂到根簇上并标脏，否则摘要永远不会被重算"""
    b, db, engine = bridge
    b.inject_knowledge("第一条知识")
    dirty = db.get_dirty_nodes()
    assert dirty, "没有任何脏节点 -> nightly_rewrite / sync_dirty_summaries 不会触发"
    root_cluster = next(nid for nid, props in db.nodes.items() if props["type"] == "summary")
    assert root_cluster in dirty
    assert db.get_children(root_cluster)


def test_sync_dirty_summaries_updates_mock_db(bridge):
    b, db, engine = bridge
    b.inject_knowledge("第一条知识")
    b.sync_dirty_summaries(summary_fn=lambda texts: "合并摘要: " + " / ".join(texts))

    root_cluster = next(nid for nid, props in db.nodes.items() if props["type"] == "summary")
    assert db.nodes[root_cluster]["content"].startswith("合并摘要:")
    assert db.is_dirty(root_cluster) is False, "同步后应清掉脏标记"


def test_sync_dirty_summaries_is_noop_when_nothing_dirty(bridge):
    b, db, engine = bridge
    b.sync_dirty_summaries(summary_fn=lambda texts: "不该被调用")


def test_stats_reflect_both_subsystems(bridge):
    b, db, engine = bridge
    b.inject_knowledge("第一条知识")
    stats = b.get_stats()
    assert stats["graph_nodes"] == 1
    assert stats["faiss_total"] == 1
    assert stats["gmm_nodes"] == 2
    assert stats["pending_sync"] == 0


def test_pending_sync_captures_graph_engine_failure(bridge):
    """图引擎写入失败时应进 pending_sync，而不是把整条注入链路打断"""
    b, db, engine = bridge

    def _boom(**kwargs):
        raise RuntimeError("faiss 索引损坏")

    engine.add_node = _boom
    node_id = b.inject_knowledge("会失败的写入")

    assert node_id in db.nodes, "GMM 侧仍应写入成功"
    assert len(b._pending_sync) == 1
    assert b._pending_sync[0]["node_id"] == node_id


# ============================================================================
# 3. 向量一致性（阶段 9：从"靠约定"变成"结构上不可能分叉"）
# ============================================================================
def test_default_embedding_fn_keeps_both_subsystems_identical():
    """不传 embedding_fn（推荐用法）时，两边必然是同一个向量"""
    db = MockGraphDB()
    engine = make_offline_engine()
    bridge = MemoryGraphBridge(
        memory_manager=IncrementalMemoryManager(db, {}),
        graph_engine=engine,
    )

    node_id = bridge.inject_knowledge("一致向量")
    assert np.allclose(db.nodes[node_id]["embedding"],
                       engine.graph.vs.find(name=vname(node_id))["dense"])


def test_custom_embedding_fn_is_passed_through_to_both_subsystems():
    """阶段 9 核心修复：自定义 embedding_fn 的向量会被**透传**给图引擎。

    修复前：embedding_fn 只喂 GMM，图引擎由 add_node 自己重新编码 →
            两边静默分叉成两个不同空间的向量（无任何报错）。
    修复后：桥接器把同一次编码的 dense/sparse 一并交给 add_node。
    """
    db = MockGraphDB()
    engine = make_offline_engine()
    custom = np.ones(engine.dim, dtype=np.float32) / np.sqrt(engine.dim)
    bridge = MemoryGraphBridge(
        memory_manager=IncrementalMemoryManager(db, {}),
        graph_engine=engine,
        embedding_fn=lambda t: custom,
    )

    node_id = bridge.inject_knowledge("自定义向量")
    gmm_side = db.nodes[node_id]["embedding"]
    graph_side = engine.graph.vs.find(name=vname(node_id))["dense"]

    assert np.allclose(graph_side, custom), "图引擎没有用透传进来的向量"
    assert np.allclose(gmm_side, graph_side), "两边仍然分叉"


def test_bridge_does_not_double_encode():
    """同一条知识只应编码一次（修复前 GMM + 图引擎各编一次）"""
    db = MockGraphDB()
    engine = make_offline_engine()
    calls = {"n": 0}
    original = engine.encode_text

    def counting(text):
        calls["n"] += 1
        return original(text)

    engine.encode_text = counting
    bridge = MemoryGraphBridge(
        memory_manager=IncrementalMemoryManager(db, {}),
        graph_engine=engine,
    )
    bridge.inject_knowledge("只编码一次的文本")
    assert calls["n"] == 1, f"编码了 {calls['n']} 次"


def test_embedding_fn_returning_bge_m3_shape_is_accepted():
    """兼容 BGE-M3 原生返回 {"dense_vecs": …, "lexical_weights": …}"""
    db = MockGraphDB()
    engine = make_offline_engine()
    stub = StubEncoder(dim=engine.dim)
    bridge = MemoryGraphBridge(
        memory_manager=IncrementalMemoryManager(db, {}),
        graph_engine=engine,
        embedding_fn=lambda t: stub.encode([t]),
    )

    node_id = bridge.inject_knowledge("原生形态")
    expected = stub.encode(["原生形态"])["dense_vecs"][0]
    assert np.allclose(db.nodes[node_id]["embedding"], expected)
    assert np.allclose(engine.graph.vs.find(name=vname(node_id))["dense"], expected)


def test_embedding_fn_returning_dict_with_sparse_uses_both_signals():
    """形态 2：{"dense":…, "sparse":…} 应当两个信号都被采纳"""
    db = MockGraphDB()
    engine = make_offline_engine()
    dense = np.ones(engine.dim, dtype=np.float32)
    bridge = MemoryGraphBridge(
        memory_manager=IncrementalMemoryManager(db, {}),
        graph_engine=engine,
        embedding_fn=lambda t: {"dense": dense, "sparse": {"工资": 0.9}},
    )

    node_id = bridge.inject_knowledge("带稀疏权重的文本")
    assert engine.graph.vs.find(name=vname(node_id))["sparse"] == {"工资": 0.9}


def test_bare_vector_embedding_fn_warns_about_missing_sparse(caplog):
    """形态 3（只有 dense）不再静默：必须打 WARNING 说明稀疏信号丢失"""
    db = MockGraphDB()
    engine = make_offline_engine()
    bridge = MemoryGraphBridge(
        memory_manager=IncrementalMemoryManager(db, {}),
        graph_engine=engine,
        embedding_fn=lambda t: np.ones(engine.dim, dtype=np.float32),
    )

    with caplog.at_level(logging.WARNING, logger="MemoryGraphBridge"):
        node_id = bridge.inject_knowledge("只有稠密向量")

    assert "稀疏权重" in caplog.text
    assert engine.graph.vs.find(name=vname(node_id))["sparse"] == {}
    # 只警告一次，不能刷屏
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="MemoryGraphBridge"):
        bridge.inject_knowledge("第二条")
    assert caplog.text == ""


def test_embedding_fn_dimension_mismatch_is_rejected():
    """自定义 embedding_fn 维度不符时立刻报错，而不是写出错误的图"""
    db = MockGraphDB()
    engine = make_offline_engine()
    bridge = MemoryGraphBridge(
        memory_manager=IncrementalMemoryManager(db, {}),
        graph_engine=engine,
        embedding_fn=lambda t: np.ones(engine.dim // 2, dtype=np.float32),
    )

    with pytest.raises(ValueError) as ei:
        bridge.inject_knowledge("维度不对")
    assert "维度" in str(ei.value)


# ============================================================================
# 4. add_node 的向量透传契约（阶段 9 新增参数）
# ============================================================================
def test_add_node_uses_precomputed_vectors_without_encoding():
    engine = make_offline_engine()
    calls = {"n": 0}

    def _boom(text):
        calls["n"] += 1
        raise AssertionError("给了 dense/sparse 就不该再调用编码器")

    engine.encode_text = _boom
    engine.add_node(node_id=1, content="文本", node_type="Raw",
                    dense=np.ones(engine.dim, dtype=np.float32),
                    sparse={"工资": 0.5})

    assert calls["n"] == 0
    v = engine.graph.vs.find(name=vname(1))
    assert np.allclose(v["dense"], 1.0)
    assert v["sparse"] == {"工资": 0.5}


def test_add_node_encodes_when_vectors_missing():
    engine = make_offline_engine()
    calls = {"n": 0}
    original = engine.encode_text

    def counting(text):
        calls["n"] += 1
        return original(text)

    engine.encode_text = counting
    engine.add_node(node_id=1, content="文本", node_type="Raw")
    assert calls["n"] == 1


def test_add_node_encodes_only_for_the_missing_half():
    """只给 dense 时仍需要编码一次（为了拿 sparse）"""
    engine = make_offline_engine()
    calls = {"n": 0}
    original = engine.encode_text

    def counting(text):
        calls["n"] += 1
        return original(text)

    engine.encode_text = counting
    engine.add_node(node_id=1, content="文本", node_type="Raw",
                    dense=np.ones(engine.dim, dtype=np.float32))
    assert calls["n"] == 1


def test_add_node_rejects_dimension_mismatch():
    engine = make_offline_engine()
    with pytest.raises(ValueError) as ei:
        engine.add_node(node_id=1, content="文本", node_type="Raw",
                        dense=np.ones(engine.dim + 1, dtype=np.float32),
                        sparse={})
    assert "维度" in str(ei.value)


def test_add_node_copies_the_incoming_vector():
    """共享同一个 ndarray 会让一侧的原地修改波及另一侧，必须拷贝"""
    engine = make_offline_engine()
    mine = np.ones(engine.dim, dtype=np.float32)
    engine.add_node(node_id=1, content="文本", node_type="Raw", dense=mine, sparse={})

    mine[0] = 999.0
    assert engine.graph.vs.find(name=vname(1))["dense"][0] == pytest.approx(1.0)


def test_add_node_still_supports_legacy_call_signature():
    """老调用方只传 4 个参数，必须继续可用"""
    engine = make_offline_engine()
    assert engine.add_node(node_id=1, content="文本", node_type="Summary", metadata={}) is True
    assert engine.graph.vcount() == 1


# ============================================================================
# 5. 阶段 9 一起修掉的同类隐患
# ============================================================================
def test_pending_sync_retains_vector_for_later_retry():
    """挂起节点必须把向量一起暂存。

    否则重试时 `add_node` 会自己重新编码 → 写进图引擎的是另一个向量，
    GMM 与图引擎又分叉了（正是阶段 9 要消灭的那个坑）。
    """
    db = MockGraphDB()
    engine = make_offline_engine()
    bridge = MemoryGraphBridge(
        memory_manager=IncrementalMemoryManager(db, {}),
        graph_engine=engine,
    )

    original = engine.add_node

    def _flaky(**kwargs):
        raise RuntimeError("图引擎临时故障")

    engine.add_node = _flaky
    node_id = bridge.inject_knowledge("会挂起的写入")
    engine.add_node = original

    assert len(bridge._pending_sync) == 1
    item = bridge._pending_sync[0]
    assert item["dense"] is not None, "挂起项没有保存向量"
    assert "sparse" in item
    assert np.allclose(item["dense"], db.nodes[node_id]["embedding"])

    bridge.retry_pending_sync()
    assert bridge._pending_sync == []
    assert np.allclose(engine.graph.vs.find(name=vname(node_id))["dense"],
                       db.nodes[node_id]["embedding"]), "重试后两边又分叉了"


def test_sync_dirty_summaries_encodes_only_once():
    db = MockGraphDB()
    engine = make_offline_engine()
    bridge = MemoryGraphBridge(
        memory_manager=IncrementalMemoryManager(db, {}),
        graph_engine=engine,
    )
    bridge.inject_knowledge("知识")

    calls = {"n": 0}
    original = engine.encode_text

    def counting(text):
        calls["n"] += 1
        return original(text)

    engine.encode_text = counting
    bridge.sync_dirty_summaries(summary_fn=lambda texts: "合并摘要")
    assert calls["n"] == 1, f"摘要在同步时被编码了 {calls['n']} 次"


def test_sync_dirty_summaries_creates_summary_vertex_with_vectors():
    """簇节点只存在于 GMM 侧，所以同步时通常走「新建 Summary 顶点」分支"""
    db = MockGraphDB()
    engine = make_offline_engine()
    bridge = MemoryGraphBridge(
        memory_manager=IncrementalMemoryManager(db, {}),
        graph_engine=engine,
    )
    bridge.inject_knowledge("知识")

    root = next(nid for nid, p in db.nodes.items() if p["type"] == "summary")
    assert not any(as_num_id(v["name"]) == root for v in engine.graph.vs), "前置条件：图里还没有这个簇"

    bridge.sync_dirty_summaries(summary_fn=lambda texts: "合并摘要")

    v = engine.graph.vs.find(name=vname(root))
    assert v["type"] == "Summary"
    assert v["content"] == "合并摘要"
    assert v["dense"] is not None
    assert v["sparse"], "新建 Summary 顶点时也应带上稀疏权重"
