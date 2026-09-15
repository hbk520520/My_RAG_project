"""
记忆挂载阈值单元测试
====================
覆盖 `dataset/IncrementalMemoryManager.py`：

  - `DynamicThresholder.split()` 的 GMM 正常路径与全部兜底分支
  - 兜底分支必须**永不抛异常**、且**永不返回空的 accept 集合**
    （原实现样本数 < 2 时直接崩，整条增量注入流程被带崩）
  - 孤儿节点新建簇时，父簇 ID 必须用 `_create_new_cluster()` 的返回值，
    而不是 `list(summary_embeddings.keys())[-1]` 猜最后一个

回归背景（阶段 6）：
  1. `GaussianMixture(n_components=2)` 在样本数不足时抛异常，且得分全同
     时两个分量没有意义 —— 两条路都会让 `split()` 崩掉。
  2. `inject_new_knowledge` 原先靠"字典最后一个 key 就是刚建的簇"来定位新簇。
     当新簇没有 embedding 时 `_create_new_cluster` **不会**把它写进摘要索引，
     于是"最后一个 key"仍是旧簇，新叶子被挂到错误的父节点上（静默错挂）。
"""
import numpy as np
import pytest

from dataset.IncrementalMemoryManager import (
    DynamicThresholder,
    IncrementalMemoryManager,
    MockGraphDB,
)


# ============================================================================
# 1. 空输入
# ============================================================================
def test_empty_scores_return_nothing():
    assert DynamicThresholder.split(np.array([])) == ([], 0.0)


def test_empty_python_list_is_accepted():
    assert DynamicThresholder.split([]) == ([], 0.0)


# ============================================================================
# 2. 兜底分支：样本太少
# ============================================================================
def test_min_samples_constant_is_four():
    assert DynamicThresholder.MIN_SAMPLES == 4


@pytest.mark.parametrize("scores", [
    [0.9],
    [0.1, 0.9],
    [0.1, 0.5, 0.9],
])
def test_small_sample_uses_median_fallback(scores):
    """样本数 < MIN_SAMPLES 时必须走中位数兜底，不得抛异常"""
    arr = np.asarray(scores, dtype=float)
    accept, threshold = DynamicThresholder.split(arr)
    assert threshold == pytest.approx(float(np.median(arr)))
    assert accept == [i for i, s in enumerate(arr) if s >= threshold]
    assert accept, "兜底路径也不能返回空 accept"


# ============================================================================
# 3. 兜底分支：得分几乎相同
# ============================================================================
@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_identical_scores_use_median_fallback(value):
    arr = np.full(6, value, dtype=float)
    accept, threshold = DynamicThresholder.split(arr)
    assert threshold == pytest.approx(value)
    assert accept == list(range(6))
    assert accept, "全同得分下 GMM 无意义，但也必须给出 accept"


def test_near_identical_scores_use_median_fallback():
    arr = np.array([0.5, 0.5, 0.5 + 1e-9, 0.5])
    accept, _ = DynamicThresholder.split(arr)
    assert accept == [0, 1, 2, 3]


# ============================================================================
# 4. GMM 正常路径
# ============================================================================
def test_bimodal_scores_are_split_by_gmm():
    arr = np.array([0.10, 0.12, 0.90, 0.92])
    accept, threshold = DynamicThresholder.split(arr)
    assert accept == [2, 3], "高相似度簇应被接受"
    assert threshold == pytest.approx(0.90)
    # 关键区分：GMM 的边界是 0.90，而不是中位数兜底的 0.51
    assert threshold > float(np.median(arr))


def test_threshold_always_within_score_range():
    """阈值必须落在 [min, max] 区间内（可以是两数中位，不必等于某个原始得分）"""
    for scores in ([0.1], [0.2, 0.8], [0.1, 0.2, 0.8, 0.9],
                   [0.3, 0.31, 0.32, 0.9]):
        arr = np.asarray(scores, dtype=float)
        accept, threshold = DynamicThresholder.split(arr)
        assert arr.min() <= threshold <= arr.max(), f"{scores} -> {threshold}"
        # 阈值必须与 accept 集合自洽：accept 就是所有 >= 阈值的下标
        assert accept == [i for i, s in enumerate(arr) if s >= threshold]


def test_accept_indices_are_valid_and_non_empty():
    for scores in ([0.1], [0.2, 0.8], [0.1, 0.2, 0.8, 0.9], [0.5, 0.5, 0.5, 0.5]):
        arr = np.asarray(scores, dtype=float)
        accept, _ = DynamicThresholder.split(arr)
        assert accept, f"{scores} 得到了空 accept"
        assert all(0 <= i < arr.size for i in accept)
        assert len(set(accept)) == len(accept), "索引不应重复"


def test_two_dimensional_input_is_raveled():
    """上游可能传 (n,1) 的列向量，必须自动展平后再走 GMM"""
    accept, threshold = DynamicThresholder.split(
        np.array([[0.10], [0.12], [0.90], [0.92]])
    )
    assert accept == [2, 3]
    assert threshold == pytest.approx(0.90)


def test_column_vector_with_too_few_samples_uses_fallback():
    accept, threshold = DynamicThresholder.split(np.array([[0.1], [0.9]]))
    assert accept == [1]
    assert threshold == pytest.approx(0.5)


def test_float32_input_does_not_crash_gmm():
    arr = np.array([0.1, 0.12, 0.9, 0.92], dtype=np.float32)
    accept, _ = DynamicThresholder.split(arr)
    assert accept == [2, 3]


# ============================================================================
# 5. _fallback_split 直接测试
# ============================================================================
def test_fallback_split_uses_median_inclusive():
    arr = np.array([0.1, 0.2, 0.3, 0.4])
    accept, threshold = DynamicThresholder._fallback_split(arr)
    assert threshold == pytest.approx(0.25)
    assert accept == [2, 3]   # >= 中位数


# ============================================================================
# 6. IncrementalMemoryManager：阈值可注入
# ============================================================================
def test_floor_threshold_can_be_injected():
    mgr = IncrementalMemoryManager(MockGraphDB(), {}, floor_threshold=0.75)
    assert mgr.ABSOLUTE_FLOOR_THRESHOLD == pytest.approx(0.75)


def test_floor_threshold_injection_does_not_pollute_class_constant():
    before = IncrementalMemoryManager.ABSOLUTE_FLOOR_THRESHOLD
    IncrementalMemoryManager(MockGraphDB(), {}, floor_threshold=0.99)
    assert IncrementalMemoryManager.ABSOLUTE_FLOOR_THRESHOLD == before


def test_default_floor_threshold_keeps_class_constant():
    mgr = IncrementalMemoryManager(MockGraphDB(), {})
    assert mgr.ABSOLUTE_FLOOR_THRESHOLD == IncrementalMemoryManager.ABSOLUTE_FLOOR_THRESHOLD


# ============================================================================
# 7. 挂载行为
# ============================================================================
def _db_with_two_clusters():
    db = MockGraphDB()
    for cid in (1, 2):
        db.add_node({"type": "summary", "content": f"簇{cid}", "embedding": None})
    return db


def test_first_node_creates_root_cluster_and_mounts_leaf():
    """回归测试（阶段 8）：第一个节点建根簇时必须**同时连线**

    原实现只调 `_create_new_cluster()` 而没有 `_mount_leaf_to_cluster()`，
    于是第一个叶子在树里没有任何父边、根簇的 `get_children()` 恒为空 ——
    `nightly_rewrite` / `memory_graph_bridge.sync_dirty_summaries` 永远看不到它。
    孤儿分支一直是"建簇 + 挂载"两步都做，两处行为应一致。
    """
    db = MockGraphDB()
    mgr = IncrementalMemoryManager(db, {})
    leaf = mgr.inject_new_knowledge("第一份知识", np.array([1.0, 0.0]))

    assert db.nodes[leaf]["type"] == "leaf"
    root_cluster = next(cid for cid, props in db.nodes.items() if props["type"] == "summary")
    assert db.get_parents(leaf) == [root_cluster], "首个叶子没有挂到根簇上"
    assert db.get_children(root_cluster) == [leaf]
    assert db.is_dirty(root_cluster) is True, "新挂载后根簇应标记为脏以触发摘要重算"
    assert mgr.summary_embeddings.get(root_cluster) is not None


def test_similar_node_mounts_to_matching_cluster():
    db = _db_with_two_clusters()
    embeddings = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
    mgr = IncrementalMemoryManager(db, embeddings, floor_threshold=0.0)

    leaf = db.add_node({"type": "leaf", "content": "劳动关系认定"})
    mgr.inject_new_knowledge("劳动关系认定", np.array([1.0, 0.0]), node_id=leaf)

    assert db.get_parents(leaf) == [1], "与簇1完全同向，应挂到簇1"
    assert db.is_dirty(1) is True, "挂载后簇应被标记为脏"


def test_orphan_node_creates_new_cluster_using_returned_id():
    """回归测试（阶段 6）：新簇 ID 必须取自返回值，而不是"字典最后一个 key"

    构造：新叶子**没有 embedding**，于是 `_create_new_cluster` 不会把新簇写进
    `summary_embeddings`。"最后一个 key" 于是仍指向旧簇 2 —— 旧实现会把叶子
    错挂到簇 2 上（静默错挂）；修复后应挂到真正新建的簇上。
    """
    db = _db_with_two_clusters()
    embeddings = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}
    mgr = IncrementalMemoryManager(db, embeddings, floor_threshold=0.9)

    leaf = db.add_node({"type": "leaf", "content": "全新领域知识"})  # 故意不带 embedding
    mgr.inject_new_knowledge("全新领域知识", np.array([0.0, 0.0]), node_id=leaf)

    parents = db.get_parents(leaf)
    assert len(parents) == 1, "孤儿节点应恰好产生一个新父簇"
    new_cluster_id = parents[0]
    assert new_cluster_id not in (1, 2), "挂到了旧簇上 —— 又用'最后一个 key'猜 ID 了"
    assert db.nodes[new_cluster_id]["type"] == "summary"
    assert db.get_children(new_cluster_id) == [leaf]


def test_orphan_detection_respects_floor_threshold():
    """相似度虽被 accept，但低于绝对保底 -> 仍应新建簇而不是误挂"""
    db = _db_with_two_clusters()
    embeddings = {1: np.array([1.0, 0.0]), 2: np.array([0.0, 1.0])}

    # 归一化向量点积 = cos，对角线方向与两个簇的相似度都约 0.707
    v = np.array([0.7071, 0.7071])
    strict = IncrementalMemoryManager(db, embeddings, floor_threshold=0.95)
    leaf = db.add_node({"type": "leaf", "content": "斜向知识"})
    strict.inject_new_knowledge("斜向知识", v, node_id=leaf)
    assert len(db.get_parents(leaf)) == 1
    assert db.get_parents(leaf)[0] not in (1, 2), "0.707 < 0.95，应判为孤儿"

    # 放宽保底后，同样的向量应挂到已有簇
    db2 = _db_with_two_clusters()
    loose = IncrementalMemoryManager(db2, embeddings, floor_threshold=0.0)
    leaf2 = db2.add_node({"type": "leaf", "content": "斜向知识"})
    loose.inject_new_knowledge("斜向知识", v, node_id=leaf2)
    assert db2.get_parents(leaf2) == [1, 2], "放宽保底后应挂到两个簇"


def test_inject_returns_the_leaf_node_id():
    db = _db_with_two_clusters()
    mgr = IncrementalMemoryManager(db, {1: np.array([1.0, 0.0])}, floor_threshold=0.0)
    leaf = db.add_node({"type": "leaf", "content": "内容"})
    assert mgr.inject_new_knowledge("内容", np.array([1.0, 0.0]), node_id=leaf) == leaf
