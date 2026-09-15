"""
双层蓝图单元测试
================
覆盖 `double_layer_plan.py`：

  - 三种历史格式的解析兼容（双层蓝图 / 扁平 task_queue / strategy_queue）
  - DAG 校验：环路、自环、悬空依赖
  - 拓扑排序与可并行组识别
  - 展平为执行队列
  - `plan_steps_from_raw` 单一提取入口

回归背景（阶段 3）：
  Planner 已输出双层蓝图 P_q={S_q,C_q}，但训练侧的 SemanticDTW 过滤器与数据校验
  仍按旧的扁平 `task_queue` 读取。解析异常被外层 try/except 吃掉，结果是
  **静默筛掉全部样本而不报错**。因此这里对「双层蓝图必须能提取出非空步骤」
  单独设了一条回归测试。
"""
import pytest

from double_layer_plan import (
    DoubleLayerPlan,
    SkeletonGraph,
    parse_double_layer_plan,
    plan_steps_from_raw,
)

# ----------------------------------------------------------------------------
# 公共夹具数据：一个菱形 DAG  1 → {2, 3} → 4
# ----------------------------------------------------------------------------
DIAMOND_NODES = [
    {"id": "1", "abstract": "核实劳动关系", "deps": []},
    {"id": "2", "abstract": "核算工作年限", "deps": ["1"]},
    {"id": "3", "abstract": "核查解除合法性", "deps": ["1"]},
    {"id": "4", "abstract": "计算赔偿总额", "deps": ["2", "3"]},
]
DIAMOND_CONCRETIONS = {
    "1": "核实张三与A公司的劳动关系",
    "2": "核算张三在本单位的工作年限",
    "3": "核查A公司的解除行为是否合法",
    "4": "计算A公司应支付的赔偿总额",
}


def _double(nodes=None, concretions=None) -> dict:
    return {
        "skeleton": {"nodes": nodes if nodes is not None else DIAMOND_NODES},
        "concretion": {"concretions": concretions if concretions is not None else DIAMOND_CONCRETIONS},
    }


def _error_of(fn, *args, **kwargs) -> str:
    """执行 fn 并返回其异常信息；不抛异常则判定测试失败。

    统一用 Exception 捕获，避免绑定到 Pydantic v1/v2 不同的包装行为
    （v2 会把 validator 里的 ValueError 包成 ValidationError）。
    """
    with pytest.raises(Exception) as ei:
        fn(*args, **kwargs)
    return str(ei.value)


# ============================================================================
# 1. 解析兼容性
# ============================================================================
def test_parse_canonical_double_layer():
    plan = parse_double_layer_plan(_double())
    assert isinstance(plan, DoubleLayerPlan)
    assert [n.id for n in plan.skeleton.nodes] == ["1", "2", "3", "4"]
    assert plan.concretion.concretions["4"] == "计算A公司应支付的赔偿总额"


def test_parse_legacy_flat_task_queue_with_strings():
    plan = parse_double_layer_plan({"task_queue": ["查劳动关系", "算年限", "算赔偿"]})
    assert [n.id for n in plan.skeleton.nodes] == ["1", "2", "3"]
    assert all(n.deps == [] for n in plan.skeleton.nodes)
    assert plan.concretion.concretions["2"] == "算年限"


def test_parse_legacy_flat_task_queue_with_dicts():
    plan = parse_double_layer_plan({
        "task_queue": [{"task_desc": "查劳动关系", "engine": "GRAPH_TRAVERSAL"}]
    })
    assert plan.concretion.concretions["1"] == "查劳动关系"


def test_parse_unknown_format_raises():
    msg = _error_of(parse_double_layer_plan, {"foo": 1, "bar": 2})
    assert "无法解析双层蓝图" in msg


# ============================================================================
# 2. DAG 校验
# ============================================================================
def test_cycle_is_rejected():
    nodes = [
        {"id": "a", "abstract": "A", "deps": ["c"]},
        {"id": "b", "abstract": "B", "deps": ["a"]},
        {"id": "c", "abstract": "C", "deps": ["b"]},
    ]
    msg = _error_of(SkeletonGraph, nodes=[{**n} for n in nodes])
    assert "环路" in msg


def test_self_loop_is_rejected():
    msg = _error_of(SkeletonGraph, nodes=[{"id": "1", "abstract": "自依赖", "deps": ["1"]}])
    assert "环路" in msg


def test_dangling_dependency_is_rejected():
    nodes = [{"id": "1", "abstract": "A", "deps": ["99"]}]
    msg = _error_of(SkeletonGraph, nodes=nodes)
    assert "99" in msg and "不存在" in msg


def test_single_node_without_deps_is_valid():
    g = SkeletonGraph(nodes=[{"id": "1", "abstract": "A"}])
    assert g.topological_order() == ["1"]


def test_empty_skeleton_is_valid_and_orders_to_nothing():
    g = SkeletonGraph(nodes=[])
    assert g.topological_order() == []
    assert g.get_parallel_groups() == []


# ============================================================================
# 3. 拓扑排序 / 并行组
# ============================================================================
def test_topological_order_is_dependency_safe():
    order = SkeletonGraph(nodes=DIAMOND_NODES).topological_order()
    assert len(order) == 4
    # 每个节点都必须排在其依赖之后
    pos = {nid: i for i, nid in enumerate(order)}
    for n in DIAMOND_NODES:
        for dep in n["deps"]:
            assert pos[dep] < pos[n["id"]]
    assert order[0] == "1" and order[-1] == "4"


def test_parallel_groups_of_diamond():
    groups = SkeletonGraph(nodes=DIAMOND_NODES).get_parallel_groups()
    assert groups == [["1"], ["2", "3"], ["4"]]


# ============================================================================
# 4. 展平为执行队列
# ============================================================================
def test_flat_queue_default_is_topological():
    plan = parse_double_layer_plan(_double())
    queue = plan.to_flat_task_queue()
    assert [q["task_desc"] for q in queue] == [
        DIAMOND_CONCRETIONS["1"],
        DIAMOND_CONCRETIONS["2"],
        DIAMOND_CONCRETIONS["3"],
        DIAMOND_CONCRETIONS["4"],
    ]


def test_flat_queue_can_follow_declaration_order():
    """respect_deps=False 时按 nodes 声明顺序；此处声明顺序与拓扑序故意不同"""
    nodes = [
        {"id": "2", "abstract": "B", "deps": ["1"]},
        {"id": "3", "abstract": "C", "deps": ["1"]},
        {"id": "1", "abstract": "A", "deps": []},
        {"id": "4", "abstract": "D", "deps": ["2", "3"]},
    ]
    plan = parse_double_layer_plan(_double(nodes=nodes, concretions={
        "1": "A'", "2": "B'", "3": "C'", "4": "D'",
    }))
    assert [q["task_desc"] for q in plan.to_flat_task_queue(respect_deps=False)] == \
        ["B'", "C'", "A'", "D'"]
    assert [q["task_desc"] for q in plan.to_flat_task_queue(respect_deps=True)] == \
        ["A'", "B'", "C'", "D'"]


def test_flat_queue_skips_nodes_without_concretion():
    concretions = {k: v for k, v in DIAMOND_CONCRETIONS.items() if k != "3"}
    plan = parse_double_layer_plan(_double(concretions=concretions))
    queue = plan.to_flat_task_queue()
    assert len(queue) == 3
    assert DIAMOND_CONCRETIONS["3"] not in [q["task_desc"] for q in queue]


def test_flat_queue_entry_shape():
    plan = parse_double_layer_plan(_double())
    for item in plan.to_flat_task_queue():
        assert set(item) == {"task_desc", "engine", "rationale"}
        assert item["engine"] == "GRAPH_TRAVERSAL"
        assert item["task_desc"]
        assert item["rationale"]


def test_flat_queue_rationale_carries_abstract_node():
    plan = parse_double_layer_plan(_double())
    queue = plan.to_flat_task_queue()
    item_4 = next(q for q in queue if q["task_desc"] == DIAMOND_CONCRETIONS["4"])
    assert "计算赔偿总额" in item_4["rationale"]
    assert "4" in item_4["rationale"]


# ============================================================================
# 5. plan_steps_from_raw —— 唯一提取入口
# ============================================================================
def test_plan_steps_from_raw_double_layer_follows_topology():
    steps = plan_steps_from_raw(_double())
    assert steps == [
        DIAMOND_CONCRETIONS["1"],
        DIAMOND_CONCRETIONS["2"],
        DIAMOND_CONCRETIONS["3"],
        DIAMOND_CONCRETIONS["4"],
    ]


def test_plan_steps_from_raw_falls_back_to_abstract():
    """concretion 缺某节点时应回退到 abstract，而不是静默丢步"""
    concretions = {k: v for k, v in DIAMOND_CONCRETIONS.items() if k != "3"}
    steps = plan_steps_from_raw(_double(concretions=concretions))
    assert len(steps) == 4
    assert "核查解除合法性" in steps


def test_plan_steps_from_raw_flat_strings():
    assert plan_steps_from_raw({"task_queue": ["a", "b"]}) == ["a", "b"]


def test_plan_steps_from_raw_flat_dicts():
    assert plan_steps_from_raw(
        {"task_queue": [{"task_desc": "a"}, {"task": "b"}]}
    ) == ["a", "b"]


def test_plan_steps_from_raw_strategy_queue():
    assert plan_steps_from_raw({"strategy_queue": ["s1", "s2"]}) == ["s1", "s2"]


def test_plan_steps_from_raw_rejects_non_dict():
    msg = _error_of(plan_steps_from_raw, ["not", "a", "dict"])
    assert "dict" in msg


def test_plan_steps_from_raw_rejects_unknown_format():
    msg = _error_of(plan_steps_from_raw, {"unexpected": 1})
    assert "无法从以下字段提取步骤" in msg


def test_plan_steps_from_raw_half_payload_fails_loudly():
    """只给一半（缺 concretion）时必须**报错**，不能静默返回空列表。

    这是与阶段 3 那个 bug 的关键区别：宁可抛异常被上层看见，
    也不要返回 [] 让过滤器把所有样本静默丢掉。
    """
    msg = _error_of(plan_steps_from_raw, {"skeleton": {"nodes": DIAMOND_NODES}})
    assert "无法解析双层蓝图" in msg


# ============================================================================
# 6. 回归：双层蓝图绝不能静默变空
# ============================================================================
def test_regression_double_layer_never_silently_empty():
    steps = plan_steps_from_raw(_double())
    assert len(steps) > 0, "双层蓝图提取出 0 步 —— DTW 过滤器会静默筛掉全部样本"
    assert "计算A公司应支付的赔偿总额" in steps
