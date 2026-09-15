"""
Replanner 硬规则表单元测试
==========================
回归背景（阶段 5）：硬规则曾经有两份且不一致 ——
  - `asynchronization/workers/replanner_worker.py` 里 7 条
  - `multiple-search/soul.py` 的 `node_replanner` 里只有 1 条（未签劳动合同）
同一个案情走 Kafka Worker 链路还是 LangGraph 链路，补充检索行为不一样。
现在统一到 `replanner_rules.py`。本文件除了测规则本身，还锁定
「两条链路共用同一份规则表」这个不变量。
"""
import inspect
import logging

import pytest

import replanner_rules
from replanner_rules import REPLANNER_RULES, apply_hard_rules, list_rules, task_has_kw

# (观察文本, 规则名, 该规则的"去重关键词")
# 去重关键词取自规则 condition 里 task_has_kw() 的第二个参数
RULE_CASES = [
    ("公司一直没跟我签合同，属于未签订劳动合同的情况",
     "未签劳动合同 → 二倍工资", "双倍工资"),
    ("入职三个月后从脚手架上摔下受伤",
     "工伤/事故 → 工伤认定", "工伤"),
    ("公司一年没给我交社保了",
     "社保断缴 → 社保核查", "社保"),
    ("每天加班到十一点，从没给过加班费",
     "加班/996 → 加班费", "加班"),
    ("离职时公司让我签竞业限制协议",
     "竞业限制 → 竞业补偿", "竞业"),
    ("试用期最后一天被辞退",
     "试用期 → 合法性核查", "试用期"),
    ("我是劳务派遣过去的，用工单位把我退了",
     "劳务派遣 → 派遣责任", "派遣"),
]


def _tasks_of(rule_name: str) -> list:
    return next(r for r in REPLANNER_RULES if r["name"] == rule_name)["tasks"]


# ============================================================================
# 1. 规则表结构
# ============================================================================
def test_rule_table_has_seven_rules():
    assert len(REPLANNER_RULES) == 7
    assert len(list_rules()) == 7
    assert len(set(list_rules())) == 7, "规则名必须唯一（日志/回溯要用）"


def test_every_rule_has_required_shape():
    for rule in REPLANNER_RULES:
        assert set(rule) == {"name", "condition", "tasks"}, rule.get("name")
        assert isinstance(rule["name"], str) and rule["name"]
        assert callable(rule["condition"])
        assert rule["tasks"], f"规则 {rule['name']} 没有任何追加任务"
        for t in rule["tasks"]:
            assert set(t) == {"task_desc", "engine", "rationale"}, rule["name"]
            assert t["task_desc"] and t["engine"] and t["rationale"]


def test_condition_is_callable_for_every_rule_without_error():
    """空观察 + 空队列时每条规则都应能正常求值（不抛异常）"""
    for rule in REPLANNER_RULES:
        assert rule["condition"]("", []) in (True, False)


# ============================================================================
# 2. task_has_kw 去重工具
# ============================================================================
@pytest.mark.parametrize("queue,keyword,expected", [
    ([{"task_desc": "核查二倍工资仲裁时效"}], "双倍工资", False),
    ([{"task_desc": "包含双倍工资的任务"}], "双倍工资", True),
    (["纯字符串任务-含社保二字"], "社保", True),
    ([], "任何词", False),
    ([{"task_desc": ""}, "无关任务"], "社保", False),
])
def test_task_has_kw(queue, keyword, expected):
    assert task_has_kw(queue, keyword) is expected


def test_task_has_kw_tolerates_non_dict_non_str_entries():
    """队列里混入 None / 数字时不能崩（生产上队列来自反序列化 JSON）"""
    assert task_has_kw([None, 123, {"task_desc": "含社保"}], "社保") is True


# ============================================================================
# 3. 每条规则都能命中
# ============================================================================
@pytest.mark.parametrize("obs,rule_name,_dedup", RULE_CASES,
                         ids=[c[1] for c in RULE_CASES])
def test_each_rule_hits_with_empty_queue(obs, rule_name, _dedup):
    assert apply_hard_rules(obs, []) == _tasks_of(rule_name)


@pytest.mark.parametrize("obs,rule_name,_dedup", RULE_CASES,
                         ids=[c[1] for c in RULE_CASES])
def test_each_rule_hit_is_not_default(obs, rule_name, _dedup):
    tasks = apply_hard_rules(obs, [])
    assert tasks, f"{rule_name} 未命中：{obs!r}"


# ============================================================================
# 4. 去重：队列里已有同主题任务就不再追加
# ============================================================================
@pytest.mark.parametrize("obs,rule_name,dedup_kw", RULE_CASES,
                         ids=[c[1] for c in RULE_CASES])
def test_rule_skipped_when_queue_already_has_topic(obs, rule_name, dedup_kw):
    queue = [{"task_desc": f"已有的{dedup_kw}相关任务", "engine": "GRAPH_TRAVERSAL", "rationale": ""}]
    assert apply_hard_rules(obs, queue) == []


def test_no_rule_hits_for_unrelated_observation():
    assert apply_hard_rules("公司拖欠了上个月的报销款", []) == []


def test_empty_observation_hits_nothing():
    assert apply_hard_rules("", []) == []


def test_multiple_rules_can_hit_at_once():
    obs = "公司未签订劳动合同，而且每天加班没有加班费"
    tasks = apply_hard_rules(obs, [])
    assert tasks == _tasks_of("未签劳动合同 → 二倍工资") + _tasks_of("加班/996 → 加班费")


# ============================================================================
# 5. 健壮性
# ============================================================================
def test_broken_rule_does_not_break_the_rest(monkeypatch, caplog):
    """单条规则抛异常时必须被跳过，不能拖垮整条重规划链路"""
    broken = {
        "name": "坏规则",
        "condition": lambda obs, queue: 1 / 0,
        "tasks": [{"task_desc": "不该出现", "engine": "X", "rationale": "X"}],
    }
    healthy = {
        "name": "好规则",
        "condition": lambda obs, queue: True,
        "tasks": [{"task_desc": "应出现", "engine": "GRAPH_TRAVERSAL", "rationale": "ok"}],
    }
    monkeypatch.setattr(replanner_rules, "REPLANNER_RULES", [broken, healthy])

    with caplog.at_level(logging.WARNING):
        tasks = apply_hard_rules("任意观察", [], logging.getLogger("test_rules"))

    assert tasks == healthy["tasks"]
    assert "坏规则" in caplog.text


def test_non_boolean_condition_result_is_treated_as_truthy():
    """condition 返回非布尔值（如字符串）时按真值判断，不抛异常"""
    rule = {
        "name": "真值规则",
        "condition": lambda obs, queue: "命中",
        "tasks": [{"task_desc": "T", "engine": "E", "rationale": "R"}],
    }
    original = replanner_rules.REPLANNER_RULES
    replanner_rules.REPLANNER_RULES = [rule]
    try:
        assert apply_hard_rules("x", []) == rule["tasks"]
    finally:
        replanner_rules.REPLANNER_RULES = original


def test_apply_hard_rules_does_not_mutate_input_queue():
    queue = [{"task_desc": "原有任务", "engine": "GRAPH_TRAVERSAL", "rationale": ""}]
    snapshot = [dict(t) for t in queue]
    apply_hard_rules("未签订劳动合同", queue)
    assert queue == snapshot


def test_apply_hard_rules_returns_a_new_list():
    """返回值必须是新列表，调用方 `hard_tasks + queue` 拼接时才不会污染规则表"""
    out1 = apply_hard_rules("未签订劳动合同", [])
    out2 = apply_hard_rules("未签订劳动合同", [])
    assert out1 == out2
    assert out1 is not replanner_rules.REPLANNER_RULES
    assert out1 is not out2


def test_apply_hard_rules_without_logger_is_silent():
    """logger 是可选参数，不传时不能报错"""
    assert apply_hard_rules("未签订劳动合同", [])


# ============================================================================
# 6. 两条链路共用同一份规则表（阶段 5 核心不变量）
# ============================================================================
def test_both_pipelines_share_one_rule_table():
    try:
        import soul
        import replanner_worker
    except Exception as e:  # pragma: no cover - 依赖缺失时跳过
        pytest.skip(f"链路模块不可导入: {e}")

    # 1) 两条链路引用的是同一个函数对象（不是各自拷贝的一份实现）
    assert soul.apply_hard_rules is replanner_rules.apply_hard_rules
    assert replanner_worker.apply_hard_rules is replanner_rules.apply_hard_rules

    # 2) 两条链路都没有自己再定义一份
    for mod in (soul, replanner_worker):
        src = inspect.getsource(mod)
        assert "def apply_hard_rules" not in src, f"{mod.__name__} 又自带了一份实现"
        assert "def _task_has_kw" not in src, f"{mod.__name__} 又自带了一份去重工具"

    # 3) LangGraph 链路的 node_replanner 确实走了共用规则
    assert "apply_hard_rules" in inspect.getsource(soul.node_replanner)


def test_rule_names_are_stable_for_operational_audit():
    """规则名会被写进日志/审计，改动时应是有意为之"""
    assert list_rules() == [
        "未签劳动合同 → 二倍工资",
        "工伤/事故 → 工伤认定",
        "社保断缴 → 社保核查",
        "加班/996 → 加班费",
        "竞业限制 → 竞业补偿",
        "试用期 → 合法性核查",
        "劳务派遣 → 派遣责任",
    ]
