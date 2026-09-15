"""
Replanner 硬规则表 —— 两条链路共用同一份
=====================================
背景（阶段 5）：原先硬规则有两份且不一致 ——
  - `asynchronization/workers/replanner_worker.py` 里有 7 条
  - `multiple-search/soul.py` 的 node_replanner 里只有 1 条（未签劳动合同）
同一个法律案情走 Worker 链路还是 LangGraph 链路，补充检索行为不一样。

现在统一到这里，两边都 `from replanner_rules import REPLANNER_RULES, apply_hard_rules`。

规则形状：
    {
        "name":    "规则名（日志/回溯用）",
        "condition": lambda obs_text, queue -> bool,   # 是否命中
        "tasks":   [ {task_desc, engine, rationale}, ... ]  # 命中的追加任务
    }

扩展建议：`replanner_rules_report.py` 里整理了 P0/P1/P2 共约 25 条高频场景，
后续可逐步补齐；补齐时**只改这一个文件**。
"""

# ------------------------------------------------------------------
# 工具
# ------------------------------------------------------------------
def task_has_kw(queue: list, keyword: str) -> bool:
    """检查队列中是否已存在含某关键词的任务（避免重复追加）"""
    for t in queue:
        desc = t.get("task_desc", "") if isinstance(t, dict) else str(t)
        if keyword in desc:
            return True
    return False


def _any_kw(obs: str, keywords) -> bool:
    return any(kw in obs for kw in keywords)


# ------------------------------------------------------------------
# 规则表
# ------------------------------------------------------------------
REPLANNER_RULES = [
    {
        "name": "未签劳动合同 → 二倍工资",
        "condition": lambda obs, queue: (
            "未签订劳动合同" in obs and not task_has_kw(queue, "双倍工资")
        ),
        "tasks": [
            {"task_desc": "核查未签劳动合同二倍工资仲裁时效", "engine": "GRAPH_TRAVERSAL",
             "rationale": "硬规则: 未签合同缺双倍工资核查"},
            {"task_desc": "合并计算二倍工资差额与违法解除赔偿金", "engine": "GRAPH_TRAVERSAL",
             "rationale": "硬规则: 合并计算赔偿总额"},
        ],
    },
    {
        "name": "工伤/事故 → 工伤认定",
        "condition": lambda obs, queue: (
            _any_kw(obs, ["工伤", "受伤", "事故"]) and not task_has_kw(queue, "工伤")
        ),
        "tasks": [
            {"task_desc": "核实是否构成工伤及其认定时效", "engine": "GRAPH_TRAVERSAL",
             "rationale": "硬规则: 工伤缺认定"},
            {"task_desc": "计算工伤赔偿项目及数额", "engine": "GRAPH_TRAVERSAL",
             "rationale": "硬规则: 追加工伤赔偿"},
        ],
    },
    {
        "name": "社保断缴 → 社保核查",
        "condition": lambda obs, queue: (
            _any_kw(obs, ["社保", "五险一金", "断缴"]) and not task_has_kw(queue, "社保")
        ),
        "tasks": [
            {"task_desc": "核查用人单位社保缴纳义务及欠缴后果", "engine": "GRAPH_TRAVERSAL",
             "rationale": "硬规则: 社保断缴"},
            {"task_desc": "计算社保补缴或赔偿金额", "engine": "GRAPH_TRAVERSAL",
             "rationale": "硬规则: 社保金额计算"},
        ],
    },
    {
        "name": "加班/996 → 加班费",
        "condition": lambda obs, queue: (
            _any_kw(obs, ["加班", "加班费", "996"]) and not task_has_kw(queue, "加班")
        ),
        "tasks": [
            {"task_desc": "核实加班事实及加班费计算基数", "engine": "GRAPH_TRAVERSAL",
             "rationale": "硬规则: 加班费核查"},
            {"task_desc": "计算应付加班费总额", "engine": "GRAPH_TRAVERSAL",
             "rationale": "硬规则: 加班费计算"},
        ],
    },
    {
        "name": "竞业限制 → 竞业补偿",
        "condition": lambda obs, queue: (
            _any_kw(obs, ["竞业", "竞业限制"]) and not task_has_kw(queue, "竞业")
        ),
        "tasks": [
            {"task_desc": "核查竞业限制协议的有效性", "engine": "GRAPH_TRAVERSAL",
             "rationale": "硬规则: 竞业核查"},
            {"task_desc": "计算竞业限制补偿金", "engine": "GRAPH_TRAVERSAL",
             "rationale": "硬规则: 竞业补偿"},
        ],
    },
    {
        "name": "试用期 → 合法性核查",
        "condition": lambda obs, queue: (
            _any_kw(obs, ["试用期", "试用"]) and not task_has_kw(queue, "试用期")
        ),
        "tasks": [
            {"task_desc": "核实试用期的合法性（期限/次数/工资）", "engine": "GRAPH_TRAVERSAL",
             "rationale": "硬规则: 试用期"},
            {"task_desc": "判断试用期解除合同的法定条件", "engine": "GRAPH_TRAVERSAL",
             "rationale": "硬规则: 试用解除"},
        ],
    },
    {
        "name": "劳务派遣 → 派遣责任",
        "condition": lambda obs, queue: (
            _any_kw(obs, ["劳务派遣", "派遣"]) and not task_has_kw(queue, "派遣")
        ),
        "tasks": [
            {"task_desc": "核实劳务派遣的合法性与用工单位责任", "engine": "GRAPH_TRAVERSAL",
             "rationale": "硬规则: 派遣"},
            {"task_desc": "判断派遣工与用工单位间的法律关系", "engine": "GRAPH_TRAVERSAL",
             "rationale": "硬规则: 派遣关系"},
        ],
    },
]


def apply_hard_rules(obs_text: str, current_queue: list, logger=None) -> list:
    """
    跑一遍规则表，返回所有命中的追加任务（可能为空）。

    :param obs_text:      已有观察文本（各条 observation 的拼接）
    :param current_queue: 当前任务队列，用于去重判断
    :param logger:        可选 logger，命中时打印规则名
    """
    new_tasks = []
    for rule in REPLANNER_RULES:
        try:
            hit = rule["condition"](obs_text, current_queue)
        except Exception as e:  # 单条规则异常不应拖垮整条链路
            if logger:
                logger.warning(f"硬规则 '{rule.get('name')}' 判定异常，跳过: {e}")
            continue
        if hit:
            if logger:
                logger.info(f"硬规则命中: {rule.get('name')}")
            new_tasks.extend(rule["tasks"])
    return new_tasks


def list_rules() -> list:
    """列出全部规则名（便于运维审计覆盖率）"""
    return [r.get("name") for r in REPLANNER_RULES]
