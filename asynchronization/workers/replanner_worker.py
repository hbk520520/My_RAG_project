"""
Replanner Worker v2 —— 基于 Pydantic 强类型约束 + 显式引擎选择
引擎：GRAPH_TRAVERSAL (图游走) / GLOBAL_DENSE_WORMHOLE (全局向量穿越)
"""
import sys
import os
import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config_loader import cfg
from state_manager import StateManager
from kafka_utils import (
    create_consumer, create_producer,
    TOPIC_REPLANNER_PENDING, TOPIC_RETRIEVER_PENDING, TOPIC_REASONER_PENDING
)

logging.basicConfig(
    level=getattr(logging, cfg.get("observability", "log_level", default="INFO")),
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger("ReplannerWorker")


# ============================================================================
# 1. 强类型 Schema
# ============================================================================
class ReplanTask(BaseModel):
    task_desc: str = Field(..., description="新的原子查询步骤或重组线索")
    engine: str = Field(
        default="GRAPH_TRAVERSAL",
        description="'GRAPH_TRAVERSAL' (图游走) 或 'GLOBAL_DENSE_WORMHOLE' (全局向量穿越)"
    )
    rationale: str = Field(..., description="推演理由，用于 GRPO 轨迹回溯")


class ReplanOutput(BaseModel):
    task_queue: List[ReplanTask] = Field(..., description="重规划后的任务队列")


# ============================================================================
# 2. ReplannerOps —— 封装 LLM 调用
# ============================================================================
class ReplannerOps:
    def __init__(self, llm_client=None):
        if llm_client is None:
            from openai import OpenAI
            llm_client = OpenAI(
                api_key=cfg.get("llm", "api_key"),
                base_url=cfg.get("llm", "base_url")
            )
        self.llm_client = llm_client

    def _call_llm(self, system_prompt: str, user_prompt: str,
                  require_json: bool = False, temperature: float = 0.4) -> str:
        kwargs = {
            "model": cfg.get("llm", "judge_model"),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": cfg.get("llm", "max_tokens_default"),
        }
        if require_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.llm_client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()

    def generate_new_plan(self,
                          original_query: str,
                          global_facts: List[str],
                          retry_context: Dict[str, Any]) -> List[Dict]:
        fail_count = retry_context.get("fail_count", 0)
        fail_log = retry_context.get("fail_log", "无明确报错，检索结果为空")

        system_prompt = f"""你是一个经过强化学习训练的顶级重规划引擎 (Replanner)。
【核心任务】：当前系统的法律检索路径已陷入死胡同，你需要基于全局事实账本，推翻或改写下一步的调查计划。

【可用引擎说明】：
1. GRAPH_TRAVERSAL（图谱游走）：系统默认引擎。在当前案件领域内寻找相邻线索（成本极低）。
2. GLOBAL_DENSE_WORMHOLE（虫洞穿越）：当当前图谱已彻底断裂，必须跨法律领域寻找依据时使用（算力消耗极大！连续失败>=3次才建议开启）。

【当前绝境状态】：
- 系统已连续碰壁次数：{fail_count}
- 碰壁原因：{fail_log}

严格按照以下 JSON Schema 输出：
{ReplanOutput.schema_json()}
"""
        user_prompt = (f"用户原始诉求：{original_query}\n"
                       f"当前已掌握的铁证：{json.dumps(global_facts, ensure_ascii=False)}")

        try:
            raw = self._call_llm(system_prompt, user_prompt, require_json=True, temperature=0.4)
            parsed = json.loads(raw)
            validated = ReplanOutput(**parsed)
            result = [t.model_dump() for t in validated.task_queue]
            logger.info(f"Replanner 生成 {len(result)} 个新任务")
            return result
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Replanner JSON 崩溃: {e}")
            return [{"task_desc": original_query, "engine": "GRAPH_TRAVERSAL",
                     "rationale": "格式降级强制重试"}]
        except Exception as e:
            logger.error(f"Replanner 调用异常: {e}")
            return [{"task_desc": "全局检索案情相关法条", "engine": "GLOBAL_DENSE_WORMHOLE",
                     "rationale": f"LLM异常降级: {str(e)[:100]}"}]


# ============================================================================
# 3. 硬规则引擎（适配新 Dict 任务格式）
# ============================================================================
def _task_has_kw(queue: list, keyword: str) -> bool:
    """检查队列中是否已存在含某关键词的任务"""
    for t in queue:
        desc = t.get("task_desc", "") if isinstance(t, dict) else str(t)
        if keyword in desc:
            return True
    return False


REPLANNER_RULES = [
    {
        "condition": lambda obs, queue: "未签订劳动合同" in obs and not _task_has_kw(queue, "双倍工资"),
        "tasks": [
            {"task_desc": "核查未签劳动合同二倍工资仲裁时效", "engine": "GRAPH_TRAVERSAL",
             "rationale": "硬规则: 未签合同缺双倍工资核查"},
            {"task_desc": "合并计算二倍工资差额与违法解除赔偿金", "engine": "GRAPH_TRAVERSAL",
             "rationale": "硬规则: 合并计算赔偿总额"},
        ],
    },
    {
        "condition": lambda obs, queue: any(kw in obs for kw in ["工伤", "受伤", "事故"]) and not _task_has_kw(queue, "工伤"),
        "tasks": [
            {"task_desc": "核实是否构成工伤及其认定时效", "engine": "GRAPH_TRAVERSAL", "rationale": "硬规则: 工伤缺认定"},
            {"task_desc": "计算工伤赔偿项目及数额", "engine": "GRAPH_TRAVERSAL", "rationale": "硬规则: 追加工伤赔偿"},
        ],
    },
    {
        "condition": lambda obs, queue: any(kw in obs for kw in ["社保", "五险一金", "断缴"]) and not _task_has_kw(queue, "社保"),
        "tasks": [
            {"task_desc": "核查用人单位社保缴纳义务及欠缴后果", "engine": "GRAPH_TRAVERSAL", "rationale": "硬规则: 社保断缴"},
            {"task_desc": "计算社保补缴或赔偿金额", "engine": "GRAPH_TRAVERSAL", "rationale": "硬规则: 社保金额计算"},
        ],
    },
    {
        "condition": lambda obs, queue: any(kw in obs for kw in ["加班", "加班费", "996"]) and not _task_has_kw(queue, "加班"),
        "tasks": [
            {"task_desc": "核实加班事实及加班费计算基数", "engine": "GRAPH_TRAVERSAL", "rationale": "硬规则: 加班费核查"},
            {"task_desc": "计算应付加班费总额", "engine": "GRAPH_TRAVERSAL", "rationale": "硬规则: 加班费计算"},
        ],
    },
    {
        "condition": lambda obs, queue: any(kw in obs for kw in ["竞业", "竞业限制"]) and not _task_has_kw(queue, "竞业"),
        "tasks": [
            {"task_desc": "核查竞业限制协议的有效性", "engine": "GRAPH_TRAVERSAL", "rationale": "硬规则: 竞业核查"},
            {"task_desc": "计算竞业限制补偿金", "engine": "GRAPH_TRAVERSAL", "rationale": "硬规则: 竞业补偿"},
        ],
    },
    {
        "condition": lambda obs, queue: any(kw in obs for kw in ["试用期", "试用"]) and not _task_has_kw(queue, "试用期"),
        "tasks": [
            {"task_desc": "核实试用期的合法性（期限/次数/工资）", "engine": "GRAPH_TRAVERSAL", "rationale": "硬规则: 试用期"},
            {"task_desc": "判断试用期解除合同的法定条件", "engine": "GRAPH_TRAVERSAL", "rationale": "硬规则: 试用解除"},
        ],
    },
    {
        "condition": lambda obs, queue: any(kw in obs for kw in ["劳务派遣", "派遣"]) and not _task_has_kw(queue, "派遣"),
        "tasks": [
            {"task_desc": "核实劳务派遣的合法性与用工单位责任", "engine": "GRAPH_TRAVERSAL", "rationale": "硬规则: 派遣"},
            {"task_desc": "判断派遣工与用工单位间的法律关系", "engine": "GRAPH_TRAVERSAL", "rationale": "硬规则: 派遣关系"},
        ],
    },
]


def apply_hard_rules(obs_text: str, current_queue: list) -> list:
    new_tasks = []
    for rule in REPLANNER_RULES:
        if rule["condition"](obs_text, current_queue):
            logger.info(f"硬规则命中")
            new_tasks.extend(rule["tasks"])
    return new_tasks


# ============================================================================
# 4. Worker 主循环
# ============================================================================
def main():
    consumer = create_consumer(TOPIC_REPLANNER_PENDING, "replanner-group",
                               bootstrap_servers=cfg.get("kafka", "bootstrap_servers"))
    producer = create_producer(bootstrap_servers=cfg.get("kafka", "bootstrap_servers"))
    state_manager = StateManager(redis_url=cfg.get("redis", "url"),
                                 expire_seconds=cfg.get("redis", "state_expire_seconds"))
    replanner_ops = ReplannerOps()
    logger.info("Replanner Worker v2 started, waiting for tasks...")

    for msg in consumer:
        session_id = msg.key or msg.value.get("session_id")
        try:
            state = state_manager.load_state(session_id)
        except KeyError:
            logger.error(f"Session {session_id} not found")
            consumer.commit()
            continue

        obs_text = "\n".join(
            str(o.get("task", "")) + " " + str(o.get("extracted_facts", ""))
            for o in state.get("past_observations", []) if isinstance(o, dict)
        )
        current_queue = state.get("task_queue", [])
        retry_context = state.get("retry_context", {})

        # 熔断保护
        max_depth = cfg.get("agent", "max_recursion_depth")
        if retry_context.get("status") == "force_stop" or state.get("recursion_depth", 0) > max_depth:
            logger.warning(f"Session {session_id}: 算力熔断，强制结案")
            state["task_queue"] = []
            state["retry_context"] = {"status": "force_stop"}
            state_manager.save_state(session_id, state)
            producer.send(TOPIC_REASONER_PENDING, key=session_id, value={"session_id": session_id})
            producer.flush()
            consumer.commit()
            continue

        # 1. 先尝试硬规则（追加到当前队列前面，不覆盖已完成部分）
        hard_tasks = apply_hard_rules(obs_text, current_queue)
        if hard_tasks:
            new_queue = hard_tasks + current_queue
            state["retry_context"] = {"status": "hard_rule_expanded"}
        else:
            # 2. 否则走 LLM 重规划 —— 只针对当前失败的子任务，而非整个问题
            retry_context["fail_log"] = obs_text[-500:] if obs_text else "无有效检索结果"

            # ---- 关键修复：Replanner 只重规划当前失败的任务 ----
            current_failed_task = current_queue[0] if current_queue else {}
            failed_desc = (
                current_failed_task.get("task_desc", "")
                if isinstance(current_failed_task, dict)
                else str(current_failed_task)
            )
            replan_target = failed_desc if failed_desc else state.get("user_query", "")

            new_tasks = replanner_ops.generate_new_plan(
                original_query=replan_target,      # ← 只传失败的子任务
                global_facts=state.get("global_facts", []),
                retry_context=retry_context
            )
            # 新任务替换队列头，保留队列尾部（其他未执行的任务）
            new_queue = new_tasks + current_queue[1:]
            state["retry_context"] = {"status": "llm_replanned"}
            state["recursion_depth"] = state.get("recursion_depth", 0) + 1

            # 监控虫洞
            wormholes = [t for t in new_queue if t.get("engine") == "GLOBAL_DENSE_WORMHOLE"]
            if wormholes:
                logger.warning(f"🌌 开启 {len(wormholes)} 个虫洞: {[w['task_desc'][:40] for w in wormholes]}")

        state["task_queue"] = new_queue
        state.setdefault("past_observations", []).append({
            "task": "__replan__",
            "new_queue": [t["task_desc"] if isinstance(t, dict) else str(t) for t in new_queue],
            "status": "replanned"
        })

        state_manager.save_state(session_id, state)

        target = TOPIC_REASONER_PENDING if not new_queue else TOPIC_RETRIEVER_PENDING
        producer.send(target, key=session_id, value={"session_id": session_id})
        producer.flush()
        consumer.commit()
        logger.info(f"Session {session_id}: 重规划完成, {len(new_queue)} 任务")

    consumer.close()


if __name__ == "__main__":
    main()
