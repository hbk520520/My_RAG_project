"""
Replanner Worker v2 —— 第四棒（救火队）：前面卡住了，重新规划
==========================================================
当前子任务怎么查都查不到时，这里接手。
先看有没有硬规则能命中（比如"提到加班却没算加班费"），
命中就追加；否则调 Replanner 模型生成新步骤，还能开启虫洞跨领域搜。

任务格式用 Pydantic 强类型：task_desc + engine + rationale，训练时 GRPO 可回溯。

技术栈: Kafka / Redis / Pydantic / DeepSeek API (JSON mode) / prompts
"""
import sys, os, json, logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError

# ---- 路径引导（同 planner_worker，见该文件说明）----
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_ASYNC_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in (_ROOT_DIR, _ASYNC_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config_loader import cfg
from state_manager import StateManager
from kafka_utils import (
    create_consumer, create_producer,
    TOPIC_REPLANNER_PENDING, TOPIC_RETRIEVER_PENDING, TOPIC_REASONER_PENDING,
    GROUP_REPLANNER, quarantine_message,
)

# 阶段 3：Prompt 统一从 prompts.py 导入，不再内联副本
from prompts import build_replanner_messages
# 阶段 5：硬规则表抽到项目根目录，与 soul.py 的 node_replanner 共用同一份
from replanner_rules import apply_hard_rules

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

    def _call_llm_messages(self, messages: List[Dict[str, str]],
                           require_json: bool = False,
                           temperature: float = 0.4) -> str:
        """按 messages 列表调用 LLM（阶段 3 新增，配合 prompts.py 的构造器）"""
        kwargs = {
            "model": cfg.get("llm", "judge_model"),
            "messages": messages,
            "temperature": temperature,
            "max_tokens": cfg.get("llm", "max_tokens_default"),
        }
        if require_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self.llm_client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content.strip()

    def _call_llm(self, system_prompt: str, user_prompt: str,
                  require_json: bool = False, temperature: float = 0.4) -> str:
        """保留原签名，内部转成 messages，兼容既有调用方"""
        return self._call_llm_messages(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": user_prompt}],
            require_json=require_json,
            temperature=temperature,
        )

    def generate_new_plan(self,
                          original_query: str,
                          global_facts: List[str],
                          retry_context: Dict[str, Any]) -> List[Dict]:
        # 阶段 3：Prompt 正文由 prompts.REPLANNER_SYSTEM 提供（含状态插值），
        # JSON Schema 由本模块注入，保证与 ReplanOutput 定义永远一致。
        messages = build_replanner_messages(
            original_query=original_query,
            global_facts=global_facts,
            retry_context=retry_context,
            schema_json=json.dumps(ReplanOutput.model_json_schema(), ensure_ascii=False),
        )

        try:
            raw = self._call_llm_messages(messages, require_json=True, temperature=0.4)
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
# 3. 硬规则引擎
# ============================================================================
# 阶段 5：规则表与判定函数已抽到项目根目录的 replanner_rules.py。
# 原先这里内联了 7 条规则，而 soul.py 的 node_replanner 只有 1 条 ——
# 同一个案情走 Worker 链路还是 LangGraph 链路，补充检索行为不一致。
# 现在两边都 `from replanner_rules import apply_hard_rules`。
#
# 规则文件顶部有扩展说明；replanner_rules_report.py 里整理了约 25 条
# P0/P1/P2 高频场景可供后续补齐（补齐时只改那一个文件）。


# ============================================================================
# 4. Worker 主循环
# ============================================================================
def main():
    consumer = create_consumer(TOPIC_REPLANNER_PENDING, GROUP_REPLANNER,
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

        # 阶段 6：单条消息的任何未处理异常都在这里收容（写 DLQ + 提交位移 + 继续）
        try:
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
                producer.send(TOPIC_REASONER_PENDING, key=session_id,
                              value={"session_id": session_id})
                producer.flush()
                consumer.commit()
                continue

            # 1. 先尝试硬规则（追加到当前队列前面，不覆盖已完成部分）
            hard_tasks = apply_hard_rules(obs_text, current_queue, logger)
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
                    logger.warning(f"🌌 开启 {len(wormholes)} 个虫洞: "
                                   f"{[w['task_desc'][:40] for w in wormholes]}")

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

        except Exception as e:
            quarantine_message(msg, e, logger)
            try:
                consumer.commit()
            except Exception as ce:
                logger.error(f"隔离毒消息时提交位移失败，该消息可能被重复投递: {ce}")

    consumer.close()


if __name__ == "__main__":
    main()
