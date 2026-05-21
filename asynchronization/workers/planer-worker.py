"""
Planner Worker —— 消费 Planner 队列，调用 Meta-Planner 生成任务队列
"""
import sys
import os
import json
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config_loader import cfg
from state_manager import StateManager
from kafka_utils import (
    create_consumer, create_producer,
    TOPIC_PLANNER_PENDING, TOPIC_RETRIEVER_PENDING
)

logging.basicConfig(
    level=getattr(logging, cfg.get("observability", "log_level", default="INFO")),
    format='%(asctime)s - [%(levelname)s] - %(name)s - %(message)s'
)
logger = logging.getLogger("PlannerWorker")


def call_planner_llm(user_query: str) -> list:
    """
    调用 Meta-Planner LLM 生成双层蓝图 P_q={S_q,C_q}，展平为可执行任务队列。
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=cfg.get("llm", "api_key"),
        base_url=cfg.get("llm", "base_url")
    )

    system_prompt = """你是一个顶级的中国法律案件拆解专家。
【核心任务】：不要回答法律问题！将用户案情拆解为「双层蓝图」。

【输出格式 (严格 JSON)】：
{
  "skeleton": {
    "nodes": [
      {"id": "1", "abstract": "核实劳动关系", "deps": []},
      {"id": "2", "abstract": "核查解除合法性", "deps": ["1"]},
      {"id": "3", "abstract": "计算赔偿金额", "deps": ["2"]}
    ]
  },
  "concretion": {
    "concretions": {
      "1": "具体查询(含实体名)",
      "2": "具体查询(含实体名)",
      "3": "具体查询(含实体名)"
    }
  }
}
【abstract 规则】：不包含具体人名/公司名/日期，用通用概念描述。
【deps 规则】：若步骤B依赖A的结果，在B的deps中写A的id。无依赖填[]。
"""
    try:
        response = client.chat.completions.create(
            model=cfg.get("llm", "judge_model"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"用户案情：{user_query}"}
            ],
            response_format={"type": "json_object"},
            temperature=cfg.get("llm", "temperature_plan"),
            max_tokens=cfg.get("llm", "max_tokens_default")
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)

        # 解析双层蓝图 → 展平
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", ".."))
        from double_layer_plan import parse_double_layer_plan
        plan = parse_double_layer_plan(data)
        flat = plan.to_flat_task_queue(respect_deps=True)
        logger.info(f"双层蓝图: {len(plan.skeleton.nodes)}节点 DAG → {len(flat)}步拓扑队列")
        return flat

    except Exception as e:
        logger.error(f"Planner LLM 调用失败: {e}")
        return [{"task_desc": "核查劳动关系基础事实", "engine": "GRAPH_TRAVERSAL", "rationale": "异常降级"}]


def main():
    consumer = create_consumer(
        TOPIC_PLANNER_PENDING, "planner-group",
        bootstrap_servers=cfg.get("kafka", "bootstrap_servers")
    )
    producer = create_producer(
        bootstrap_servers=cfg.get("kafka", "bootstrap_servers")
    )
    state_manager = StateManager(
        redis_url=cfg.get("redis", "url"),
        expire_seconds=cfg.get("redis", "state_expire_seconds")
    )

    logger.info("Planner Worker started, waiting for tasks...")

    for msg in consumer:
        session_id = msg.key or (msg.value.get("session_id") if isinstance(msg.value, dict) else None)
        if not session_id:
            logger.error("No session_id in message, skip")
            consumer.commit()
            continue

        try:
            state = state_manager.load_state(session_id)
        except KeyError:
            logger.error(f"Session {session_id} not found, skip")
            consumer.commit()
            continue

        # 执行规划
        plan = call_planner_llm(state["user_query"])
        state["task_queue"] = plan
        state["current_step"] = "planner_done"
        state["recursion_depth"] = 0

        # 脱水保存
        state_manager.save_state(session_id, state)

        # 发送给 Retriever
        producer.send(TOPIC_RETRIEVER_PENDING, key=session_id,
                      value={"session_id": session_id})
        producer.flush()

        # 手动提交位移
        consumer.commit()
        logger.info(f"Session {session_id}: plan generated ({len(plan)} tasks)")

    consumer.close()


if __name__ == "__main__":
    main()