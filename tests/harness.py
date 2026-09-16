"""
P0 · 离线测试夹具（harness）
============================
让"一条命令跑完整链路"成为可能。**全程不联网、不需要 API key、不需要 Kafka/Redis/Docker。**

三个夹具：

  `ProgrammableLLM`  —— 可编程假 LLM，按 system prompt 关键词路由应答，
                        并**记录每一次调用**（便于断言"Grader 被调了几次"）
  `MockKafka`        —— 内存任务总线，模拟 append-only log + 按消费组记录 offset，
                        可断言"投递序列"与"消息积压"
  `make_agent()`     —— 装配一套完整的 LangGraph Agent（真图引擎 + StubEncoder + 假 LLM）

为什么要有这个：
  P1 起每一步都会动运行时与节点接线。没有"一键跑完整图"的沙盘，
  每次改动只能靠手测，必然反复回归 —— 这是 P0 存在的全部理由。
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

# ===========================================================================
# 1. 可编程假 LLM
# ===========================================================================
@dataclass
class LLMCall:
    """一次 LLM 调用的完整记录，供断言使用"""

    index: int
    system: str
    user: str
    require_json: bool
    temperature: float
    response: str

    def matches(self, keyword: str) -> bool:
        return keyword in self.system or keyword in self.user


class ProgrammableLLM:
    """
    按关键词路由的假 LLM。

    用法：
        llm = ProgrammableLLM()
        llm.on("事实调查官", '{"status": "sufficient", "extracted_facts": []}')
        llm.set_default("Mock 回答")
        llm.calls                          # → List[LLMCall]
        llm.calls_for("事实调查官")         # → 命中该关键词的调用
    """

    def __init__(self, default: str = "Mock 回答"):
        self._rules: List[Tuple[str, str]] = []
        self._default = default
        self.calls: List[LLMCall] = []
        self._callable_hooks: Dict[str, Callable[[List[Dict[str, str]]], str]] = {}

    # ---------------------------------------------------------------- 配置
    def on(self, keyword: str, response: str) -> "ProgrammableLLM":
        """命中 keyword（出现在 system 或 user 里）则返回 response。

        后注册的规则优先 —— 便于在基类规则之上覆盖个别场景。
        """
        self._rules.insert(0, (keyword, response))
        return self

    def on_dynamic(self, keyword: str,
                   fn: Callable[[List[Dict[str, str]]], str]) -> "ProgrammableLLM":
        """按需生成应答（如"第一次说资料不足、第二次说充足"）"""
        self._callable_hooks[keyword] = fn
        return self

    def set_default(self, response: str) -> "ProgrammableLLM":
        self._default = response
        return self

    def reset(self) -> None:
        self.calls.clear()

    # ---------------------------------------------------------------- 调用
    def __call__(self, messages: List[Dict[str, str]],
                 require_json: bool = False,
                 temperature: float = 0.1) -> str:
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if len(messages) > 1 else ""

        response = self._default
        for keyword, hook in self._callable_hooks.items():
            if keyword in system or keyword in user:
                response = hook(messages)
                break
        else:
            for keyword, canned in self._rules:
                if keyword in system or keyword in user:
                    response = canned
                    break

        self.calls.append(LLMCall(index=len(self.calls), system=system, user=user,
                                  require_json=require_json,
                                  temperature=temperature, response=response))
        return response

    # ---------------------------------------------------------------- 断言辅助
    def calls_for(self, keyword: str) -> List[LLMCall]:
        return [c for c in self.calls if c.matches(keyword)]

    def count_for(self, keyword: str) -> int:
        return len(self.calls_for(keyword))

    def keywords_hit(self) -> List[str]:
        seen, out = set(), []
        for c in self.calls:
            key = c.system[:24]
            if key not in seen:
                seen.add(key)
                out.append(key)
        return out


# ===========================================================================
# 2. 内存任务总线（模拟 Kafka 的 append-only log + 消费组 offset）
# ===========================================================================
@dataclass
class BusMessage:
    topic: str
    key: Optional[str]
    value: Dict[str, Any]
    offset: int


class MockKafka:
    """
    模拟 Kafka 最关键的语义：
      · 每个 topic 是一条 append-only 日志（消息带 offset）
      · 消费者按 **消费组** 记录自己的 offset；未 commit 的消息会重复消费
      · 可查看"积压"（未 commit 的消息数）

    不模拟：分区、副本、再均衡、压缩 —— P0 用不到。
    """

    def __init__(self):
        self._log: Dict[str, List[BusMessage]] = defaultdict(list)
        self._offsets: Dict[Tuple[str, str], int] = {}   # (topic, group) -> next offset

    # ---------------------------------------------------------------- 生产
    def publish(self, topic: str, value: Dict[str, Any],
                key: Optional[str] = None) -> BusMessage:
        msg = BusMessage(topic=topic, key=key, value=value,
                         offset=len(self._log[topic]))
        self._log[topic].append(msg)
        return msg

    # ---------------------------------------------------------------- 消费
    def poll(self, topic: str, group: str = "default") -> List[BusMessage]:
        """取出上次 commit 之后的消息（不改变 offset —— Kafka 是拉取语义）"""
        start = self._offsets.get((topic, group), 0)
        return self._log[topic][start:]

    def commit(self, msg: BusMessage, group: str = "default") -> None:
        """手动提交：offset 推进到该消息之后（本项目用 enable_auto_commit=false）"""
        self._offsets[(topic := msg.topic, group)] = msg.offset + 1

    def commit_all(self, topic: str, group: str = "default") -> None:
        self._offsets[(topic, group)] = len(self._log[topic])

    # ---------------------------------------------------------------- 断言辅助
    def pending(self, topic: str, group: str = "default") -> int:
        return max(len(self._log[topic]) - self._offsets.get((topic, group), 0), 0)

    def all_messages(self, topic: str) -> List[BusMessage]:
        return list(self._log[topic])

    def values(self, topic: str) -> List[Dict[str, Any]]:
        return [m.value for m in self._log[topic]]

    def topics(self) -> List[str]:
        return sorted(t for t, msgs in self._log.items() if msgs)

    def drain(self, topic: str, group: str = "default") -> List[BusMessage]:
        """一次性取出并提交（模拟 worker 处理完一批）"""
        msgs = self.poll(topic, group)
        for m in msgs:
            self.commit(m, group)
        return msgs

    def reset(self) -> None:
        self._log.clear()
        self._offsets.clear()


# ===========================================================================
# 3. 检索可控编码器
# ===========================================================================
# 为什么不能直接用 `StubEncoder`：它按文本哈希生成**互相近乎正交**的向量，
# 任何查询与任何法条的余弦相似度都在 0 附近。而 `Reasoner.retrieve()`
# 有 `sim > 0.6` 的硬过滤 —— 于是检索恒定返回空串，整条链路滑向
# "未检索到相关文档 → irrelevant → 虫洞重规划"，测试就测不到真实路径。
#
# `ScriptedEncoder` 改为**按关键词分桶**：同一主题的文本映射到同一个向量（cos=1.0），
# 不同主题近乎正交。于是"检索到哪一条"变成完全可断言的事实。
SCRIPTED_TOPICS: Dict[str, List[str]] = {
    "unlawful_dismissal": ["违法解除", "辞退", "解除劳动合同", "赔偿金", "二倍"],
    "severance": ["经济补偿", "工作年限", "每满一年", "一个月工资"],
    "written_contract": ["书面劳动合同", "建立劳动关系", "订立"],
}


class ScriptedEncoder:
    """按关键词主题分桶的编码器，接口与 `BGEM3FlagModel.encode()` 对齐"""

    def __init__(self, dim: int = 1024,
                 topics: Optional[Dict[str, List[str]]] = None,
                 seed: int = 7):
        import numpy as np

        self.dim = dim
        self.topics = topics or SCRIPTED_TOPICS
        rng = np.random.RandomState(seed)
        self._topic_vecs: Dict[str, Any] = {}
        for name in self.topics:
            v = rng.randn(dim).astype(np.float32)
            self._topic_vecs[name] = v / np.linalg.norm(v)

    def topic_of(self, text: str) -> Optional[str]:
        """命中关键词最多的主题；无命中返回 None"""
        best, best_hits = None, 0
        for name, keywords in self.topics.items():
            hits = sum(1 for kw in keywords if kw in text)
            if hits > best_hits:
                best, best_hits = name, hits
        return best

    def _vec(self, text: str):
        import hashlib as _hashlib
        import numpy as np

        topic = self.topic_of(text)
        if topic:
            return self._topic_vecs[topic]
        # 无关键词命中 → 确定性向量；dim 较大时与主题向量近乎正交（cos ≈ 0）
        digest = _hashlib.blake2b(str(text).encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "big") % (2 ** 31)
        v = np.random.RandomState(h).randn(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def encode(self, texts, return_dense=True, return_sparse=True):
        import numpy as np

        dense = (np.stack([self._vec(t) for t in texts]) if len(texts)
                 else np.zeros((0, self.dim), dtype=np.float32))
        return {
            "dense_vecs": dense.astype(np.float32),
            "lexical_weights": [{self.topic_of(str(t)) or "unknown": 0.5} for t in texts],
        }


# ===========================================================================
# 4. 离线 Agent 装配
# ===========================================================================
DEFAULT_DUMMY_NODES = [
    {"id": 1, "content": "用人单位违法解除劳动合同，应当按经济补偿标准的二倍支付赔偿金。",
     "type": "Raw", "metadata": {"source": "劳动合同法"}},
    {"id": 2, "content": "经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资。",
     "type": "Raw", "metadata": {"source": "劳动合同法"}},
    {"id": 3, "content": "建立劳动关系，应当订立书面劳动合同。",
     "type": "Raw", "metadata": {"source": "劳动合同法"}},
]


def make_offline_graph_engine(nodes: Optional[List[Dict[str, Any]]] = None,
                              encoder=None):
    """
    构造一个**已注入离线编码器**的图引擎，并灌入若干法条节点。

    绝不会有网络访问 —— 默认编码器是纯本地的 `ScriptedEncoder`。
    """
    from dataset.graph import LegalDenseGraphBuilder

    engine = LegalDenseGraphBuilder.from_config()
    engine.set_encoder(encoder if encoder is not None
                       else ScriptedEncoder(engine.dim))

    engine.build_initial_graph_batch(
        nodes if nodes is not None else DEFAULT_DUMMY_NODES)
    return engine


class _ForbiddenLLMClient:
    """
    「碰了就报错」的假 LLM 客户端。

    为什么需要它：`AgenticNodesOperator` 继承 `LegalLLMBase`，后者在 `__init__` 里
    就建好了 `self.client`。如果有代码**绕过 `_call_messages`** 直接调 `self.client`，
    离线测试就会真的去打网络（或在没有 Key 时静默失败），而测试**照样通过** ——
    这类"绕过注入接缝"的缺陷此前只能靠人肉 review 发现
    （P1 前 `node_replanner` 的虫洞分支就是直连 `get_llm_client()` 的）。

    现在把它换成这个替身：任何属性访问都会抛 AssertionError，于是"绕过接缝"
    立刻变成一次响亮的测试失败。
    """

    def __getattr__(self, name: str):
        raise AssertionError(
            f"有代码绕过了 `_call_messages` 去直接用 `self.client.{name}` —— "
            f"这会让离线测试失去隔离性。请改走 _call_messages 注入接缝。"
        )


def make_agent(llm: ProgrammableLLM,
               engine=None,
               checkpointer=None):
    """
    装配一套完整的 Plan-and-Replan Agent。

    :param llm:     `ProgrammableLLM`（会替换掉 Operator 的 `_call_messages`）
    :param engine:  图引擎；None 时用 `make_offline_graph_engine()`
    :param checkpointer: 传入 LangGraph checkpointer 即可获得持久化能力（P1 用）
    :return: (compiled_graph, engine, ops)
    """
    from soul import AgenticNodesOperator, Reasoner, build_plan_replan_agent

    engine = engine or make_offline_graph_engine()

    class _ProgrammableOps(AgenticNodesOperator):
        """唯一 LLM 接缝就在这里覆写 —— 节点逻辑本身一行不改"""

        def _call_messages(self, messages, require_json=False, temperature=0.1):
            return llm(messages, require_json=require_json, temperature=temperature)

    ops = _ProgrammableOps(api_key="offline-test-key")     # 假 key，不会真的发请求
    ops.client = _ForbiddenLLMClient()                     # 任何绕过接缝的调用都会炸

    reasoner = Reasoner(legal_graph=engine,
                        embedding_fn=lambda t: engine.encode_text(t)["dense"],
                        agentic_ops=ops)

    agent = build_plan_replan_agent(reasoner, ops, checkpointer=checkpointer)
    return agent, engine, ops


def initial_state(user_query: str) -> Dict[str, Any]:
    """满足 AgentState 的最小初始状态"""
    return {
        "user_query": user_query,
        "task_queue": [],
        "past_observations": [],
        "final_report": "",
        "global_facts": [],
        "retry_context": {},
        "recursion_depth": 0,
        "sandbox_session_id": None,
        "generated_code": None,
        "calc_result": None,
        "sandbox_retries": 0,
        "needs_sandbox_calc": False,
    }


def run_and_trace(agent, user_query: str, thread_id: Optional[str] = None
                  ) -> Tuple[List[str], Dict[str, Any]]:
    """
    跑一遍图并返回 (节点访问序列, 终态)。

    同时订阅两种 stream 模式：
      · `updates` —— 每个节点跑完后的增量，用来还原**节点访问顺序**
      · `values`  —— 每个 super-step 之后的**累计状态**，用来取终态

    为什么不自己 `state.update(delta)`：`past_observations` 是 `operator.add` 归约字段，
    手工 update 会把它当覆盖写，终态就错了。这是必须交给 LangGraph 做的事。

    :param thread_id: 挂了 checkpointer 时必须给（否则 LangGraph 会报缺 thread_id）
    """
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None
    visited: List[str] = []
    final: Dict[str, Any] = {}
    stream = (agent.stream(initial_state(user_query), config,
                           stream_mode=["updates", "values"])
              if config else
              agent.stream(initial_state(user_query), stream_mode=["updates", "values"]))
    for mode, chunk in stream:
        if mode == "updates":
            visited.extend(chunk.keys())
        else:
            final = chunk
    return visited, final
