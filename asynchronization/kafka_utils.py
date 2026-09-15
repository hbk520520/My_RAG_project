"""
Kafka 消息工具 —— Worker 之间靠它在不直接见面的情况下传递任务
========================================================
五个 Topic 对应五个 Worker 角色，Producer 打 gzip 压缩发消息，
Consumer 手动提交位移保证至少处理一次。

技术栈: kafka-python (KafkaProducer / KafkaConsumer)
"""
from kafka import KafkaConsumer, KafkaProducer
import json
import logging
import os
import sys

# 支持从 config_loader 导入（处理路径问题）
# 注意：本文件在 asynchronization/ 下，而 config_loader.py 在项目根目录，
# 所以必须把「根目录」也加进 sys.path。之前只加了 asynchronization/，
# 导致 import 静默失败、_cfg 恒为 None，config.yaml 的 kafka 段被完全忽略。
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in (_ROOT_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from config_loader import cfg as _cfg
except Exception:  # 允许脱离 config.yaml 独立运行，回落到环境变量
    _cfg = None

def _cfg_get(*keys, default=None):
    """带兜底的配置读取（_cfg 为 None 表示脱离 config.yaml 独立运行）"""
    if _cfg is None:
        return default
    return _cfg.get(*keys, default=default)


# ---- Kafka 主题名称（阶段 2：改读 config.yaml 的 kafka.topics.*，字面量仅兜底）----
TOPIC_PLANNER_PENDING = _cfg_get("kafka", "topics", "planner_pending", default="topic.planner.pending")
TOPIC_RETRIEVER_PENDING = _cfg_get("kafka", "topics", "retriever_pending", default="topic.retriever.pending")
TOPIC_GRADER_PENDING = _cfg_get("kafka", "topics", "grader_pending", default="topic.grader.pending")
TOPIC_REPLANNER_PENDING = _cfg_get("kafka", "topics", "replanner_pending", default="topic.replanner.pending")
TOPIC_REASONER_PENDING = _cfg_get("kafka", "topics", "reasoner_pending", default="topic.reasoner.pending")

# ---- 消费者组名（阶段 2：改读 config.yaml 的 kafka.consumer.groups.*）----
GROUP_PLANNER = _cfg_get("kafka", "consumer", "groups", "planner", default="planner-group")
GROUP_RETRIEVER = _cfg_get("kafka", "consumer", "groups", "retriever", default="retriever-group")
GROUP_GRADER = _cfg_get("kafka", "consumer", "groups", "grader", default="grader-group")
GROUP_REPLANNER = _cfg_get("kafka", "consumer", "groups", "replanner", default="replanner-group")
GROUP_REASONER = _cfg_get("kafka", "consumer", "groups", "reasoner", default="reasoner-group")

_DEFAULT_BOOTSTRAP = "localhost:9092"


def _get_bootstrap():
    return _cfg_get("kafka", "bootstrap_servers") or os.environ.get("KAFKA_BOOTSTRAP", _DEFAULT_BOOTSTRAP)


def create_producer(bootstrap_servers: str = None) -> KafkaProducer:
    """创建 Producer（阶段 2：acks / 压缩 / 在途请求数改读 config.yaml）"""
    if bootstrap_servers is None:
        bootstrap_servers = _get_bootstrap()
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: k.encode('utf-8') if k else None,
        acks=_cfg_get("kafka", "producer", "acks", default="all"),
        compression_type=_cfg_get("kafka", "producer", "compression_type", default="gzip"),
        max_in_flight_requests_per_connection=_cfg_get(
            "kafka", "producer", "max_in_flight", default=5),
    )


def create_consumer(topic: str, group_id: str, bootstrap_servers: str = None) -> KafkaConsumer:
    """
    创建 Consumer（阶段 2：位移提交策略改读 config.yaml，保证至少处理一次）。

    注意：调用方的消息循环**必须**自己把单条消息的处理包在 try/except 里，
    否则一条毒消息会直接终结 Worker 进程、且位移未提交会被反复重投。
    统一做法见各 Worker 的 main()：捕获 -> 写 DLQ -> 提交位移 -> 继续。
    """
    if bootstrap_servers is None:
        bootstrap_servers = _get_bootstrap()
    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        key_deserializer=lambda k: k.decode('utf-8') if k else None,
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        auto_offset_reset=_cfg_get("kafka", "consumer", "auto_offset_reset", default="earliest"),
        enable_auto_commit=_cfg_get("kafka", "consumer", "enable_auto_commit", default=False),
        max_poll_records=_cfg_get("kafka", "consumer", "max_poll_records", default=1),
    )


# ============================================================================
# 消息处理失败的统一收容（阶段 6）
# ============================================================================
def quarantine_message(msg, error, logger=None):
    """
    收容一条处理失败的消息：写 DLQ + 提交位移，然后返回，让调用方继续下一条。

    为什么必须显式调用而不用生成器包装：
      在 `for msg in gen():` 里，循环体抛出的异常**不会**从 gen 的 yield 处
      抛回生成器（那需要显式 gen.throw()）。所以"用一个包装迭代器自动捕获
      循环体异常"的做法在 Python 里行不通，只能在循环体内显式 try/except。

    调用方用法：
        for msg in consumer:
            session_id = ...
            try:
                ... 业务处理 ...
            except Exception as e:
                quarantine_message(msg, e, logger)
                continue
    """
    log = logger or logging.getLogger("kafka_utils")
    topic = getattr(msg, "topic", "") or ""
    key = getattr(msg, "key", "") or ""
    value = getattr(msg, "value", None)

    error_text = f"{type(error).__name__}: {error}"
    log.error(f"消息处理失败，写入 DLQ 并跳过: topic={topic} key={key} {error_text}",
              exc_info=True)

    try:
        from observability import push_to_dlq
        push_to_dlq(topic, str(key),
                    value if isinstance(value, dict) else {"raw": str(value)},
                    error_text)
    except Exception as de:
        log.error(f"DLQ 写入不可用: {de}") 