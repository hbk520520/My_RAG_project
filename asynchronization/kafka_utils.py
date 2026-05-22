"""
Kafka 消息工具 —— Worker 之间靠它在不直接见面的情况下传递任务
========================================================
五个 Topic 对应五个 Worker 角色，Producer 打 gzip 压缩发消息，
Consumer 手动提交位移保证至少处理一次。

技术栈: kafka-python (KafkaProducer / KafkaConsumer)
"""
from kafka import KafkaConsumer, KafkaProducer
import json
import os
import sys

# 支持从 config_loader 导入（处理路径问题）
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from config_loader import cfg as _cfg
except Exception:
    _cfg = None

# Kafka 主题名称常量
TOPIC_PLANNER_PENDING = "topic.planner.pending"
TOPIC_RETRIEVER_PENDING = "topic.retriever.pending"
TOPIC_GRADER_PENDING = "topic.grader.pending"
TOPIC_REPLANNER_PENDING = "topic.replanner.pending"
TOPIC_REASONER_PENDING = "topic.reasoner.pending"

_DEFAULT_BOOTSTRAP = "localhost:9092"


def _get_bootstrap():
    if _cfg is not None:
        return _cfg.get("kafka", "bootstrap_servers", default=_DEFAULT_BOOTSTRAP)
    return os.environ.get("KAFKA_BOOTSTRAP", _DEFAULT_BOOTSTRAP)


def create_producer(bootstrap_servers: str = None) -> KafkaProducer:
    if bootstrap_servers is None:
        bootstrap_servers = _get_bootstrap()
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: k.encode('utf-8') if k else None,
        acks='all',
        compression_type='gzip',
        max_in_flight_requests_per_connection=5
    )


def create_consumer(topic: str, group_id: str, bootstrap_servers: str = None) -> KafkaConsumer:
    if bootstrap_servers is None:
        bootstrap_servers = _get_bootstrap()
    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        key_deserializer=lambda k: k.decode('utf-8') if k else None,
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=False,
        max_poll_records=1
    ) 