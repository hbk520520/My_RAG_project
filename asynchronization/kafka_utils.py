from kafka import KafkaConsumer, KafkaProducer
import json

# Kafka 主题名称常量
TOPIC_PLANNER_PENDING = "topic.planner.pending"
TOPIC_RETRIEVER_PENDING = "topic.retriever.pending"
TOPIC_GRADER_PENDING = "topic.grader.pending"
TOPIC_REPLANNER_PENDING = "topic.replanner.pending"
TOPIC_REASONER_PENDING = "topic.reasoner.pending"

def create_producer(bootstrap_servers: str = "localhost:9092") -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: k.encode('utf-8') if k else None,
        acks='all',
        compression_type='gzip',
        max_in_flight_requests_per_connection=5
    )

def create_consumer(topic: str, group_id: str, bootstrap_servers: str = "localhost:9092") -> KafkaConsumer:
    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        key_deserializer=lambda k: k.decode('utf-8') if k else None,
        value_deserializer=lambda v: json.loads(v.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=False,  # 手动提交，保证至少一次处理
        max_poll_records=1
    ) 