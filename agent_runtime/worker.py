"""Run with python -m agent_runtime.worker on the same host as the run database."""
import signal
import threading

from config_loader import cfg


def main():
    from asynchronization.kafka_utils import create_producer, quarantine_message
    from kafka import KafkaConsumer
    import json
    import logging
    from observability import setup_structured_logging
    from .service import AgentService
    from .transport import KafkaTransport
    setup_structured_logging()
    logger = logging.getLogger(__name__)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *args: stop.set())
    consumer = KafkaConsumer(
        cfg.get("runtime", "pending_topic"), bootstrap_servers=cfg.get("kafka", "bootstrap_servers"),
        group_id="agent-runtime-v1", enable_auto_commit=False, max_poll_records=1,
        max_poll_interval_ms=900000, auto_offset_reset="earliest",
        value_deserializer=lambda raw: json.loads(raw.decode("utf-8")))
    producer = create_producer(bootstrap_servers=cfg.get("kafka", "bootstrap_servers"))
    service = AgentService()
    transport = KafkaTransport(service, producer)
    try:
        while not stop.is_set():
            service.recover()
            transport.dispatch()
            for messages in consumer.poll(timeout_ms=1000, max_records=1).values():
                for message in messages:
                    try:
                        transport.receive(message.value)
                    except (KeyError, ValueError) as exc:
                        # Durable quarantine must succeed before committing the offset.
                        quarantine_message(message, exc, logger)
                    consumer.commit()
    finally:
        consumer.close()
        producer.close()


if __name__ == "__main__":
    main()
