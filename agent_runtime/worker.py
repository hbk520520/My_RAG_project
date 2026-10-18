"""Run with python -m agent_runtime.worker on the same host as the run database."""
import signal
import threading

from config_loader import cfg


def main():
    from asynchronization.kafka_utils import create_producer, quarantine_message
    from kafka import KafkaConsumer
    import json
    import logging
    import sqlite3
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
        value_deserializer=None)
    producer = create_producer(bootstrap_servers=cfg.get("kafka", "bootstrap_servers"))
    service = AgentService()
    transport = KafkaTransport(service, producer)
    def execute():
        while not stop.wait(0.2):
            try:
                service.recover()
            except sqlite3.Error:
                logger.exception("Run storage unavailable; queued work remains durable")
    worker = threading.Thread(target=execute, name="agent-kafka-executor", daemon=True)
    worker.start()
    try:
        while not stop.is_set():
            if not worker.is_alive():
                raise RuntimeError("Agent executor stopped")
            transport.dispatch()
            for messages in consumer.poll(timeout_ms=1000, max_records=1).values():
                for message in messages:
                    try:
                        transport.accept(json.loads(message.value.decode("utf-8")))
                    except (KeyError, ValueError) as exc:
                        # Durable quarantine must succeed before committing the offset.
                        quarantine_message(message, exc, logger)
                    consumer.commit()
    finally:
        stop.set()
        worker.join(timeout=1)
        consumer.close()
        producer.close()


if __name__ == "__main__":
    main()
