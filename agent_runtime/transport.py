"""Kafka delivery is at-least-once; SQLite owns run state and deduplication."""
from pydantic import BaseModel, ConfigDict, Field

from .contracts import RunConflict


class Envelope(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = Field(default=1, ge=1, le=1)
    event_id: str = Field(min_length=1)
    run_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    epoch: int = Field(ge=0)


class KafkaTransport:
    def __init__(self, service, producer):
        self.service = service
        self.producer = producer

    def dispatch(self):
        count = 0
        for event in self.service.store.pending_outbox():
            # Waiting for each broker acknowledgement is necessary before marking sent.
            self.producer.send(event["topic"], key=event["payload"]["run_id"],
                               value=event["payload"]).get(timeout=30)
            self.service.store.acknowledge_outbox(event["event_id"])
            count += 1
        return count

    def receive(self, value):
        event = Envelope.model_validate(value)
        return self.service.execute(event.run_id, epoch=event.epoch)
