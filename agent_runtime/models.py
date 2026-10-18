"""One serialized encoder owner per model/version/device in this process."""
import threading
from collections import OrderedDict
from config_loader import cfg
from .resources import resource, settings

_encoders = OrderedDict()
_lock = threading.Lock()


class SharedEncoder:
    def __init__(self, name, fp16, device, encoder=None):
        self.name, self.fp16, self.device = name, fp16, device
        self._encoder = encoder
        self._lock = threading.Lock()

    def encode(self, texts, **kwargs):
        # Serialize a non-thread-safe model; branches can still overlap network
        # calls and CPU work. GPU batching belongs to the inference backend.
        with self._lock, resource("encoder:" + self.device, settings().encoder_slots):
            if self._encoder is None:
                from FlagEmbedding import BGEM3FlagModel
                if self.device.startswith("cuda"):
                    from .resources import accelerator_owner
                    accelerator_owner()
                self._encoder = BGEM3FlagModel(self.name, use_fp16=self.fp16, devices=self.device)
            return self._encoder.encode(texts, **kwargs)


def shared_encoder(engine):
    device = cfg.get("graph", "device", default=cfg.get("embedding", "device", default="cpu"))
    # A custom injected encoder has its own identity; it must never be silently
    # replaced by a different project's test or application-provided encoder.
    injected = engine._encoder
    key = (engine.model_name, engine.use_fp16, device, engine.dim,
           engine._memory_space_id, id(injected) if injected is not None else None)
    with _lock:
        if key not in _encoders:
            _encoders[key] = SharedEncoder(engine.model_name, engine.use_fp16, device, injected)
        _encoders.move_to_end(key)
        while len(_encoders) > settings().model_cache_entries:
            _encoders.popitem(last=False)
        return _encoders[key]
