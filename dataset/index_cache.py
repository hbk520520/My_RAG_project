"""Bounded process-local FAISS artifacts; callers always receive private indexes."""
from __future__ import annotations

from collections import OrderedDict
import logging
import threading
import time

import numpy as np


logger = logging.getLogger(__name__)


class SnapshotIndexCache:
    """Cache trusted, in-process HNSW builds rather than mutable graph objects.

    A key identifies a project and the complete canonical snapshot digest.
    Serialized artifacts never go to disk and are not accepted from callers.
    Count and byte limits bound retained artifacts, not active run indexes.
    """

    def __init__(self, *, max_entries: int = 2, max_bytes: int = 268435456):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("snapshot_cache_entries must be a positive integer")
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("snapshot_cache_max_bytes must be a positive integer")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get_or_build(self, key, build):
        import faiss

        started = time.perf_counter()
        # Serialize cold builds so concurrent starts of the same snapshot do
        # not build it twice. The cache is intentionally small and CPU-only.
        with self._lock:
            artifact = self._entries.get(key)
            if artifact is not None:
                self._entries.move_to_end(key)
            else:
                index = build()
                serialized = faiss.serialize_index(index)
                size = serialized.nbytes
                retained = size <= self.max_bytes
                if retained:
                    artifact = serialized.tobytes()
                    while (len(self._entries) >= self.max_entries
                           or self._bytes + size > self.max_bytes):
                        _, evicted = self._entries.popitem(last=False)
                        self._bytes -= len(evicted)
                    self._entries[key] = artifact
                    self._bytes += size
                logger.info("snapshot_index_cache hit=false retained=%s bytes=%d seconds=%.6f",
                            retained, size, time.perf_counter() - started)
                # The original built index is private too: only bytes remain
                # in the cache, so mutations in this run cannot leak to others.
                return index
        index = faiss.deserialize_index(np.frombuffer(artifact, dtype=np.uint8))
        logger.info("snapshot_index_cache hit=true bytes=%d seconds=%.6f",
                    len(artifact), time.perf_counter() - started)
        return index
