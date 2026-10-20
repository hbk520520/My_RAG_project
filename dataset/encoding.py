"""Bounded dense/sparse encoding batches, returned in the original text order."""
from __future__ import annotations

import logging
import time

import numpy as np


logger = logging.getLogger(__name__)


def encode_batches(texts, encode, dimension: int, *, batch_size: int = 32) -> dict:
    """Call a list-based encoder without changing row IDs or dropping sparse data.

    ``encode`` must return BGE-M3's dense_vecs and lexical_weights fields.
    Single-text callables belong to the existing encode_text/vectorize paths.
    Length grouping uses character counts; it does not change tokenization.
    """
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("encoding batch_size must be a positive integer")
    if isinstance(texts, str):
        raise TypeError("Batch encoding requires a sequence of texts")
    texts = list(texts)
    if any(not isinstance(text, str) for text in texts):
        raise TypeError("Every encoding input must be a string")
    dense = np.empty((len(texts), dimension), dtype=np.float32)
    sparse = [{} for _ in texts]
    order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
    started = time.perf_counter()
    batches = 0
    for start in range(0, len(order), batch_size):
        positions = order[start:start + batch_size]
        output = encode([texts[index] for index in positions])
        vectors = np.asarray(output["dense_vecs"], dtype=np.float32)
        weights = output["lexical_weights"]
        if vectors.shape != (len(positions), dimension):
            raise ValueError(
                f"Batch encoder returned shape {vectors.shape}; "
                f"expected {(len(positions), dimension)}")
        if not np.isfinite(vectors).all():
            raise ValueError("Batch encoder returned non-finite vectors")
        if len(weights) != len(positions) or any(not isinstance(row, dict) for row in weights):
            raise ValueError("Batch encoder must return one sparse mapping per text")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors.copy()
        np.divide(vectors, norms, out=normalized, where=norms > 0)
        dense[positions] = normalized
        for offset, index in enumerate(positions):
            sparse[index] = dict(weights[offset])
        batches += 1
    logger.info("batch_encoding texts=%d batches=%d batch_size=%d seconds=%.6f",
                len(texts), batches, batch_size, time.perf_counter() - started)
    return {"dense_vecs": dense, "lexical_weights": sparse}
