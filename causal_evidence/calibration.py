"""Held-out, source-grouped isotonic calibration of textual support labels."""
from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
import math
from datetime import date
from typing import Literal

from pydantic import Field, model_validator

from .models import Contract


class SupportLabel(Contract):
    source_id: str = Field(min_length=1)
    raw_support: float = Field(gt=0, le=1)
    supported: Literal[0, 1]


class SupportCalibrator(Contract):
    version: Literal[1] = 1
    extractor_id: str = Field(min_length=1)
    x: list[float] = Field(min_length=1)
    y: list[float] = Field(min_length=1)
    training_sources: list[str]
    evaluation_sources: list[str]
    labels_sha256: str
    trained_through: date

    @model_validator(mode="after")
    def valid_curve(self):
        if len(self.x) != len(self.y):
            raise ValueError("Calibration knot lengths differ")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in self.x + self.y):
            raise ValueError("Calibration knots must be finite probabilities")
        if any(a >= b for a, b in zip(self.x, self.x[1:])):
            raise ValueError("Calibration x must be strictly increasing")
        if self.y != sorted(self.y):
            raise ValueError("Calibration must be monotone")
        if not self.training_sources or not self.evaluation_sources:
            raise ValueError("Both train and held-out sources are required")
        if set(self.training_sources) & set(self.evaluation_sources):
            raise ValueError("Calibration source leakage")
        return self

    def predict(self, score: float) -> float:
        if not math.isfinite(score) or not 0 < score <= 1:
            raise ValueError("raw_support must be in (0, 1]")
        if score <= self.x[0]:
            return self.y[0]
        if score >= self.x[-1]:
            return self.y[-1]
        right = bisect_right(self.x, score)
        left = right - 1
        fraction = (score - self.x[left]) / (self.x[right] - self.x[left])
        return self.y[left] + fraction * (self.y[right] - self.y[left])


def calibration_metrics(scores: list[float], labels: list[int], bins: int = 10) -> dict:
    if not scores or len(scores) != len(labels) or bins < 1:
        raise ValueError("Require nonempty aligned scores/labels and bins >= 1")
    if any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores):
        raise ValueError("Scores must be finite probabilities")
    if any(y not in (0, 1) for y in labels):
        raise ValueError("Labels must be binary")
    buckets = [[] for _ in range(bins)]
    for score, label in zip(scores, labels):
        buckets[min(int(score * bins), bins - 1)].append((score, label))
    ece = sum(abs(sum(s - y for s, y in bucket)) for bucket in buckets) / len(scores)
    return {"n": len(scores), "brier": sum((s-y)**2 for s, y in zip(scores, labels)) / len(scores),
            "ece": ece, "ece_bins": bins}


def fit_calibration(rows: list[SupportLabel], *, extractor_id: str,
                    trained_through: date, holdout_fraction: float = 0.25) -> tuple[SupportCalibrator, dict]:
    """Fit on training sources only. Never refit on held-out labels after evaluation."""
    from sklearn.isotonic import IsotonicRegression

    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in (0, 1)")
    sources = sorted({r.source_id for r in rows}, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    if len(sources) < 4:
        raise ValueError("Need at least four independent source groups")
    count = max(1, math.ceil(len(sources) * holdout_fraction))
    test_sources = set(sources[:count])
    train = [r for r in rows if r.source_id not in test_sources]
    test = [r for r in rows if r.source_id in test_sources]
    if len({r.supported for r in train}) != 2 or len({r.raw_support for r in train}) < 2:
        raise ValueError("Training split needs both labels and at least two distinct scores")
    estimator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    estimator.fit([r.raw_support for r in train], [r.supported for r in train])
    serialized = json.dumps([r.model_dump() for r in rows], sort_keys=True).encode()
    model = SupportCalibrator(
        extractor_id=extractor_id, x=estimator.X_thresholds_.tolist(),
        y=estimator.y_thresholds_.tolist(), training_sources=sorted(set(sources)-test_sources),
        evaluation_sources=sorted(test_sources), labels_sha256=hashlib.sha256(serialized).hexdigest(),
        trained_through=trained_through,
    )
    labels = [r.supported for r in test]
    report = {"split": "sha256-source-grouped", "training_n": len(train),
              "raw": calibration_metrics([r.raw_support for r in test], labels),
              "calibrated": calibration_metrics([model.predict(r.raw_support) for r in test], labels)}
    return model, report
