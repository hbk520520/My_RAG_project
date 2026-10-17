"""Fit source-grouped support calibration; no LLM calls and no synthetic training labels."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causal_evidence.calibration import SupportLabel, fit_calibration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path, help="JSONL: source_id, raw_support, supported (0/1)")
    parser.add_argument("--extractor-id", required=True, help="Extractor/model/prompt version identifier")
    parser.add_argument("--trained-through", required=True, type=date.fromisoformat,
                        help="Latest date on which any calibration annotation became available")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    args = parser.parse_args(argv)
    rows = [SupportLabel.model_validate_json(line) for line in args.labels.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    model, report = fit_calibration(rows, extractor_id=args.extractor_id,
                                   trained_through=args.trained_through,
                                   holdout_fraction=args.holdout_fraction)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps({"artifact": str(args.out), "held_out": report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
