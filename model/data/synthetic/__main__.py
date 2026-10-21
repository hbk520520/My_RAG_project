"""python -m model.data.synthetic --output D:/... [--config configuration.json]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .export import export_dataset
from .schemas import GenerationConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate independently replayed synthetic evidence fixtures")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="GenerationConfig JSON; omitted means clean baseline")
    args = parser.parse_args(argv)
    config = GenerationConfig.model_validate_json(args.config.read_text(encoding="utf-8")) if args.config else GenerationConfig()
    manifest = export_dataset(config, args.output)
    print(json.dumps({"output": str(args.output.resolve()), "counts": manifest["counts"],
                      "training_executed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
