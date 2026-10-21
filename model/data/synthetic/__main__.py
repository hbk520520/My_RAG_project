"""Offline CLI; model providers can only be explicitly injected through the Python API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import TypeAdapter

from .contracts import Config, SeedGroup
from .export import export_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate verified fixtures and separately stored pending semantic drafts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--seeds", type=Path, help="Optional JSON array of source groups; semantic drafts only")
    parser.add_argument("--trajectories", action="store_true", help="Also export scripted candidates and scores")
    args = parser.parse_args(argv)
    config = Config.model_validate_json(args.config.read_text(encoding="utf-8")) if args.config else Config()
    if args.trajectories:
        config = config.model_copy(update={"include_trajectories": True})
    seeds = TypeAdapter(list[SeedGroup]).validate_json(args.seeds.read_text(encoding="utf-8")) if args.seeds else []
    manifest = export_dataset(config, args.output, seed_groups=seeds)
    print(json.dumps({"output": str(args.output.resolve()), "counts": manifest["counts"],
                      "candidate_provider_calls": manifest["candidate_provider_calls"], "training_executed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
