"""Build PA-Tool mappings from saved samples or an explicitly selected local model."""
import argparse
import json
import os
from pathlib import Path

from agent_tools.schema_alignment import CATALOGS, build_profile, sample_names


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=CATALOGS, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--samples", type=Path, help="Saved reference/candidate names; makes no model calls")
    source.add_argument("--generate", action="store_true", help="Explicitly sample the configured local role model")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--alpha", type=float, default=0.2)
    args = parser.parse_args(argv)
    if args.count < 1 or args.temperature <= 0 or not 0 <= args.alpha <= 1:
        parser.error("count/temperature must be positive and alpha must be in [0, 1]")
    if args.output.exists():
        parser.error("output already exists")
    if args.generate:
        from config_loader import cfg
        if cfg.get("llm", "backend") != "vllm":
            parser.error("--generate requires the explicitly configured vllm backend")
        if not Path(cfg.get("vllm", "base_model")).is_dir():
            parser.error("--generate requires existing local model files")
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        from legal_agent.operators import AgenticNodesOperator
        ops = AgenticNodesOperator()
        records = sample_names(args.role, lambda messages, **kw: ops._role_messages(
            args.role, messages, require_json=True, **kw), count=args.count, temperature=args.temperature)
    else:
        records = json.loads(args.samples.read_text(encoding="utf-8"))
    profile = build_profile(args.role, records, alpha=args.alpha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(profile, stream, ensure_ascii=False, indent=2)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
