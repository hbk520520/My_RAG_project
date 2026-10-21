"""Stream verified fixtures and pending semantic drafts into separate files."""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile

import pydantic

from .candidates import KINDS, CandidateProvider, pending_candidate
from .contracts import Config, SeedGroup, Source, VERSION
from .generation import generate_groups, split_for, trajectory_candidates
from .simulation import oracle, reference, score


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def validate_seeds(seed_groups):
    owners, group_ids, result = {}, set(), []
    for raw in seed_groups:
        seed = SeedGroup.model_validate(raw)
        if seed.id in group_ids or len({s.id for s in seed.sources}) != len(seed.sources):
            raise ValueError("Seed groups and their source IDs must be unique")
        group_ids.add(seed.id)
        for source in seed.sources:
            if source.id.startswith("fixture-"):
                raise ValueError("fixture- source IDs are reserved for generated environments")
            for key in ("id:" + source.id, "text:" + hashlib.sha256(source.text.strip().encode()).hexdigest()):
                if key in owners and owners[key] != seed.id:
                    raise ValueError("Shared seed sources must be placed in the same group")
                owners[key] = seed.id
        result.append(seed)
    return result


def export_dataset(config: Config, output, *, provider: CandidateProvider | None = None, seed_groups=()):
    config = Config.model_validate(config)
    if config.candidate_call_limit and provider is None:
        raise ValueError("A positive candidate_call_limit requires an explicitly injected provider")
    output = Path(output).resolve()
    if output.drive and output.drive.casefold() != "d:":
        raise ValueError("Keep generated files on D:")
    if output.exists():
        raise FileExistsError("Use a new directory; existing datasets are never overwritten")
    seeds = validate_seeds(seed_groups)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts, files, calls = Counter(), {}, 0
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".synthetic-") as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        with ExitStack() as stack:
            kinds = ["verified", "pending"] + (["trajectories", "ranking"] if config.include_trajectories else [])
            names = [f"{split}.{kind}.jsonl" for split in ("train", "validation", "test", "ood") for kind in kinds]
            names.append("group_exposures.jsonl")
            streams = {name: stack.enter_context((staging / name).open("w", encoding="utf-8", newline="\n")) for name in names}
            files = {name: {"rows": 0, "hash": hashlib.sha256()} for name in names}

            def write(name, row):
                line = dumps(row) + "\n"
                streams[name].write(line)
                files[name]["hash"].update(line.encode("utf-8"))
                files[name]["rows"] += 1

            def pending(group, split, sources, ordinal):
                nonlocal calls
                for i in range(config.pending_per_group):
                    selected = provider if calls < config.candidate_call_limit else None
                    if selected is not None:
                        calls += 1
                    row = pending_candidate(group, split, KINDS[(ordinal + i) % len(KINDS)], sources, selected)
                    write(f"{split}.pending.jsonl", row)
                    counts["pending"] += 1
                    counts[f"pending_provider:{row['provenance']['provider']}"] += 1

            # Supplied seed material gets the optional provider budget before fixture templates.
            for ordinal, seed in enumerate(seeds):
                group = "seed-" + seed.id
                pending(group, split_for(group, config), seed.sources, ordinal)
                counts["seed_groups"] += 1

            for ordinal, (group, split, tasks) in enumerate(generate_groups(config), start=len(seeds)):
                exposures, reads = Counter(), Counter()
                for task in tasks:
                    gold = reference(task)
                    report = score(task, gold)
                    if not all(report[key] for key in ("final_correct", "steps_correct", "state_correct")):
                        raise ValueError(f"Reference replay failed: {task.id}")
                    write(f"{split}.verified.jsonl", {
                        "schema_version": VERSION, "id": task.id, "group_id": group, "split": split, "synthetic": True,
                        "family": task.family, "input": task.public_input(),
                        "environment": {"kind": "in_memory_fixture", "task": task.model_dump(mode="json")},
                        "label": oracle(task).model_dump(mode="json"),
                        "verification": {"status": "verified", "method": "reference_replay", "checks": report},
                        "provenance": {"generator": VERSION, "seed": config.seed, "transforms": task.transforms}})
                    exposures.update({s.id for r in task.records for s in r.sources})
                    for step in gold.steps:
                        if step.tool == task.tool_names["read"]:
                            reads.update(s["id"] for s in step.observation.get(task.response_format, []))
                    counts["verified"] += 1
                    counts[f"split:{split}"] += 1
                    counts[f"family:{task.family}"] += 1
                    counts[f"outcome:{gold.answer.status}"] += 1
                    counts[f"records:{len(task.records)}"] += 1
                    for transform in task.transforms:
                        counts[f"transform:{transform}"] += 1
                    if config.include_trajectories:
                        ranked = []
                        for trajectory in trajectory_candidates(task):
                            result = score(task, trajectory)
                            identity = f"{task.id}:{trajectory.name}"
                            write(f"{split}.trajectories.jsonl", {"task_id": task.id, "group_id": group, "split": split,
                                "id": identity, "synthetic": True, "trajectory": trajectory.model_dump(mode="json"), "score": result})
                            counts["trajectories"] += 1
                            ranked.append((identity, result["reward"]))
                        for i, (left, reward) in enumerate(ranked):
                            for right, other in ranked[i+1:]:
                                if reward != other:
                                    write(f"{split}.ranking.jsonl", {"task_id": task.id, "group_id": group, "split": split,
                                        "synthetic": True, "chosen": left if reward > other else right,
                                        "rejected": right if reward > other else left, "reward_gap": round(abs(reward-other), 6)})
                write("group_exposures.jsonl", {"group_id": group, "split": split,
                    "task_count": len(tasks), "source_occurrences": dict(sorted(exposures.items())),
                    "reference_source_reads": dict(sorted(reads.items()))})
                counts[f"group_exposures:{len(tasks)}"] += 1
                sources = [Source(id=s.id, text="\n".join(s.statements)) for r in tasks[0].records for s in r.sources]
                pending(group, split, sources, ordinal)

        manifest = {"schema_version": VERSION, "synthetic": True, "status": "complete",
                    "runtime": {"python": platform.python_version(), "pydantic": pydantic.__version__},
                    "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(Path(__file__).parent.glob("*.py"))},
                    "config": config.model_dump(mode="json"), "counts": dict(sorted(counts.items())),
                    "candidate_provider_calls": calls, "training_executed": False,
                    "files": {name: {"rows": meta["rows"], "sha256": meta["hash"].hexdigest()} for name, meta in files.items()},
                    "verified_scope": "In-memory task semantics only; semantic drafts are never promoted",
                    "split_policy": "Source/entity groups before variants; independent OOD namespace",
                    "reproducibility": "Seeded fixtures; provider responses captured but not assumed deterministic"}
        (staging / "manifest.json").write_text(dumps(manifest) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError("Output directory appeared during generation")
        os.rename(staging, output)
    return manifest
