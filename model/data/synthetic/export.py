"""Streaming JSONL exports with a completion manifest and file digests."""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import tempfile

from .environment import resolve, score_trajectory
from .generator import generate, preference_pair, trajectories
from .schemas import GenerationConfig, VERSION


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def export_dataset(config: GenerationConfig, output: str | Path):
    config = GenerationConfig.model_validate(config)
    output = Path(output).resolve()
    if output.drive and output.drive.casefold() != "d:":
        raise ValueError("Synthetic artifacts must stay on D:")
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing datasets are never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts, files = Counter(), {}
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".synthetic-") as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        with ExitStack() as stack:
            streams = {}
            for split in ("train", "validation", "test", "ood"):
                for kind in ("tasks", "grader_sft", "grader_dpo", "trajectories", "ranking"):
                    name = f"{split}.{kind}.jsonl"
                    streams[name] = stack.enter_context((staging / name).open("w", encoding="utf-8", newline="\n"))
                    files[name] = {"rows": 0, "digest": hashlib.sha256()}

            def write(split, kind, record):
                name = f"{split}.{kind}.jsonl"
                line = dumps(record) + "\n"
                streams[name].write(line)
                files[name]["digest"].update(line.encode("utf-8"))
                files[name]["rows"] += 1

            for sample in generate(config):
                if sample.expected != resolve(sample.documents, sample.record_ids, missing_user=sample.family == "missing"):
                    raise ValueError("Fixture ground truth does not match its environment")
                metadata = {"sample_id": sample.sample_id, "group_id": sample.group_id,
                            "split": sample.split, "synthetic": True, "generator_version": VERSION}
                write(sample.split, "tasks", sample.model_dump(mode="json"))
                # Input excludes expected, family, transforms and simulator-private failure state.
                from prompts import build_grader_messages
                messages = build_grader_messages(sample.query, dumps([d.model_dump(mode="json") for d in sample.documents]))
                messages[0]["content"] += ('\n若缺失信息须由用户提供，使用 missing_info_kind="user"；'
                                            '若需补充证据，使用 missing_info_kind="evidence"。')
                chosen, rejected = preference_pair(sample)
                write(sample.split, "grader_sft", {**metadata, "messages": messages + [
                    {"role": "assistant", "content": dumps(chosen)}]})
                write(sample.split, "grader_dpo", {**metadata, "prompt": "\n\n".join(m["content"] for m in messages),
                                                  "chosen": dumps(chosen), "rejected": dumps(rejected)})
                scored = []
                for trajectory in trajectories(sample):
                    score = score_trajectory(sample, trajectory)
                    if trajectory.name == "reference" and not (score["steps_correct"] and score["final_correct"]):
                        raise ValueError("Reference trajectory failed replay")
                    write(sample.split, "trajectories", {**metadata, "trajectory": trajectory.model_dump(mode="json"),
                                                         "score": score})
                    counts[f"trajectory:{trajectory.name}"] += 1
                    counts["trajectory:final_correct"] += int(score["final_correct"])
                    counts["trajectory:steps_invalid"] += int(not score["steps_correct"])
                    scored.append((trajectory.name, score["reward"]))
                for i, (left, reward) in enumerate(scored):
                    for right, other in scored[i+1:]:
                        if reward != other:
                            write(sample.split, "ranking", {**metadata, "chosen": left if reward > other else right,
                                                           "rejected": right if reward > other else left,
                                                           "reward_gap": round(abs(reward-other), 6)})
                counts[f"split:{sample.split}"] += 1
                counts[f"family:{sample.family}"] += 1
                counts[f"records:{len(sample.record_ids)}"] += 1
                counts[f"topic:{sample.topic_rank}"] += 1
                counts[f"status:{sample.expected.status}"] += 1
                for transform in sample.transforms:
                    counts[f"transform:{transform}"] += 1
        manifest = {"generator_version": VERSION, "synthetic": True, "status": "complete",
                    "config": config.model_dump(mode="json"), "counts": dict(sorted(counts.items())),
                    "files": {name: {"rows": value["rows"], "sha256": value["digest"].hexdigest()}
                              for name, value in files.items()},
                    "trajectory_origin": "scripted_reference", "training_executed": False,
                    "split_policy": "entity/source group before augmentation; OOD uses new groups and renamed tools",
                    "scope": "Fictional evidence and arithmetic; no legal conclusions or model-quality claims"}
        (staging / "manifest.json").write_text(dumps(manifest) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError("Output directory appeared during generation")
        os.rename(staging, output)
    return manifest
