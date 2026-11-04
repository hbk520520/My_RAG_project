"""Optional, offline Heretic adapter preparation. No ML imports or downloads."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HERETIC_VERSION = "2.0.0"
MARKERS = {
    "nosyco": ["您的想法非常出色", "您的判断完全正确", "您说得太对了", "您真有远见"],
    "noslop": ["下面我将详细为您介绍", "让我们一起深入探讨", "希望以上内容对您有所帮助"],
}


def _table(name, values, *, array=False):
    heading = f"[[{name}]]" if array else f"[{name}]" if name else ""
    return heading + "\n" + "\n".join(
        f"{key} = {json.dumps(value, ensure_ascii=False)}" for key, value in values.items()
    ) + "\n"


def prepare(model, good, bad, evaluation, output, *, trials, goal="both",
            markers=None, max_tokens=2048, seed=42):
    """Snapshot local contrast/optimization prompts and produce a Heretic 2 config.

    Evaluation prompts here participate in optimization; they are NOT a held-out
    task-quality test. Good/bad prompts must elicit contrasting *behaviors*.
    """
    model, output = Path(model).resolve(), Path(output).resolve()
    if not (model / "config.json").is_file():
        raise ValueError("model must be a local full model with config.json, not a LoRA-only directory")
    if output.is_relative_to(model):
        raise ValueError("output must be outside the original model directory")
    if trials < 1 or max_tokens < 1:
        raise ValueError("trials and max_tokens must be positive")
    goals = ("nosyco", "noslop") if goal == "both" else (goal,)
    if any(item not in MARKERS for item in goals):
        raise ValueError("goal must be nosyco, noslop, or both")
    markers = markers if markers is not None else [word for item in goals for word in MARKERS[item]]
    if not isinstance(markers, list) or not markers or any(
            not isinstance(word, str) or not word.strip() for word in markers):
        raise ValueError("markers must be a nonempty JSON list of nonempty strings")
    prompts = {}
    for name, source in (("good", good), ("bad", bad), ("evaluation", evaluation)):
        data = Path(source).read_bytes()
        if not data.decode("utf-8-sig").strip():
            raise ValueError(f"{name} prompts must not be empty")
        prompts[name] = data

    # A fresh directory keeps checkpoints/exports separate from previous runs.
    output.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for name, data in prompts.items():
        (output / f"{name}.txt").write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    source = lambda name: {"dataset": str(output / f"{name}.txt")}
    config = _table("", {
        "model": str(model), "seed": seed, "n_trials": trials,
        "n_startup_trials": min(trials, 10), "batch_size": 1,
        "max_response_length": max_tokens, "response_prefix": "",
        "study_checkpoint_dir": str(output / "checkpoints"),
        "model_action": "save", "export_strategy": "adapter", "trial_index": 0,
        "save_directory": str(output / "adapter"),
        "modified_model_type": "behavior-candidate", "upload_tags": [],
    })
    for plugin in ("keyword_rate.KeywordRate", "kl_divergence.KLDivergence"):
        config += _table("scorers", {"plugin": f"heretic.scorers.{plugin}",
                                     "optimization": "minimize"}, array=True)
    config += _table("modifiers", {"plugin": "heretic.modifiers.ara.ARA"}, array=True)
    # Override every upstream prompt source, including automatic probe datasets.
    for name, values, array in (
        ("batch_size_test_prompts", source("good"), False),
        ("response_prefix_test_prompts", source("good"), True),
        ("scorer.KeywordRate", {"score_name": "Unwanted expression rate",
                                "keyword_markers": markers}, False),
        ("scorer.KeywordRate.prompts", source("evaluation"), False),
        ("scorer.KLDivergence.prompts", source("good"), False),
        ("modifier.ARA.good_prompts", source("good"), False),
        ("modifier.ARA.bad_prompts", source("bad"), False),
    ):
        config += _table(name, values, array=array)
    (output / "config.toml").write_bytes(config.encode("utf-8"))
    manifest = {
        "heretic_version": HERETIC_VERSION, "model": str(model), "goal": goal,
        "model_config_sha256": hashlib.sha256((model / "config.json").read_bytes()).hexdigest(),
        "prompt_sha256": hashes, "config_sha256": hashlib.sha256(config.encode()).hexdigest(),
        "status": "prepared", "quality_validated": False,
    }
    _manifest(output, manifest)
    return output


def _manifest(output, data):
    (output / "manifest.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run(output, python=sys.executable):
    """Explicitly launch the optional tool in its own, offline Python process."""
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "prepared":
        raise ValueError("prepare a fresh run directory before executing")
    if hashlib.sha256((output / "config.toml").read_bytes()).hexdigest() != manifest["config_sha256"]:
        raise ValueError("prepared configuration changed; prepare a fresh run")
    env = {key: value for key, value in os.environ.items() if not key.startswith("HERETIC_")}
    env.update({"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                "HF_HOME": str(output / "cache" / "huggingface"),
                "TORCH_HOME": str(output / "cache" / "torch"),
                "TRITON_CACHE_DIR": str(output / "cache" / "triton"),
                "XDG_CACHE_HOME": str(output / "cache"), "PYTHONDONTWRITEBYTECODE": "1",
                "TMP": str(output / "tmp"),
                "TEMP": str(output / "tmp"), "TMPDIR": str(output / "tmp")})
    (output / "tmp").mkdir()
    command = [str(python), "-c", (
        "from importlib.metadata import version\n"
        "v = version('heretic-llm')\n"
        f"if v != {HERETIC_VERSION!r}:\n"
        f"    raise RuntimeError('Expected heretic-llm {HERETIC_VERSION}, got '+v)\n"
        "from heretic.main import main\nmain()\n"
    )]
    manifest["status"] = "running"
    _manifest(output, manifest)
    try:
        result = subprocess.run(command, cwd=output, env=env, check=False)
    except OSError:
        manifest["status"] = "failed"
        _manifest(output, manifest)
        raise
    manifest["returncode"] = result.returncode
    adapter = output / "adapter"
    exported = (adapter / "adapter_config.json").is_file() and any(adapter.glob("*.safetensors"))
    manifest["status"] = "exported_unvalidated" if result.returncode == 0 and exported else "failed"
    _manifest(output, manifest)
    result.check_returncode()
    if not exported:
        raise RuntimeError("Heretic exited without exporting an adapter; see its output")
    return adapter


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "good", "bad", "evaluation", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--trials", type=int, required=True)
    parser.add_argument("--goal", choices=("nosyco", "noslop", "both"), default="both")
    parser.add_argument("--markers", type=Path, help="Optional JSON list overriding expression markers")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", action="store_true", help="Execute Heretic; otherwise only prepare files")
    parser.add_argument("--heretic-python", type=Path, default=Path(sys.executable))
    args = vars(parser.parse_args(argv))
    execute, python = args.pop("run"), args.pop("heretic_python")
    markers = args["markers"]
    args["markers"] = json.loads(markers.read_text(encoding="utf-8")) if markers else None
    output = prepare(**args)
    print(run(output, python) if execute else output / "config.toml")


if __name__ == "__main__":
    main()
