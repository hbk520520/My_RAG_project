"""Offline component contracts; the external Heretic process is a test double."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest

from model.behavior import HERETIC_VERSION, main, prepare, run


@pytest.fixture
def inputs(tmp_path):
    model = tmp_path / "本地模型"
    model.mkdir()
    (model / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    paths = {"model": model, "output": tmp_path / "候选模型", "trials": 2}
    for name in ("good", "bad", "evaluation"):
        paths[name] = tmp_path / f"{name}.txt"
        paths[name].write_text(f"{name}：按照给定事实重新规划任务。\n", encoding="utf-8")
    return paths


def test_prepare_is_offline_and_explicitly_replaces_upstream_defaults(inputs):
    output = prepare(**inputs)
    config = tomllib.loads((output / "config.toml").read_text(encoding="utf-8"))
    assert config["model"] == str(inputs["model"].resolve())
    assert config["export_strategy"] == "adapter" and config["model_action"] == "save"
    assert config["save_directory"] == str(output / "adapter")
    assert config["modifiers"] == [{"plugin": "heretic.modifiers.ara.ARA"}]
    assert {s["plugin"] for s in config["scorers"]} == {
        "heretic.scorers.keyword_rate.KeywordRate", "heretic.scorers.kl_divergence.KLDivergence"}
    assert config["max_response_length"] == 2048
    for spec in (config["batch_size_test_prompts"], *config["response_prefix_test_prompts"],
                 config["scorer"]["KeywordRate"]["prompts"],
                 config["scorer"]["KLDivergence"]["prompts"],
                 config["modifier"]["ARA"]["good_prompts"],
                 config["modifier"]["ARA"]["bad_prompts"]):
        assert Path(spec["dataset"]).is_file()
        assert Path(spec["dataset"]).parent == output
    markers = config["scorer"]["KeywordRate"]["keyword_markers"]
    assert "您说得太对了" in markers
    assert not any(word in markers for word in ("sorry", "不知道", "抱歉", "缺少证据"))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "prepared" and manifest["quality_validated"] is False
    assert not (output / "adapter").exists()


@pytest.mark.parametrize("goal,expected", [("nosyco", "您说得太对了"),
                                           ("noslop", "让我们一起深入探讨")])
def test_selectable_goals_and_custom_markers(inputs, goal, expected):
    output = prepare(**inputs, goal=goal)
    config = tomllib.loads((output / "config.toml").read_text(encoding="utf-8"))
    markers = config["scorer"]["KeywordRate"]["keyword_markers"]
    assert expected in markers
    assert len(markers) < 7
    inputs["output"] = output.with_name("custom")
    custom = prepare(**inputs, goal=goal, markers=["自定义表达"])
    assert tomllib.loads((custom / "config.toml").read_text(encoding="utf-8"))[
        "scorer"]["KeywordRate"]["keyword_markers"] == ["自定义表达"]


@pytest.mark.parametrize("change", ["missing_model", "empty_prompts", "empty_markers", "bad_goal", "zero_trials"])
def test_invalid_inputs_fail_before_creating_output(inputs, change):
    if change == "missing_model":
        inputs["model"] = inputs["model"] / "not-here"
    elif change == "empty_prompts":
        inputs["good"].write_text(" \n", encoding="utf-8")
    elif change == "empty_markers":
        inputs["markers"] = []
    elif change == "bad_goal":
        inputs["goal"] = "refusals"
    else:
        inputs["trials"] = 0
    with pytest.raises(ValueError):
        prepare(**inputs)
    assert not inputs["output"].exists()


def test_preserves_original_and_previous_run(inputs):
    inputs["output"] = inputs["model"] / "modified"
    with pytest.raises(ValueError, match="outside"):
        prepare(**inputs)
    inputs["output"] = inputs["model"].parent / "fresh"
    output = prepare(**inputs)
    original = (output / "config.toml").read_bytes()
    with pytest.raises(FileExistsError):
        prepare(**inputs)
    assert (output / "config.toml").read_bytes() == original


def fake_tool(folder, monkeypatch, *, export=True, version=HERETIC_VERSION):
    site = folder / "fake-site"
    package = site / "heretic"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    metadata = site / f"heretic_llm-{version}.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(f"Name: heretic-llm\nVersion: {version}\n", encoding="utf-8")
    (package / "main.py").write_text('''
import json, os, tomllib
from pathlib import Path
def main():
    root = Path.cwd()
    assert (root / "config.toml").is_file()  # Heretic 2.0 reads config from cwd
    assert "HERETIC_MODEL_ACTION" not in os.environ
    assert all(os.environ[key] == "1" for key in (
        "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"))
    assert Path(os.environ["TEMP"]).parent == root
    config = tomllib.loads((root / "config.toml").read_text(encoding="utf-8"))
    assert config["model_action"] == "save"
''' + ('''
    target = Path(config["save_directory"])
    target.mkdir()
    (target / "adapter_config.json").write_text("{}")
    (target / "adapter_model.safetensors").write_bytes(b"fake test artifact, not model weights")
''' if export else ""), encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(site))
    monkeypatch.setenv("HERETIC_MODEL_ACTION", "upload")


def test_explicit_run_process_boundary_and_unvalidated_export(inputs, tmp_path, monkeypatch):
    fake_tool(tmp_path, monkeypatch)
    output = prepare(**inputs)
    original = (inputs["model"] / "config.json").read_bytes()
    adapter = run(output)
    assert adapter == output / "adapter"
    assert (inputs["model"] / "config.json").read_bytes() == original
    assert os.environ["HERETIC_MODEL_ACTION"] == "upload"  # parent environment unchanged
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "exported_unvalidated" and manifest["quality_validated"] is False
    with pytest.raises(ValueError, match="fresh"):
        run(output)


@pytest.mark.parametrize("export,version", [(False, HERETIC_VERSION), (True, "1.4.0")])
def test_failed_tool_never_reports_success(inputs, tmp_path, monkeypatch, export, version):
    fake_tool(tmp_path, monkeypatch, export=export, version=version)
    output = prepare(**inputs)
    with pytest.raises((RuntimeError, subprocess.CalledProcessError)):
        run(output)
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_changed_configuration_is_not_executed(inputs):
    output = prepare(**inputs)
    with (output / "config.toml").open("a", encoding="utf-8") as stream:
        stream.write('\nmodel_action = "upload"\n')
    with pytest.raises(ValueError, match="configuration changed"):
        run(output)


def test_missing_interpreter_records_failure(inputs):
    output = prepare(**inputs)
    with pytest.raises(OSError):
        run(output, output / "missing-python")
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["status"] == "failed"


def test_cli_prepare_does_not_require_ml_packages(inputs, capsys):
    args = [value for key, value in inputs.items() for value in (f"--{key}", str(value))]
    main(args)
    assert "config.toml" in capsys.readouterr().out
    manifest = json.loads((inputs["output"] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "prepared"


def test_import_does_not_load_optional_dependencies(repo_root):
    result = subprocess.run([sys.executable, "-B", "-c", (
        "import sys; import model.behavior; "
        "assert not {'torch', 'heretic', 'unsloth', 'vllm'} & sys.modules.keys()"
    )], cwd=repo_root, capture_output=True, text=True, check=True)
    assert not result.stdout and not result.stderr


def test_existing_role_route_can_plug_and_unplug_behavior_adapter(monkeypatch):
    from config_loader import cfg
    from legal_agent.llm import complete, model_role
    seen = []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])
    local = SimpleNamespace(chat=lambda *args, **kwargs: seen.append(kwargs) or response)
    mapping = {"replanner": "replanner_behavior", "reasoner": "reasoner"}
    monkeypatch.setitem(cfg._data["vllm"], "role_adapters", mapping)
    for role, expected in (("replanner", "replanner_behavior"), ("reasoner", "reasoner"),
                           ("generator", None)):
        with model_role(role):
            complete(local, "base", [{"role": "user", "content": "q"}])
        assert seen[-1]["lora_name"] == expected
    mapping["replanner"] = "replanner"
    with model_role("replanner"):
        complete(local, "base", [{"role": "user", "content": "q"}])
    assert seen[-1]["lora_name"] == "replanner"
