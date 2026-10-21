"""Task mixtures, entity exposure counts and complexity are sampled independently."""
from __future__ import annotations

import hashlib
import json
import random

from .contracts import Config, Evidence, Record, Step, Task
from .simulation import Environment, reference


def split_for(group_id, config):
    bucket = int(hashlib.sha256(group_id.encode()).hexdigest()[:16], 16) / 2**64
    if bucket < config.train_fraction:
        return "train"
    return "validation" if bucket < config.train_fraction + config.validation_fraction else "test"


def weighted_count(rng, low, high, exponent):
    return rng.choices(range(low, high + 1), weights=[1 / k**exponent for k in range(low, high + 1)])[0]


def generate_groups(config: Config):
    families = sorted(config.family_weights)
    for index in range(config.groups + config.ood_groups):
        ood = index >= config.groups
        local = index - config.groups if ood else index
        group = f"fixture-{config.seed}-{'ood' if ood else 'base'}-{local}"
        split = "ood" if ood else split_for(group, config)
        rng = random.Random(group + ":facts")
        facts = [{"units": rng.randint(1, 10000), "status": rng.choice(["待核对", "已归档", "需补证"])}
                 for _ in range(config.max_items)]
        copies = weighted_count(random.Random(group + ":exposure"), 1, config.max_variants, config.exposure_exponent)
        tasks = []
        for variant in range(copies):
            rng = random.Random(f"{group}:v{variant}")
            family = rng.choices(families, weights=[config.family_weights[f] for f in families])[0]
            size = 1 if family == "lookup" else weighted_count(rng, 2, config.max_items, config.complexity_exponent)
            mode = rng.choice(["missing", "permission", "budget"]) if family == "constraints" else "normal"
            field = "status" if family in ("dependency", "parallel") else "units"
            if family == "lookup":
                field = rng.choice(["units", "status"])
            operation = "copy" if family in ("lookup", "dependency") else "collect" if family == "parallel" else "sum"
            records, transforms = [], []
            for i in range(size):
                identity = f"{group}-record-{i}"
                values = dict(facts[i])
                if mode == "missing" and i == size - 1:
                    values[field] = None
                statements = [json.dumps(values, ensure_ascii=False, sort_keys=True)]
                if rng.random() < config.repetition_rate:
                    statements.append("复核同一记录：" + "；".join(f"{k}={v}" for k, v in sorted(values.items())))
                    transforms.append("intra_document_repetition")
                source = Evidence(id=identity + ("-primary-masked" if values[field] is None else "-primary"),
                                  facts=values, statements=statements)
                records.append(Record(id=identity, sources=[source],
                                      requires=[f"{group}-record-{i-1}"] if family == "dependency" and i else [],
                                      next_id=f"{group}-record-{i+1}" if family == "dependency" and i < size - 1 else None))
            if mode == "normal" and rng.random() < config.conflict_rate:
                unresolved = rng.random() < config.unresolved_conflict_rate
                obsolete = not unresolved and rng.choice([False, True])
                changed = dict(records[0].sources[0].facts)
                changed[field] = changed[field] + 1 if isinstance(changed[field], int) else "有争议"
                records[0].sources.append(Evidence(id=f"{records[0].id}-conflict-v{variant}", facts=changed,
                                                  authority="primary" if unresolved or obsolete else "reference", active=not obsolete,
                                                  statements=[json.dumps(changed, ensure_ascii=False, sort_keys=True)]))
                rng.shuffle(records[0].sources)
                transforms.append("same_authority_conflict" if unresolved else "obsolete_version" if obsolete else "lower_authority_conflict")
            names = {op: op for op in ("read", "combine", "handoff", "write", "request", "finish", "distractor")}
            if ood:
                names = {op: f"operation_{i}" for i, op in enumerate(reversed(list(names)))}
                transforms.extend(["renamed_tools", "alternate_return_format", "reworded_query"])
            if family == "dependency":
                transforms.append("dependency_chain")
            if family == "parallel":
                transforms.append("role_handoff")
            if mode != "normal":
                transforms.append(mode)
            instruction = {
                "lookup": "读取指定记录的字段，不进行额外推断。",
                "composition": "读取全部记录，汇总 units，并把结果写入本任务内存结果区。",
                "dependency": "从起始记录沿 next_id 读取至末尾，返回末尾记录的 status。",
                "parallel": "两名 researcher 分别读取独立分支并交接给 coordinator，再按记录 ID 汇合 status。",
                "recovery": "读取全部记录并汇总 units；遇到暂时读取错误，重试一次。",
                "constraints": "汇总 units 并归档到内存结果区；遵守权限和调用预算，缺失时请求用户补充。"}[family]
            query = ("虚构任务：" + instruction + " 仅采用 active=true 的 primary 来源；同级来源冲突须补证，"
                     "空字段须补充，不能猜测。这里的来源优先级仅是模拟规则，不表示现实法律效力。")
            if ood:
                query = "请遵循所列工具与状态规则完成下述工作。" + query
            tasks.append(Task(id=f"{group}-v{variant}", group_id=group, split=split, family=family, query=query,
                              field=field, operation=operation, records=records, tool_names=names,
                              response_format="rows" if ood else "sources",
                              transient_record=records[0].id if family == "recovery" else None,
                              max_calls=0 if mode == "budget" else 2 * size + 5,
                              write_required=family in ("composition", "constraints"),
                              write_allowed=mode != "permission", transforms=sorted(set(transforms))))
        yield group, split, tasks


def trajectory_candidates(task):
    """Independent optional stage: these are scripted fixtures, never model rollouts."""
    gold = reference(task)
    yield gold
    if task.max_calls:
        redundant = gold.model_copy(deep=True, update={"name": "redundant"})
        observation = Environment(task).execute("coordinator", task.tool_names["distractor"], {})
        redundant.steps.insert(0, Step(tool=task.tool_names["distractor"], arguments={}, observation=observation))
        yield redundant
    fabricated = gold.model_copy(deep=True, update={"name": "fabricated_observation"})
    fabricated.steps[0].observation = {"status": "fabricated"}
    yield fabricated
    invalid = gold.model_copy(deep=True, update={"name": "invalid_tool_use"})
    invalid.steps.insert(0, Step(tool=task.tool_names["write"], arguments={"value": gold.answer.value},
                                 observation={"status": "stored"}))
    yield invalid
    wrong = gold.model_copy(deep=True, update={"name": "wrong_final"})
    value = gold.answer.value
    wrong.answer.value = value + 1 if isinstance(value, int) else "未获证据支持的替代结果"
    wrong.steps[-1].arguments = wrong.answer.model_dump(mode="json")
    yield wrong
