"""Seeded task mixture and bounded Zipf-like sampling, grouped before augmentation."""
from __future__ import annotations

import hashlib
import random
from decimal import Decimal

from .environment import reference_trajectory, resolve
from .schemas import Document, GenerationConfig, Sample


def split_for(group_id, config):
    fraction = int(hashlib.sha256(group_id.encode()).hexdigest()[:16], 16) / 2**64
    if fraction < config.train_fraction:
        return "train"
    return "validation" if fraction < config.train_fraction + config.validation_fraction else "test"


def generate(config: GenerationConfig):
    """Yield one variant at a time; all variants of an entity/source group stay together."""
    config = GenerationConfig.model_validate(config)
    families = sorted(config.family_weights)
    for index in range(config.groups + config.ood_groups):
        ood = index >= config.groups
        local_index = index - config.groups if ood else index
        group_id = f"syn-{config.seed}-{'ood' if ood else 'base'}-{local_index}"
        split = "ood" if ood else split_for(group_id, config)
        base_rng = random.Random(f"{group_id}:base")
        base = [base_rng.randint(100, 1000000) for _ in range(config.max_records)]
        topic = base_rng.choices(range(1, config.topic_count + 1),
                                weights=[1 / k**config.topic_exponent for k in range(1, config.topic_count + 1)])[0]
        for variant in range(config.variants):
            rng = random.Random(f"{group_id}:{variant}")
            family = rng.choices(families, weights=[config.family_weights[name] for name in families])[0]
            count = 1 if family == "lookup" else rng.choices(
                range(2, config.max_records + 1),
                weights=[1 / k**config.complexity_exponent for k in range(2, config.max_records + 1)])[0]
            records = [f"{group_id}-record-{i}" for i in range(count)]
            documents, transforms = [], []
            for i, record in enumerate(records):
                amount = None if family == "missing" and i == count - 1 else base[i]
                statement = f"{record} 的金额（整数分）为 {amount if amount is not None else '尚待用户提供'}。"
                statements = [statement]
                if rng.random() < config.repetition_rate:
                    statements.append(f"再次核对：记录 {record} 的分币值为 {amount if amount is not None else '未填写'}。")
                    transforms.append("intra_document_repetition")
                documents.append(Document(id=f"{record}-primary", record_id=record, amount=amount,
                                          statements=statements))
            if family != "missing" and rng.random() < config.conflict_rate:
                unresolved = rng.random() < config.unresolved_conflict_rate
                documents.append(Document(id=f"{records[0]}-conflict", record_id=records[0],
                                          amount=base[0] + 100,
                                          authority="primary" if unresolved else "reference",
                                          statements=[f"{records[0]} 的另一份记录金额（整数分）为 {base[0] + 100}。"]))
                transforms.append("unresolved_conflict" if unresolved else "lower_authority_conflict")
            documents.append(Document(id=f"{group_id}-unrelated", record_id=f"{group_id}-other",
                                      amount=12345, authority="reference", statements=["无关记录，金额 12345 分。 "]))
            rng.shuffle(documents)
            tools = {"read": "read_record", "sum": "sum_records", "ask": "request_clarification",
                     "distractor": "read_weather"}
            if ood:
                tools = {operation: f"operation_{i}" for i, operation in enumerate(("ask", "distractor", "sum", "read"))}
                transforms.extend(["renamed_tools", "reworded_query"])
            operation_text = "核对并汇总" if not ood else "逐项找出金额，随后给出合计"
            query = (f"虚构资料核对任务，主题类别 {topic}：{operation_text}以下记录：{', '.join(records)}。"
                     "金额以整数分存储，合计转换为两位小数的元。仅采用 active=true 的 primary 资料；"
                     "同一记录的 primary 金额冲突时不得任选；金额未填写时请求用户补充；"
                     "证据冲突时请求补充证据。重试一次 retryable_error。此规则仅属于虚构环境，不判断法律效力。")
            expected = resolve(documents, records, missing_user=family == "missing")
            yield Sample(sample_id=f"{group_id}-v{variant}", group_id=group_id, split=split,
                         family=family, topic_rank=topic, query=query, record_ids=records, documents=documents,
                         tools=tools, transient_record=records[0] if family == "recovery" else None,
                         transforms=sorted(set(transforms)), expected=expected)


def trajectories(sample):
    """Scripted candidates, not Agent rollouts; never label all steps by final success."""
    gold = reference_trajectory(sample)
    yield gold
    redundant = gold.model_copy(deep=True, update={"name": "redundant"})
    # Repeat a successful read after transient recovery, before the terminal operation.
    read = next(step for step in gold.steps if step.observation.get("status") in ("sufficient", "partial"))
    redundant.steps.insert(-1, read.model_copy(deep=True))
    yield redundant
    fabricated = gold.model_copy(deep=True, update={"name": "fabricated_observation"})
    fabricated.steps[0].observation = {"status": "fabricated"}
    yield fabricated
    wrong = gold.model_copy(deep=True, update={"name": "wrong_final"})
    wrong.answer.amount = str(Decimal(gold.answer.amount or "0") + Decimal("0.01"))
    yield wrong


def grader_record(sample):
    """Keep the existing Grade output fields; facts are quoted fixture values, not legal labels."""
    facts = [f"[{d.id}] {d.record_id}: {d.amount} 分" for d in sample.documents
             if d.id in sample.expected.evidence_ids]
    missing = sample.expected.missing_records
    return {"status": sample.expected.status, "extracted_facts": facts,
            "rationale": "资料覆盖完整，可执行确定性求和。" if not missing else "存在未填写金额或无法消解的主资料冲突。",
            "missing_info": "请补充或核对：" + ", ".join(missing) if missing else "",
            "missing_info_kind": sample.expected.missing_info_kind}


def preference_pair(sample):
    """Plausible unsupported facts, not self-labelled or conspicuously absurd negatives."""
    chosen = grader_record(sample)
    facts = list(chosen["extracted_facts"])
    if sample.expected.missing_records:
        record = sample.expected.missing_records[0]
        source = next(d for d in sample.documents if d.record_id == record and d.authority == "primary")
        facts.append(f"[{source.id}] {record}: {source.amount if source.amount is not None else 10000} 分")
    else:
        source = next(d for d in sample.documents if d.id in sample.expected.evidence_ids)
        facts[facts.index(f"[{source.id}] {source.record_id}: {source.amount} 分")] = (
            f"[{source.id}] {source.record_id}: {source.amount + 1} 分")
    rejected = {**chosen, "status": "sufficient", "extracted_facts": facts,
                "rationale": "资料覆盖完整，可执行确定性求和。", "missing_info": ""}
    return chosen, rejected
