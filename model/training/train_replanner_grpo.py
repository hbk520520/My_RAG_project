"""
train_replanner_grpo.py —— Replanner GRPO 强化训练
===================================================
模型：Qwen2.5-14B-Instruct + LoRA
算法：GRPO (群体相对策略优化)
数据：Evol-Instruct 生成的 replanner_scenarios.jsonl
"""
import os
import sys
import json
import torch
import numpy as np
from typing import List, Dict, Any
from datasets import Dataset, load_dataset
from trl import GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.unsloth_loader import UnslothLoader

# ============================================================================
# 1. 加载 SFT 冷启动模型
# ============================================================================
sft_path = "./outputs/replanner_sft"
if os.path.exists(sft_path):
    loader = UnslothLoader("replanner", custom_model_name=sft_path)
    model, tokenizer = loader.load()
else:
    loader = UnslothLoader("replanner")
    model, tokenizer = loader.load()


# ============================================================================
# 2. 模拟执行环境 (simulate_execution)
# ============================================================================
# Replanner 输出是任务队列，不是最终答案。
# simulate_execution 通过 LLM-as-Judge 模拟该计划执行后的效果，
# 返回 {accuracy, steps, format_error} 三个维度评分。
# ============================================================================

def simulate_execution(prompt: str, task_queue: List[Dict]) -> Dict[str, float]:
    """
    模拟执行 Replanner 生成的任务队列，返回评分。

    评分逻辑（不调用真实图引擎，用规则 + LLM 模拟）：
    1. format_error: 队列是否合法 JSON / 结构正确
    2. steps: 队列长度（越短越高效，但需覆盖关键维度）
    3. accuracy: 调用 LLM-as-Judge 判断队列是否覆盖了 prompt 中缺失的维度

    :param prompt:  失败的上下文描述（含用户问题 + 已有事实 + 失败记录）
    :param task_queue: Replanner 生成的新队列 [{"task_desc", "engine", "rationale"}]
    :return: {"accuracy": 0~1, "steps": int, "format_error": 0~1}
    """
    result = {"accuracy": 0.0, "steps": 0, "format_error": 0.0}

    # ---- 2a. 格式校验 ----
    if not isinstance(task_queue, list) or len(task_queue) == 0:
        result["format_error"] = 1.0
        return result

    for i, task in enumerate(task_queue):
        if not isinstance(task, dict):
            result["format_error"] += 0.3
            continue
        if "task_desc" not in task:
            result["format_error"] += 0.2
        if "engine" not in task:
            result["format_error"] += 0.1
        elif task["engine"] not in ("GRAPH_TRAVERSAL", "GLOBAL_DENSE_WORMHOLE"):
            result["format_error"] += 0.1
    result["format_error"] = min(result["format_error"], 1.0)

    # ---- 2b. 步骤计数 ----
    result["steps"] = len(task_queue)

    # ---- 2c. 语义准确性评估（LLM-as-Judge） ----
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com"
        )

        tasks_desc = "\n".join([
            f"{i+1}. [{t.get('engine','?')}] {t.get('task_desc','')} ({t.get('rationale','')})"
            for i, t in enumerate(task_queue)
        ])

        judge_prompt = f"""你是一个法律任务队列评估裁判。

【失败场景】：
{prompt[-500:]}

【Replanner 生成的新任务队列】：
{tasks_desc}

请从以下维度评分（0~1，保留两位小数）：
1. 相关性：新任务是否针对失败原因做了调整？（权重 0.4）
2. 完整性：是否覆盖了原始问题的关键法律维度？（权重 0.35）
3. 可行性：每个任务是否切实可执行、不过于宽泛？（权重 0.25）

只输出一个数字（如 0.75），不要其他文字。"""

        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=10
        )
        result["accuracy"] = float(resp.choices[0].message.content.strip())
    except Exception:
        # LLM 不可用时，使用启发式规则
        result["accuracy"] = _heuristic_accuracy(prompt, task_queue)

    return result


def _heuristic_accuracy(prompt: str, task_queue: List[Dict]) -> float:
    """
    启发式准确度评估（无需 LLM 的降级方案）。

    评分规则：
    - 队列长度 2~5 之间 (+0.3)
    - 包含虫洞穿越任务（说明识别到图谱断裂） (+0.2)
    - 任务描述不是原文照抄 fail_log (+0.2)
    - 每个任务有 rationale (+0.2)
    - 没有空任务 (+0.1)
    """
    score = 0.0

    if 2 <= len(task_queue) <= 5:
        score += 0.3

    has_wormhole = any(t.get("engine") == "GLOBAL_DENSE_WORMHOLE" for t in task_queue)
    if has_wormhole:
        score += 0.2

    # 检查任务是否原创（非照抄 prompt）
    prompt_lower = prompt.lower()
    originality = sum(
        1 for t in task_queue
        if t.get("task_desc", "").lower()[:20] not in prompt_lower
    )
    if originality >= len(task_queue) * 0.5:
        score += 0.2

    has_rationale = sum(1 for t in task_queue if len(t.get("rationale", "")) > 5)
    if has_rationale >= len(task_queue) * 0.5:
        score += 0.2

    all_non_empty = all(len(t.get("task_desc", "")) > 3 for t in task_queue)
    if all_non_empty:
        score += 0.1

    return min(score, 1.0)


# ============================================================================
# 3. GRPO Reward Function（核心：连接 trainer 与模拟环境）
# ============================================================================
def reward_func(prompts: List[str], completions: List[str]) -> List[torch.Tensor]:
    """
    GRPO Reward Function —— 评价每个 completion 的质量。

    评分公式：
      reward = accuracy + 0.1 * (1 - steps/10) - 0.5 * format_error

    含义：
      - accuracy:  语义准确度 (LLM-as-Judge, 权重最高)
      - steps:     步骤效率奖励 (越短越好，但上限 0.1)
      - penalty:   格式错误惩罚 (严格扣分)
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        try:
            # 解析 Replanner 输出
            parsed = json.loads(completion.strip())
            task_queue = parsed.get("task_queue", [])

            # 模拟执行，获取三维评分
            sim_result = simulate_execution(prompt, task_queue)

            # 综合奖励计算
            accuracy = sim_result["accuracy"]
            steps = sim_result["steps"]
            format_error = sim_result["format_error"]

            reward = (
                accuracy                          # 准确性 (0~1)
                + 0.1 * (1 - min(steps, 10) / 10)  # 效率奖励 (0~0.1)
                - 0.5 * format_error               # 格式惩罚 (0~0.5)
            )

            rewards.append(torch.tensor(reward, dtype=torch.float))

        except (json.JSONDecodeError, KeyError, TypeError):
            # JSON 解析失败：严厉惩罚
            rewards.append(torch.tensor(-1.0, dtype=torch.float))

    return rewards


# ============================================================================
# 4. 训练数据准备
# ============================================================================
data_path = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "training_data", "generated", "replanner_scenarios.jsonl"
)

if os.path.exists(data_path):
    ds = load_dataset("json", data_files=data_path, split="train")
else:
    print("⚠️ 数据文件不存在，使用示例 prompts")
    prompts_data = [
        "用户问题：试用期最后一天被辞退。已有事实：确认存在劳动关系，月薪8000。失败记录：检索'试用期辞退赔偿'未找到相关法条。失败次数：2。请生成新的任务队列。",
        "用户问题：公司拖欠3个月工资。已有事实：劳动合同约定月薪10000。失败记录：检索'拖欠工资劳动法'返回空。失败次数：3。请生成新的任务队列。",
        "用户问题：未签合同被辞退。已有事实：工作满11个月。失败记录：检索'未签劳动合同'仅返回1条，不充分。失败次数：1。请生成新的任务队列。",
    ]
    ds = Dataset.from_dict({"prompt": prompts_data})

# ============================================================================
# 5. 训练配置 & 执行
# ============================================================================
training_args = TrainingArguments(
    output_dir="./outputs/replanner_grpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=1e-5,
    logging_steps=10,
    bf16=True,
    report_to="none",
    remove_unused_columns=False,
    save_strategy="epoch",
)

grpo_trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    tokenizer=tokenizer,
    reward_funcs=reward_func,
    num_generation_per_prompt=4,  # 每个 prompt 生成 4 个候选
    max_new_tokens=512,
)

# ============================================================================
# 6. 训练 & 保存
# ============================================================================
grpo_trainer.train()
loader.save_lora(model, tokenizer, "./saved_loras/replanner_grpo")
