# train_replanner_grpo.py
from trl import GRPOTrainer
from datasets import Dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

# 假设已经有一个 SFT 后的 Replanner 模型
model_path = "./replanner-sft"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
model.gradient_checkpointing_enable()

# 定义奖励函数（需要模拟环境）
def reward_func(prompts, completions):
    """
    prompts: 失败状态描述 + 原始任务队列
    completions: 模型生成的重规划 task_queue
    返回每个 completion 的奖励值 tensor [batch_size]
    """
    rewards = []
    for prompt, completion in zip(prompts, completions):
        try:
            new_plan = json.loads(completion)["task_queue"]
            # 在沙箱中模拟执行 new_plan
            sim_result = simulate_execution(prompt, new_plan)  # 自定义函数
            # 奖励 = 最终答案正确性 + 步骤效率 - 格式错误惩罚
            reward = sim_result["accuracy"] + 0.1*(1 - sim_result["steps"]/10) - 0.5*sim_result["format_error"]
            rewards.append(torch.tensor(reward, dtype=torch.float))
        except:
            rewards.append(torch.tensor(-1.0))  # 格式错误严惩
    return rewards

# GRPO 训练数据：只需要 prompt，不需要正确输出
prompts_data = [
    "当前失败状态：计划 [查劳动关系] 未找到证据。原始队列：... 请生成新的任务队列。",
    # ...
]
ds = Dataset.from_dict({"prompt": prompts_data})

training_args = TrainingArguments(
    output_dir="./replanner-grpo",
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_steps=10,
    bf16=True,
    report_to="none",
    remove_unused_columns=False,
)

grpo_trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    tokenizer=tokenizer,
    reward_funcs=reward_func,
    num_generation_per_prompt=4,   # 生成 4 个不同重规划
    max_new_tokens=256,
)
grpo_trainer.train()
grpo_trainer.save_model("replanner-grpo-final")