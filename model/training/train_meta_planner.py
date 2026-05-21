"""
train_meta_planner.py —— Meta-Planner SFT 训练
==============================================
模型：Qwen2.5-14B-Instruct + LoRA (Unsloth 加速)
数据：Evol-Instruct 生成的 planner_trajectories.jsonl
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import Dataset, load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

from utils.unsloth_loader import UnslothLoader

# ============================================================================
# 1. 加载模型
# ============================================================================
loader = UnslothLoader("meta_planner")
model, tokenizer = loader.load()

# ============================================================================
# 2. 加载数据
# ============================================================================
data_path = os.path.join(os.path.dirname(__file__), "..", "..", "training_data", "generated", "planner_trajectories.jsonl")
if os.path.exists(data_path):
    dataset = load_dataset("json", data_files=data_path, split="train")
else:
    print(f"⚠️ 数据文件不存在: {data_path}，使用示例数据")
    dataset = Dataset.from_list([{
        "messages": [
            {"role": "user", "content": "俺被口头辞退了，能要多少钱？"},
            {"role": "assistant", "content": '{"task_queue": ["查核是否存在劳动关系", "查核辞退是否合法", "计算赔偿金额"]}'}
        ]
    }])

dataset = dataset.map(lambda x: loader.format_chat(x, tokenizer))

# ============================================================================
# 3. 训练配置
# ============================================================================
training_args = TrainingArguments(
    output_dir="./outputs/planner_lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=not FastLanguageModel.is_bfloat16_supported(),
    bf16=FastLanguageModel.is_bfloat16_supported(),
    logging_steps=10,
    optim="adamw_8bit",
    save_strategy="epoch",
    report_to="none",
)

from unsloth import FastLanguageModel, is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=4096,
    args=training_args,
)

# ============================================================================
# 4. 训练 & 保存
# ============================================================================
trainer.train()
loader.save_lora(model, tokenizer, "./saved_loras/planner_lora")
