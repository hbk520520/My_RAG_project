"""
Reasoner 训练 —— 长上下文 SFT，学会 IRAC 推演
===========================================
7B 模型，Flash Attention 2 开满，梯度检查点省显存。
数据是 IRAC 格式（Issue→Rule→Application→Conclusion），
训练后能基于已有事实做严格的法律三段论。

技术栈: SFTTrainer (trl) / Unsloth / Flash Attention 2
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import Dataset, load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported

from utils.unsloth_loader import UnslothLoader

# ============================================================================
# 1. 加载模型（Reasoner 需要长上下文支持）
# ============================================================================
loader = UnslothLoader("reasoner")
model, tokenizer = loader.load()

# 开启梯度检查点
model.gradient_checkpointing_enable()

# ============================================================================
# 2. 数据加载
# ============================================================================
data_path = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "training_data", "generated", "reasoner_qa.jsonl"
)

if os.path.exists(data_path):
    dataset = load_dataset("json", data_files=data_path, split="train")
else:
    print("⚠️ 使用示例 Reasoner 数据")
    dataset = Dataset.from_list([{
        "messages": [
            {"role": "system", "content": "你是一名资深法官助理。基于给定事实进行 IRAC 格式的法律推演。"},
            {"role": "user", "content": "案件事实：张三于2022年3月入职A公司，月薪8000元。2024年1月公司以'部门取消'为由口头通知辞退，未提前30日通知，未支付任何补偿。问：公司应赔偿多少？"},
            {"role": "assistant", "content": "【Issue】公司以'客观情况发生重大变化'为由解除合同...\n【Rule】劳动合同法第40条第3项、第46条、第47条...\n【Application】张三工作年限1年10个月≈2年，月薪8000元...\n【Conclusion】公司应支付N+1=3×8000=24000元补偿。"}
        ]
    }])

dataset = dataset.map(lambda x: loader.format_chat(x, tokenizer))

# ============================================================================
# 3. 训练配置
# ============================================================================
training_args = TrainingArguments(
    output_dir="./outputs/reasoner_sft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    num_train_epochs=2,
    max_steps=1000,
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    gradient_checkpointing=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=8192,
)

trainer.train()
loader.save_lora(model, tokenizer, "./saved_loras/reasoner_sft")
