"""
Extractor + Grader 训练 —— DPO 对齐，教模型"别脑补"
=================================================
chosen 是谨慎的、基于事实的回答；rejected 是武断的、靠猜测的结论。
DPO 让模型学会：不确定就说"需要补充信息"，而不是拍脑袋说"肯定违法"。
3B 小模型就够用。

技术栈: DPOTrainer (trl) / Unsloth / LoRA
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import Dataset, load_dataset
from trl import DPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

from utils.unsloth_loader import UnslothLoader

# ============================================================================
# 1. 加载模型（Extractor/Grader 只需 3-4B 参数）
# ============================================================================
loader = UnslothLoader("extractor_grader")
model, tokenizer = loader.load()

# ============================================================================
# 2. DPO 数据加载
# ============================================================================
data_path = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "training_data", "generated", "extractor_pairs.jsonl"
)

if os.path.exists(data_path):
    dataset = load_dataset("json", data_files=data_path, split="train")
else:
    print("⚠️ 使用示例 DPO 数据")
    data = [
        {
            "prompt": "给定上下文：公司口头通知辞退，未出具书面证明。问：试用期辞退是否合法？",
            "chosen": "根据上下文，未提供公司是否证明不符合录用条件的事实，需要补充相关信息。",
            "rejected": "公司肯定违法，应赔偿2N工资。"  # 脑补、无依据 → rejected
        },
        {
            "prompt": "上下文：员工工作满2年，月薪6000。问：能拿多少经济补偿？",
            "chosen": "根据劳动合同法第47条，工作年限为2年，应支付2个月工资即12000元的经济补偿。但需确认解除原因是否符合第46条情形。",
            "rejected": "肯定能拿2N赔偿金24000元。"  # 未区分 N 和 2N → rejected
        },
    ]
    dataset = Dataset.from_list(data)

# ============================================================================
# 3. 训练配置
# ============================================================================
training_args = TrainingArguments(
    output_dir="./outputs/extractor_grader_dpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=1,
    logging_steps=10,
    bf16=True,
    report_to="none",
    save_strategy="epoch",
)

dpo_trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=dataset,
    beta=0.1,            # DPO 温度参数
    loss_type="sigmoid",
)

dpo_trainer.train()
loader.save_lora(model, tokenizer, "./saved_loras/extractor_grader_dpo")
