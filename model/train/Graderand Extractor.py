# train_extractor_dpo.py
from datasets import Dataset
from trl import DPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

# 1. SFT 冷启动（可选，此处直接演示 DPO）
# 假设已有一个 SFT 模型路径
model_path = "./extractor-sft"  # 或原始基座

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

# 2. DPO 数据：chosen/rejected 对
data = [
    {
        "prompt": "给定上下文：... 问：试用期辞退是否合法？",
        "chosen": "根据上下文，未提供公司是否证明不符合录用条件的事实，需要补充相关信息。",
        "rejected": "公司肯定违法，应赔偿2N工资。"  # 脑补、无依据
    }
]

# 转换为 DPO 格式
def format_dpo(example):
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"]
    }
ds = Dataset.from_list(data).map(format_dpo)

training_args = TrainingArguments(
    output_dir="./extractor-dpo",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    num_train_epochs=1,
    logging_steps=10,
    bf16=True,
    report_to="none"
)

dpo_trainer = DPOTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=ds,
    beta=0.1,            # DPO 温度参数
    loss_type="sigmoid"
)
dpo_trainer.train()
dpo_trainer.save_model("extractor-dpo-final")