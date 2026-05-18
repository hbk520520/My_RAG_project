# train_planner.py
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# 数据格式：{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "{\"task_queue\": [...]}"}]}
data = [
    {"messages": [
        {"role": "user", "content": "俺被口头辞退了，能要多少钱？"},
        {"role": "assistant", "content": '{"task_queue": ["查核是否存在劳动关系", "查核辞退是否合法", "计算赔偿金额"]}'}
    ]}
]

# 转换为 Dataset
def format_example(ex):
    return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}

ds = Dataset.from_list(data).map(format_example)

# 加载基座模型 (推荐 Qwen2.5-7B-Instruct)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map="auto")

training_args = TrainingArguments(
    output_dir="./planner-sft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    bf16=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    tokenizer=tokenizer,
    data_collator=None  # 使用默认 collator
)
trainer.train()
model.save_pretrained("planner-final")