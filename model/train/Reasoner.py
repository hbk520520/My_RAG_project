# train_reasoner_long_sft.py
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset

model_name = "Qwen/Qwen2.5-7B-Instruct"  # 或更长的上下文模型如 Qwen2.5-14B

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    attn_implementation="flash_attention_2",   # FlashAttention-2
    use_cache=False,                           # 梯度检查点要求关闭 KV cache
)

# 开启梯度检查点
model.gradient_checkpointing_enable()

# 数据：超长上下文问答对
data = [
    {
        "text": tokenizer.apply_chat_template([
            {"role": "system", "content": "你是一名资深法官助理..."},
            {"role": "user", "content": "案件事实：...(长文本)"},
            {"role": "assistant", "content": "IRAC 分析：...(法理报告)"}
        ], tokenize=False)
    }
]
ds = Dataset.from_list(data)

training_args = TrainingArguments(
    output_dir="./reasoner-long-sft",
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
    deepspeed=None,          # 根据硬件配置
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    tokenizer=tokenizer,
)
trainer.train()
model.save_pretrained("reasoner-final")