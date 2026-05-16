from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. 配置模型与参数 (以训练 Meta-Planner 为例)
model_name = "Qwen/Qwen2.5-7B-Instruct" # 假设选用 Qwen2.5 7B 作为基座
max_seq_length = 4096

# 2. 极速加载基座模型 (自动 4-bit 量化)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_name,
    max_seq_length = max_seq_length,
    dtype = None, 
    load_in_4bit = True, # 核心：将 7B 模型压缩到不到 5GB 显存
)

# 3. 挂载 LoRA 权重
model = FastLanguageModel.get_peft_model(
    model,
    r = 64,               # Meta-Planner 推荐的容量
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                      "gate_proj", "up_proj", "down_proj"], # All-linear
    lora_alpha = 128,     # r 的 2 倍
    lora_dropout = 0,     # Unsloth 优化要求
    bias = "none",
    use_gradient_checkpointing = "unsloth", # 极限节省激活值显存
)

# 4. 加载与格式化数据集
dataset = load_dataset("json", data_files="planner_train_data.jsonl", split="train")

# 使用 tokenizer 自动套用对话模板 (极其关键，防止模型乱吐格式)
def format_chat_template(row):
    row["text"] = tokenizer.apply_chat_template(row["messages"], tokenize=False)
    return row

dataset = dataset.map(format_chat_template)

# 5. 配置 Trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4, # 模拟更大的 Batch Size (2*4=8)
        warmup_steps = 50,
        num_train_epochs = 3, # SFT 通常跑 2-3 个 Epoch
        learning_rate = 2e-4, # LoRA 标准学习率
        fp16 = not FastLanguageModel.is_bfloat16_supported(),
        bf16 = FastLanguageModel.is_bfloat16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit", # 8-bit 优化器，进一步省显存
        output_dir = "outputs/planner_lora_v1",
        save_strategy = "epoch",
    ),
)

# 6. 开始炼丹！
trainer.train()

# 7. 只保存 LoRA 权重 (不包含基座)
model.save_pretrained("saved_loras/planner_lora")
tokenizer.save_pretrained("saved_loras/planner_lora")
print("Meta-Planner LoRA 训练完成并保存！")