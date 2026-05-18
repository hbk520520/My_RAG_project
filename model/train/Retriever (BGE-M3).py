# train_retriever.py
from datasets import Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType

# 1. 数据准备 (三元组: query, positive, negative)
data = [
    {"query": "试用期最后一天被辞退，赔偿怎么算？",
     "pos": "用人单位在试用期解除劳动合同需证明不符合录用条件。",
     "neg": "正常辞退应提前30天通知。"}
]
# 转换为 InputExample
train_examples = []
for d in data:
    train_examples.append(InputExample(texts=[d['query'], d['pos'], d['neg']]))

train_dataset = Dataset.from_list(data)

# 2. 加载 BGE-M3 模型
model = SentenceTransformer("BAAI/bge-m3")
# 应用 LoRA (仅微调 Attention 的 Q/V 投影)
lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj"],  # BGE-M3 内部模块名需根据模型架构调整
    lora_dropout=0.1
)
model._modules["0"].auto_model = get_peft_model(model._modules["0"].auto_model, lora_config)

# 3. 定义损失
train_loss = losses.MultipleNegativesRankingLoss(model)

# 4. 训练
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    optimizer_params={'lr': 2e-5}
)

model.save("retriever-bge-m3-lora")