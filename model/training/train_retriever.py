"""
train_retriever.py —— Retriever (BGE-M3) LoRA 微调
====================================================
模型：BAAI/bge-m3 + LoRA (仅微调 q_proj/v_proj)
损失：MultipleNegativesRankingLoss (三元组: query, positive, negative)
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import Dataset
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from peft import get_peft_model, LoraConfig, TaskType

# ============================================================================
# 1. 数据加载
# ============================================================================
data_path = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "training_data", "generated", "retriever_triplets.jsonl"
)

if os.path.exists(data_path):
    import json
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
else:
    print("⚠️ 使用示例三元组数据")
    data = [
        {"query": "试用期最后一天被辞退，赔偿怎么算？",
         "pos": "用人单位在试用期解除劳动合同需证明不符合录用条件。",
         "neg": "正常辞退应提前30天通知。"},
        {"query": "拖欠工资三个月能要求赔偿吗？",
         "pos": "用人单位拖欠劳动报酬的，劳动者可以解除合同并要求经济补偿。",
         "neg": "劳动者应当提前三十日书面通知用人单位。"},
        {"query": "没签劳动合同被开除怎么办？",
         "pos": "未签订劳动合同，用人单位应支付双倍工资差额。",
         "neg": "劳动合同期满自动续签。"},
    ]

# 转换为 InputExample
train_examples = [
    InputExample(texts=[d['query'], d['pos'], d['neg']]) for d in data
]

# ============================================================================
# 2. 加载模型并挂载 LoRA
# ============================================================================
model = SentenceTransformer("BAAI/bge-m3")

lora_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    r=8, lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
)

# BGE-M3 底层 AutoModel 挂载 LoRA
model._modules["0"].auto_model = get_peft_model(
    model._modules["0"].auto_model, lora_config
)

# ============================================================================
# 3. 训练
# ============================================================================
train_loss = losses.MultipleNegativesRankingLoss(model)
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    optimizer_params={'lr': 2e-5},
)

# ============================================================================
# 4. 保存
# ============================================================================
output_path = "./saved_loras/retriever_bge_m3"
model.save(output_path)
print(f"Retriever LoRA 已保存至 {output_path}")
print("使用方式 (在图引擎中加载): graph_engine.load_lora_from_sentence_transformers(output_path)")
