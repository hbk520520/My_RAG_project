import os
import json
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

FINETUNED_MODEL_PATH = " /path/to/your/finetuned/model"  # 替换为实际路径
CORPUS_DIR = "/path/to/your/corpus/directory"  # 替换为实际路径
OUTPUT_NODES_FILE = ""
OUTPUT_VECTORS_FILE = ""

print("\n[准备阶段] 正在扫描法律文书库，生成图谱节点...")
corpus_nodes = []
corpus_texts = []
node_id_counter = 1

for filename in os.listdir(CORPUS_DIR):
    if filename.endswith(".txt"):
        filepath = os.path.join(CORPUS_DIR, filename)
        law_source = os.path.splitext(filename)[0]
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if len(text) > 10:
                    node_data = {
                        "id": node_id_counter,
                        "content": text,
                        "type": "Raw",
                        "metadata": {
                            "source_file": law_source,
                            "length": len(text)
                        }
                    }
                    corpus_nodes.append(node_data)
                    corpus_texts.append(text)
                    node_id_counter += 1

print(f"总计加载了 {len(corpus_nodes)} 个标准节点。")

print(f"\n[加载阶段] 正在将专属法律大模型加载至显存...")
model = SentenceTransformer(FINETUNED_MODEL_PATH, device='cuda')
model.max_seq_length = 1024

print(f"\n[计算阶段] 开始全速向量化！")
with torch.no_grad():
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=1024,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

print(f"\n[保存阶段] 向量计算完成，矩阵维度: {corpus_embeddings.shape}")
np.save(OUTPUT_VECTORS_FILE, corpus_embeddings)
with open(OUTPUT_NODES_FILE, 'w', encoding='utf-8') as f:
    json.dump(corpus_nodes, f, ensure_ascii=False, indent=2)

print(f"节点与向量对齐完成，随时准备注入图引擎！")