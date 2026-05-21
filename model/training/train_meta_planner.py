"""
train_meta_planner.py —— Meta-Planner SFT 训练 (v2 + SemanticDTW 数据筛选)
==========================================================================
模型：Qwen2.5-14B-Instruct + LoRA (Unsloth)
新增：SemanticDTWRewarder 语义 DTW 评分器 —— 用于 DPO 数据自动筛选
"""
import os, sys, json, logging
import numpy as np
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datasets import Dataset, load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported

from utils.unsloth_loader import UnslothLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MetaPlanner")


# ============================================================================
# 0. SemanticDTW 语义对齐评分器
# ============================================================================
class SemanticDTWRewarder:
    """语义 DTW：将 Planner 步骤序列与图数据库黄金路径对齐评分，产出 DPO 筛选信号"""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(model_name)

    def _get_embeddings(self, texts: List[str], prefix: str = "") -> np.ndarray:
        if not texts:
            return np.array([]).reshape(0, 1)
        prefixed = [f"{prefix}{t}" for t in texts]
        emb = self.encoder.encode(prefixed, convert_to_numpy=True)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        return emb / np.where(norms > 0, norms, 1e-8)

    def _similarity_matrix(self, plan_steps: List[str], graph_nodes: List[str]) -> np.ndarray:
        v_plan = self._get_embeddings(plan_steps, "为这个句子生成表示以用于检索相关文章：")
        u_graph = self._get_embeddings(graph_nodes, "")
        if v_plan.size == 0 or u_graph.size == 0:
            return np.zeros((len(plan_steps), len(graph_nodes)))
        return np.dot(v_plan, u_graph.T)

    def compute_dtw_reward(self, plan_trajectory: List[str],
                           graph_trajectory: List[str],
                           gamma_penalty: float = 0.15,
                           lambda_scale: float = 3.0) -> float:
        m, n = len(plan_trajectory), len(graph_trajectory)
        if m == 0 or n == 0:
            return 0.0
        M = self._similarity_matrix(plan_trajectory, graph_trajectory)
        DP = np.full((m + 1, n + 1), -np.inf)
        DP[0, 0] = 0.0
        for i in range(1, m + 1):
            DP[i, 0] = DP[i - 1, 0] - gamma_penalty
        for j in range(1, n + 1):
            DP[0, j] = DP[0, j - 1] - gamma_penalty
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                DP[i, j] = M[i - 1, j - 1] + max(
                    DP[i - 1, j - 1],
                    DP[i - 1, j] - gamma_penalty,
                    DP[i, j - 1] - gamma_penalty
                )
        normalized = max(0.0, DP[m, n] / min(m, n))
        return float(np.exp(-lambda_scale * (1.0 - normalized)))


# ============================================================================
# 1. 加载模型
# ============================================================================
loader = UnslothLoader("meta_planner")
model, tokenizer = loader.load()

# ============================================================================
# 2. 加载数据 + SemanticDTW 质量过滤
# ============================================================================
data_path = os.path.join(os.path.dirname(__file__), "..", "..",
                         "training_data", "generated", "planner_trajectories.jsonl")
if os.path.exists(data_path):
    raw_dataset = load_dataset("json", data_files=data_path, split="train")
    # DTW 筛选低质量样本
    try:
        rewarder = SemanticDTWRewarder()
        filtered = []
        rejected = 0
        for sample in raw_dataset:
            msgs = sample.get("messages", [])
            if len(msgs) < 2:
                continue
            try:
                plan_data = json.loads(msgs[-1].get("content", "{}"))
                plan_steps = plan_data.get("task_queue", [])
                graph_truth = sample.get("ground_truth", {}).get("key_facts", [])
                if graph_truth:
                    score = rewarder.compute_dtw_reward(plan_steps, graph_truth)
                    if score >= 0.3:
                        sample["dtw_score"] = score
                        filtered.append(sample)
                    else:
                        rejected += 1
                else:
                    filtered.append(sample)
            except Exception:
                filtered.append(sample)
        dataset = Dataset.from_list(filtered)
        logger.info(f"SemanticDTW 筛选: {len(filtered)} 保留, {rejected} 剔除")
    except Exception as e:
        logger.warning(f"DTW 跳过: {e}")
        dataset = raw_dataset
else:
    logger.warning(f"数据文件不存在: {data_path}，使用示例")
    dataset = Dataset.from_list([{
        "messages": [
            {"role": "user", "content": "俺被口头辞退了，能要多少钱？"},
            {"role": "assistant", "content": '{"task_queue": ["查核劳动关系", "查核辞退合法性", "计算赔偿金额"]}'}
        ],
        "ground_truth": {"key_facts": ["入职2022-03", "口头辞退", "月薪8000"]}
    }])

dataset = dataset.map(lambda x: loader.format_chat(x, tokenizer))

# ============================================================================
# 3. 训练
# ============================================================================
training_args = TrainingArguments(
    output_dir="./outputs/planner_lora",
    per_device_train_batch_size=2, gradient_accumulation_steps=4,
    warmup_steps=50, num_train_epochs=3, learning_rate=2e-4,
    fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(),
    logging_steps=10, optim="adamw_8bit",
    save_strategy="epoch", report_to="none",
)

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=dataset,
    dataset_text_field="text", max_seq_length=4096, args=training_args,
)

logger.info("🚀 Meta-Planner SFT 训练 (SemanticDTW 已筛选)...")
trainer.train()
loader.save_lora(model, tokenizer, "./saved_loras/planner_lora")
logger.info("✅ 完成!")
