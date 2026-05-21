"""
train_replanner_grpo.py —— Replanner GRPO 强化训练 (v2 Tournament)
==================================================================
模型：Qwen2.5-14B-Instruct + LoRA (Unsloth)
算法：GRPO + 8→4→2 三层淘汰赛制
PRM  ：BGE-Reranker-v2-m3 毫秒级过程奖励模型

核心：每个 prompt 生成 8 候选 → 逐跳淘汰 → 幸存者得终局奖金
"""
import os, sys, json, torch, logging
import numpy as np
from typing import List, Dict, Any
from datasets import Dataset, load_dataset
from trl import GRPOTrainer
from transformers import TrainingArguments

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.unsloth_loader import UnslothLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GRPO_Tournament")

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False
    logger.warning("sentence_transformers 未安装，pip install sentence-transformers")


# ============================================================================
# 1. GRPO 淘汰赛评估器
# ============================================================================
class GRPOTournamentEvaluator:
    """8→4→2 淘汰赛裁判，BGE-Reranker 作为 PRM"""

    def __init__(self, prm_model: str = "BAAI/bge-reranker-v2-m3"):
        self.has_prm = HAS_CROSS_ENCODER
        self.prm_encoder = CrossEncoder(prm_model) if self.has_prm else None
        if self.has_prm:
            logger.info(f"PRM 加载完成: {prm_model}")

    def score_context_relevance(self, task_desc: str, retrieved_docs: str) -> float:
        if not retrieved_docs or not task_desc:
            return 0.0
        if self.prm_encoder is not None:
            return self.prm_encoder.predict([(task_desc, retrieved_docs[:1024])])[0]
        overlap = len(set(task_desc) & set(retrieved_docs))
        return min(overlap / max(len(task_desc), 1), 1.0)

    @staticmethod
    def _mock_retrieve(task_desc: str, global_facts: List[str]) -> str:
        keywords = task_desc.replace("核查","").replace("核实","").replace("计算","")
        matched = [f for f in global_facts if any(kw in f for kw in keywords.split())]
        return "\n".join(matched) if matched else "\n".join(global_facts)[:200]

    def evaluate_rollout_batch(self, candidate_plans: List[Dict],
                               initial_state: Dict, global_facts: List[str]) -> List[float]:
        N = len(candidate_plans)
        rewards = np.zeros(N)
        survivors = list(range(N))

        # Round 1: Hop 1 (8→4)
        logger.info(f"Round 1: Hop 1 ({len(survivors)} alive)")
        hop1_scores = []
        for idx in survivors:
            plan = candidate_plans[idx].get("task_queue", [])
            if not plan:
                hop1_scores.append(-10.0); continue
            task_1 = plan[0]
            docs = self._mock_retrieve(task_1.get("task_desc",""), global_facts)
            s = self.score_context_relevance(task_1.get("task_desc",""), docs)
            hop1_scores.append(s); rewards[idx] += s * 5.0
        top_4 = np.argsort(hop1_scores)[-4:]
        for idx in survivors:
            if idx not in top_4: rewards[idx] -= 5.0
        survivors = [survivors[i] for i in top_4]

        # Round 2: Hop 2 (4→2)
        logger.info(f"Round 2: Hop 2 ({len(survivors)} alive)")
        hop2_scores = []
        for idx in survivors:
            plan = candidate_plans[idx].get("task_queue", [])
            if len(plan) < 2:
                hop2_scores.append(0.0); continue
            task_2 = plan[1]
            docs = self._mock_retrieve(task_2.get("task_desc",""), global_facts)
            s = self.score_context_relevance(task_2.get("task_desc",""), docs)
            hop2_scores.append(s); rewards[idx] += s * 5.0
        top_2 = np.argsort(hop2_scores)[-2:]
        for idx in survivors:
            if idx not in [survivors[i] for i in top_2]: rewards[idx] -= 2.0
        survivors = [survivors[i] for i in top_2]

        # Final Round (2 survivors)
        logger.info(f"Final Round ({len(survivors)} alive)")
        for idx in survivors:
            plan = candidate_plans[idx].get("task_queue", [])
            tasks_text = " ".join(t.get("task_desc","") for t in plan)
            dims = ["劳动关系","解除","辞退","赔偿","补偿","工资","合同"]
            covered = sum(1 for kw in dims if kw in tasks_text)
            has_wormhole = any(t.get("engine")=="GLOBAL_DENSE_WORMHOLE" for t in plan)
            if covered >= 2 or has_wormhole:
                rewards[idx] += 15.0
            else:
                rewards[idx] -= 5.0
        return rewards.tolist()


# ============================================================================
# 2. 模型加载
# ============================================================================
sft_path = "./outputs/replanner_sft"
loader = UnslothLoader("replanner", custom_model_name=sft_path if os.path.exists(sft_path) else None)
model, tokenizer = loader.load()
tournament = GRPOTournamentEvaluator()


# ============================================================================
# 3. Reward Function
# ============================================================================
def reward_func(prompts: List[str], completions: List[str]) -> List[torch.Tensor]:
    candidate_plans = []
    for c in completions:
        try:
            candidate_plans.append(json.loads(c.strip()))
        except Exception:
            candidate_plans.append({})

    global_facts = []
    for p in prompts:
        if "已有事实：" in p:
            facts = p.split("已有事实：")[-1].split("失败记录")[0].strip()
            global_facts.append([facts])
        else:
            global_facts.append([])

    flat_facts = []
    for gf in global_facts:
        flat_facts.append("\n".join(gf) if isinstance(gf, list) else str(gf))

    batch_rewards = tournament.evaluate_rollout_batch(candidate_plans, {}, flat_facts)
    return [torch.tensor(r, dtype=torch.float) for r in batch_rewards]


# ============================================================================
# 4. 数据 & 训练
# ============================================================================
data_path = os.path.join(os.path.dirname(__file__), "..", "..",
                         "training_data", "generated", "replanner_scenarios.jsonl")
if os.path.exists(data_path):
    ds = load_dataset("json", data_files=data_path, split="train")
else:
    prompts_data = [
        "用户问题：试用期最后一天被辞退。已有事实：确认劳动关系，月薪8000。失败记录：检索'试用期辞退赔偿'未找到。失败次数：2。请生成新任务队列。",
        "用户问题：拖欠3个月工资。已有事实：合同月薪10000，未缴社保。失败记录：检索'拖欠工资'返回空。失败次数：3。请生成新任务队列。",
    ]
    ds = Dataset.from_dict({"prompt": prompts_data})

training_args = TrainingArguments(
    output_dir="./outputs/replanner_grpo", per_device_train_batch_size=1,
    gradient_accumulation_steps=8, num_train_epochs=1, learning_rate=1e-5,
    logging_steps=10, bf16=True, report_to="none",
    remove_unused_columns=False, save_strategy="epoch",
)

grpo_trainer = GRPOTrainer(
    model=model, args=training_args, train_dataset=ds,
    tokenizer=tokenizer, reward_funcs=reward_func,
    num_generation_per_prompt=8, max_new_tokens=512,
)

logger.info("🏟️ GRPO 淘汰赛训练开始 (8→4→2)...")
grpo_trainer.train()
loader.save_lora(model, tokenizer, "./saved_loras/replanner_grpo")
logger.info("✅ 完成!")
