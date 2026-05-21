"""
训练数据管理与版本控制建议
============================
当前问题：训练数据管线未工程化，数据散落在各脚本中，无版本管理。

建议实施方案：
"""

# ============================================================================
# 1. 目录结构建议
# ============================================================================
SUGGESTED_STRUCTURE = """
training_data/
├── raw/                          # 原始数据（只读，永不修改）
│   ├── verdicts/                 # 判决书原始文本
│   │   └── 2024_q1_labor.jsonl
│   └── law_articles/             # 法条原文
│       └── labor_law_articles.json
├── generated/                    # Evol-Instruct 生成的数据
│   ├── planner_trajectories.jsonl
│   ├── extractor_pairs.jsonl
│   ├── reasoner_qa.jsonl
│   └── replanner_scenarios.jsonl
├── processed/                    # 经过清洗/验证的数据
│   ├── planner_sft_train.jsonl
│   ├── extractor_dpo_train.jsonl
│   └── reasoner_sft_train.jsonl
├── splits/                       # 训练/验证/测试集划分
│   ├── train/
│   ├── val/
│   └── test/
├── dvc.yaml                      # DVC 流水线定义
└── README.md                     # 数据文档
"""

# ============================================================================
# 2. 数据版本管理（DVC）
# ============================================================================
DVC_PIPELINE_EXAMPLE = """
# dvc.yaml
stages:
  generate_planner_data:
    cmd: python model/path-from-ds.py
    deps:
      - model/path-from-ds.py
      - training_data/raw/verdicts/
    outs:
      - training_data/generated/planner_trajectories.jsonl

  validate_planner_data:
    cmd: python -m scripts.validate_training_data --type planner
    deps:
      - training_data/generated/planner_trajectories.jsonl
    outs:
      - training_data/processed/planner_sft_train.jsonl

  split_dataset:
    cmd: python -m scripts.split_dataset --input training_data/processed/
    deps:
      - training_data/processed/
    outs:
      - training_data/splits/
"""

# ============================================================================
# 3. 数据质量验证脚本骨架
# ============================================================================
VALIDATION_CODE_EXAMPLE = '''
"""训练数据质量验证器"""
import json
import re
from typing import List, Dict

class TrainingDataValidator:
    """验证 Evol-Instruct 生成的训练数据质量"""

    @staticmethod
    def validate_planner_data(samples: List[Dict]) -> Dict[str, int]:
        """验证 Meta-Planner 训练数据"""
        stats = {"total": len(samples), "valid": 0, "invalid_json": 0,
                 "empty_queue": 0, "too_many_steps": 0, "step_too_short": 0}

        for s in samples:
            try:
                msgs = s.get("messages", [])
                if len(msgs) < 2:
                    stats["invalid_json"] += 1
                    continue

                assistant_msg = msgs[-1].get("content", "")
                parsed = json.loads(assistant_msg)
                task_queue = parsed.get("task_queue", [])

                if not task_queue:
                    stats["empty_queue"] += 1
                    continue
                if len(task_queue) > 8:
                    stats["too_many_steps"] += 1
                    continue
                if any(len(t) < 3 for t in task_queue):
                    stats["step_too_short"] += 1
                    continue

                stats["valid"] += 1
            except json.JSONDecodeError:
                stats["invalid_json"] += 1

        return stats

    @staticmethod
    def validate_extractor_dpo_data(samples: List[Dict]) -> Dict[str, int]:
        """验证 Extractor DPO 数据：chosen 应比 rejected 更谨慎/基于事实"""
        stats = {"total": len(samples), "valid": 0,
                 "chosen_longer_than_rejected": 0,
                 "rejected_contains_臆断": 0}

        for s in samples:
            chosen = s.get("chosen", "")
            rejected = s.get("rejected", "")

            # chosen 应该更谨慎（通常更长，因为有更多限定词）
            if len(chosen) > len(rejected):
                stats["chosen_longer_than_rejected"] += 1

            # rejected 通常包含武断结论
            臆断_keywords = ["肯定违法", "一定", "绝对", "必须赔偿", "毋庸置疑"]
            if any(kw in rejected for kw in 臆断_keywords):
                stats["rejected_contains_臆断"] += 1

            stats["valid"] += 1

        return stats
'''

# ============================================================================
# 4. 数据生成流水线建议
# ============================================================================
PIPELINE_RECOMMENDATION = """
推荐数据生成流程：

第一阶段：Seed 数据生成
  1. 从真实判决书中提取法律锚点（法条 + 事实）
  2. 调用 DeepSeek API（temperature=0.7）批量生成 1000 条初始数据
  3. 人工抽检 100 条，合格率 > 90% 则进入下一阶段

第二阶段：质量增强
  4. 使用法官模型交叉验证：
     - 同一法条生成 3 条 query → 检查答案一致性
     - 不一致的判定为低质量，丢弃或人工复核
  5. 注入对抗样本：10% 的数据故意加入错误的 ground truth，用于鲁棒性测试

第三阶段：数据版本化
  6. dvc add training_data/generated/
  7. git commit + dvc push
  8. 记录数据版本与模型版本的对应关系

数据量建议：
  - Meta-Planner SFT:  3000+ 条（含 20% 噪声注入测试样本）
  - Extractor DPO:     2000+ 对 chosen/rejected
  - Replanner GRPO:    1000+ 个失败场景 prompt
  - Reasoner SFT:      2000+ 条长上下文问答
  - Retriever MNR:     5000+ 个三元组 (query, pos, neg)
"""

print("训练数据工程化建议已生成。")
print(SUGGESTED_STRUCTURE)
print(DVC_PIPELINE_EXAMPLE)
print(PIPELINE_RECOMMENDATION)
