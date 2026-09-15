"""
Evol-Instruct 数据工厂 —— 用大模型造训练数据
==========================================
从真实法条反向出题：虚构具体案件→生成双层蓝图→提取关键事实→附带计算代码。
同时能注入口语化噪声和故意信息残缺，造出逼真的用户提问当测试集。

阶段 3 变更：
  1. 所有 Prompt 正文统一由 prompts.py 提供，本模块不再保留副本
  2. Ground Truth 里的 Planner 产物改为「双层蓝图」P_q={S_q,C_q}，
     与在线推理、训练侧的解析口径对齐（原先这里是扁平 task_queue）

技术栈: DeepSeek API (AsyncOpenAI) / JSON mode / asyncio
"""
import json
import asyncio
import logging
import os
import sys
from typing import List, Dict, Optional
from openai import AsyncOpenAI

logger = logging.getLogger("EvolInstruct")

# 路径引导：让根目录的 config_loader / prompts 可被导入
_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

import prompts  # 阶段 3：Prompt 单一真源

# ============================================================================
# 配置（从 config.yaml / 环境变量读取）
# ============================================================================
try:
    from config_loader import cfg
    API_KEY = cfg.get("llm", "api_key")
    BASE_URL = cfg.get("llm", "base_url")
    MODEL = cfg.get("llm", "judge_model")
except Exception:
    API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
    BASE_URL = "https://api.deepseek.com"
    MODEL = "deepseek-chat"

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)


# ============================================================================
# 任务一：从法条生成 Ground Truth（Meta-Planner + Extractor 训练数据）
# ============================================================================
async def synthesize_batch(anchor_laws_list: List[str]) -> List[Dict]:
    """
    批量生成 Ground Truth 数据。

    模板：prompts.SYNTHESIZE_GROUND_TRUTH_SYSTEM
    产出结构：{"user_query": ..., "ground_truth": {"correct_planner_plan": {S_q, C_q}, ...}}
    """
    results = []
    for anchor in anchor_laws_list:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system",
                     "content": prompts.SYNTHESIZE_GROUND_TRUTH_SYSTEM},
                    {"role": "user", "content": f"【核心法律锚点】:\n{anchor}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            results.append(json.loads(resp.choices[0].message.content))
        except Exception as e:
            logger.error(f"合成失败 (anchor={anchor[:30]}...): {e}")
    return results


# ============================================================================
# 任务二：高噪点口语化 Query 生成（测试集 + 对抗样本）
# ============================================================================
async def generate_noisy_queries(verdict_text: str) -> List[Dict]:
    """从判决书生成高噪点口语化测试用例（模板：prompts.EVOL_INSTRUCT_SYSTEM）"""
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompts.EVOL_INSTRUCT_SYSTEM},
                {"role": "user", "content": f"【案件原始事实】\n{verdict_text}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.8,
        )
        return json.loads(resp.choices[0].message.content).get("queries", [])
    except Exception as e:
        logger.error(f"Query 生成失败: {e}")
        return []


# ============================================================================
# 任务三：Replanner GRPO 场景生成
# ============================================================================
async def generate_replanner_scenarios(cases: List[Dict]) -> List[Dict]:
    """生成 Replanner GRPO 训练场景（模板：prompts.REPLANNER_SCENARIO_SYSTEM）"""
    results = []
    for case in cases:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": prompts.REPLANNER_SCENARIO_SYSTEM},
                    {"role": "user", "content": json.dumps(case, ensure_ascii=False)},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )
            results.append(json.loads(resp.choices[0].message.content))
        except Exception as e:
            logger.error(f"场景生成失败: {e}")
    return results


# ============================================================================
# 批量运行入口
# ============================================================================
async def main():
    # 示例：从法条生成训练数据
    sample_laws = [
        "劳动合同法第47条：经济补偿按劳动者在本单位工作的年限，每满一年支付一个月工资的标准向劳动者支付。",
        "劳动合同法第87条：用人单位违反本法规定解除或者终止劳动合同的，应当依照本法第47条规定的经济补偿标准的二倍向劳动者支付赔偿金。",
    ]

    gt_data = await synthesize_batch(sample_laws)
    print(f"生成 {len(gt_data)} 条 Ground Truth 数据")

    # 保存
    output_path = os.path.join(_ROOT_DIR, "training_data", "generated")
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "planner_trajectories.jsonl"), "w", encoding="utf-8") as f:
        for item in gt_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"数据已保存至 {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
