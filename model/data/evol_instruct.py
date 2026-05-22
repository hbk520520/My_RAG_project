"""
Evol-Instruct 数据工厂 —— 用大模型造训练数据
==========================================
从真实法条反向出题：虚构具体案件→生成 Planner 步骤→提取关键事实→附带计算代码。
同时能注入口语化噪声和故意信息残缺，造出逼真的用户提问当测试集。

技术栈: DeepSeek API (AsyncOpenAI) / JSON mode / asyncio
"""
import json, asyncio, logging
from typing import List, Dict, Optional
from openai import AsyncOpenAI

logger = logging.getLogger("EvolInstruct")

# ============================================================================
# 配置（从 config.yaml / 环境变量读取）
# ============================================================================
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
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
SYNTHESIZE_PROMPT = """你是一个顶级的中国法律考试命题专家与数据标注工程师。
我会给你几段真实的法律条款。你必须严格基于这些条款，进行"反向出题与解答"。

【任务步骤】：
1. 虚构一个极其具体的案件（包含具体的人物、入职时间、工资数额、冲突事件）。
   案件必须刚好需要用到我提供的这些法律条款来解决。
2. 生成解决这个案件的 Meta-Planner 执行骨架（最多 4 步）。
3. 提取这个案件的关键事实要素（用于检验 Extractor）。
4. 如果案件涉及赔偿金/经济补偿的计算，请写出一段纯 Python 代码来计算最终结果，
   变量命名需清晰，且必须附带 `result = ...`。

【输出约束】：
严格输出以下 JSON 格式，不要包含任何 Markdown 代码块标签：
{
  "user_query": "虚构的用户提问...",
  "ground_truth": {
    "correct_planner_queue": ["步骤1", "步骤2"],
    "key_facts": ["事实1", "事实2"],
    "python_code": "def calc_compensation():...",
    "expected_result": 50000.0
  }
}
"""


async def synthesize_batch(anchor_laws_list: List[str]) -> List[Dict]:
    """批量生成 Ground Truth 数据"""
    results = []
    for anchor in anchor_laws_list:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYNTHESIZE_PROMPT},
                    {"role": "user", "content": f"【核心法律锚点】:\n{anchor}"}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
            )
            results.append(json.loads(resp.choices[0].message.content))
        except Exception as e:
            logger.error(f"合成失败 (anchor={anchor[:30]}...): {e}")
    return results


# ============================================================================
# 任务二：高噪点口语化 Query 生成（测试集 + 对抗样本）
# ============================================================================
EVOL_QUERY_PROMPT = """你是一个顶级的 AI 合成数据专家与社会心理学家。
我将提供一份真实的【中国劳动争议判决书/案件事实】。
你的任务是：深度代入该案中【劳动者】的视角，将干瘪的法律事实反向还原为 3 个逼真、口语化、甚至充满情绪的真实用户提问（Query）。

【角色扮演绝对铁律】：
1. 严禁使用法言法语：用"赔钱"、"开除"、"没签合同"、"告他"等生活化词汇替代专业术语。
2. 注入情绪与废话（高噪点）：真实用户常夹带私货，如"昨天气得没吃饭"、"同事都没事就针对我"。
3. 强制信息残缺：3 个问题中，必须有 2 个故意遗漏核心定案事实！

【输出格式】：
{
  "queries": [
    {
      "query_text": "劳动者大白话提问（带情绪和废话）",
      "noise_injected": "注入了什么噪音？",
      "missing_facts": "故意遗漏了什么关键事实？(没有则填 null)",
      "ground_truth_answer": "根据判决书推导的终极客观答案（<=50字）"
    }
  ]
}
"""


async def generate_noisy_queries(verdict_text: str) -> List[Dict]:
    """从判决书生成高噪点口语化测试用例"""
    try:
        resp = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": EVOL_QUERY_PROMPT},
                {"role": "user", "content": f"【案件原始事实】\n{verdict_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.8
        )
        return json.loads(resp.choices[0].message.content).get("queries", [])
    except Exception as e:
        logger.error(f"Query 生成失败: {e}")
        return []


# ============================================================================
# 任务三：Replanner GRPO 场景生成
# ============================================================================
REPLANNER_SCENARIO_PROMPT = """你是一个重规划场景生成器。
基于给定的法律问题，构造一个【检索失败】的场景，用于训练 Replanner 模型。

【场景要求】：
1. 前序 Plan 已部分执行但某步骤检索失败
2. 提供已有的 global_facts（已成功检索的事实）
3. 提供 fail_log（失败原因描述）
4. 提供 fail_count（已失败次数，1-4）

严格输出 JSON：
{
  "prompt": "作为 Replanner，用户问题：...\\n已有事实：...\\n失败记录：...\\n失败次数：N\\n请生成新的任务队列。",
  "reference_queue": [
    {"task_desc": "新查询步骤", "engine": "GRAPH_TRAVERSAL", "rationale": "理由"}
  ]
}
"""


async def generate_replanner_scenarios(cases: List[Dict]) -> List[Dict]:
    """生成 Replanner GRPO 训练场景"""
    results = []
    for case in cases:
        try:
            resp = await client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": REPLANNER_SCENARIO_PROMPT},
                    {"role": "user", "content": json.dumps(case, ensure_ascii=False)}
                ],
                response_format={"type": "json_object"},
                temperature=0.7
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
    output_path = os.path.join(os.path.dirname(__file__), "..", "..", "training_data", "generated")
    os.makedirs(output_path, exist_ok=True)

    with open(os.path.join(output_path, "planner_trajectories.jsonl"), "w", encoding="utf-8") as f:
        for item in gt_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"数据已保存至 {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
