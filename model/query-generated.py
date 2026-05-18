import json
import asyncio
from typing import List, Dict
from openai import AsyncOpenAI

# ==========================================
# 0. 基础配置
# ==========================================
API_KEY = "sk-xxxxxxxxxxxxxxxx"  # 替换为你的真实 Key
client = AsyncOpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

# ==========================================
# 1. 核心资产：2026 级工业演化 Prompt
# ==========================================
EVOL_INSTRUCT_PROMPT = """你是一个顶级的 AI 合成数据专家与社会心理学家。
我将提供一份真实的【中国劳动争议判决书/案件事实】。
你的任务是：深度代入该案中【劳动者】的视角，将干瘪的法律事实反向还原为 3 个逼真、口语化、甚至充满情绪的真实用户提问（Query）。

【角色扮演绝对铁律 - Persona Constraints】：
1. 严禁使用法言法语：绝不允许在提问中出现“经济补偿金”、“违法解除劳动合同”、“二倍工资”、“仲裁时效”等专业词汇。用“赔钱”、“开除”、“没签合同”、“告他”等生活化词汇替代。
2. 注入情绪与废话（高噪点）：真实用户常常夹带私货。例如抱怨“昨天气得没吃饭”、“同事都没事就针对我”、“家里还有小孩要养”。这些噪音能极大测试检索系统的抗干扰能力。
3. 强制信息残缺（残缺度）：真实用户提问绝对不可能一次性把“入职时间、工资基数、解除理由”交代清楚。你生成的 3 个问题中，必须有 2 个故意遗漏核心定案事实！

【输出格式约束】：
你必须输出且仅输出一个合法的 JSON 对象，格式如下：
{
  "queries": [
    {
      "query_text": "生成的劳动者大白话提问（带有情绪和废话）",
      "noise_injected": "注入了什么噪音/废话？（简述，如'提及同事的待遇作为对比'）",
      "missing_facts": "故意遗漏了什么关键前置事实？（如'未说明试用期是否结束'，没有遗漏填 null）",
      "ground_truth_answer": "根据判决书推导出的终极客观答案（不超过50字，作为这道题的金标准）"
    }
  ] # 数组内必须有 3 个对象
}
"""

# ==========================================
# 2. 异步生成流水线
# ==========================================
async def generate_realistic_queries(verdict_text: str) -> List[Dict]:
    """
    接收判决书文本，异步调用 DeepSeek 生成高噪点测试集
    """
    print(f"⏳ [阶段一] 正在解析判决书并生成高噪点提问...")
    
    try:
        response = await client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": EVOL_INSTRUCT_PROMPT},
                {"role": "user", "content": f"【案件原始事实】\n{verdict_text}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.8  # 稍微调高温度，增加提问的多样性和口语化程度
        )
        
        raw_json = response.choices[0].message.content
        parsed_data = json.loads(raw_json)
        
        queries = parsed_data.get("queries", [])
        print(f"✅ [阶段一] 成功生成 {len(queries)} 条黄金测试用例。")
        return queries
        
    except json.JSONDecodeError as e:
        print(f"❌ [数据熔断] JSON 解析失败: {e}")
        return []
    except Exception as e:
        print(f"❌ [网络异常] API 调用失败: {e}")
        return []

# ==========================================
# 3. 模拟执行入口
# ==========================================
async def main():
    # 模拟从数据库里抽出来的一份干巴巴的判决书事实
    sample_verdict = """
    原告张三于2022年3月1日入职被告A公司，担任销售专员，双方未签订书面劳动合同。
    原告每月基本工资4000元，提成另算，通过银行转账发放。
    2024年1月15日，A公司以张三‘连续三个月未完成销售KPI’为由，通过微信口头通知其明天不用来了，未出具解除劳动合同证明，未支付任何经济补偿。
    经查，A公司并未在规章制度中明确规定销售KPI未达标可直接解除劳动合同。
    法院最终判决A公司属违法解除劳动合同，需支付经济赔偿金及未签劳动合同二倍工资差额。
    """
    
    generated_data = await generate_realistic_queries(sample_verdict)
    
    # 打印成果
    print("\n" + "="*50)
    for idx, item in enumerate(generated_data, 1):
        print(f"【用例 {idx}】")
        print(f"🗣️ 劳动者原话 (Query): {item['query_text']}")
        print(f"🌪️ 噪音分析: {item['noise_injected']}")
        print(f"🕳️ 信息残缺: {item.get('missing_facts')}")
        print(f"🎯 标准答案 (GT): {item['ground_truth_answer']}\n")

if __name__ == "__main__":
    # 启动异步事件循环
    asyncio.run(main())