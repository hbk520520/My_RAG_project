import json
import logging
# from openai import OpenAI

logger = logging.getLogger("GroundTruthSynthesizer")

def synthesize_ground_truth_batch(anchor_laws: str, client) -> dict:
    """
    输入真实的法律条款（锚点），输出包含复杂案情和全套 Ground Truth 的 JSON
    """
    system_prompt = """
    你是一个顶级的中国法律考试命题专家与数据标注工程师。
    我会给你几段真实的法律条款。你必须严格基于这些条款，进行“反向出题与解答”。
    
    【任务步骤】：
    1. 虚构一个极其具体的案件（包含具体的人物、入职时间、工资数额、冲突事件）。案件必须刚好需要用到我提供的这些法律条款来解决。
    2. 生成解决这个案件的 Meta-Planner 执行骨架（最多 4 步）。
    3. 提取这个案件的关键事实要素（用于检验 Extractor）。
    4. 如果案件涉及赔偿金/经济补偿的计算，请写出一段纯 Python 代码来计算最终结果，变量命名需清晰，且必须附带 `result = ...`。
    
    【输出约束】：
    严格输出以下 JSON 格式，不要包含任何 Markdown 代码块标签，不要任何解释性废话：
    {
      "user_query": "虚构的用户提问...",
      "ground_truth": {
        "correct_planner_queue": ["步骤1", "步骤2"],
        "key_facts": ["事实1", "事实2"],
        "python_code": "def calc_compensation():...",
        "expected_result": 50000.0  // 如果无计算则为 null
      }
    }
    """
    
    # 真实环境调用 (开启 JSON Mode 确保输出 100% 可被代码解析)
    # response = client.chat.completions.create(
    #     model="deepseek-chat",
    #     messages=[
    #         {"role": "system", "content": system_prompt},
    #         {"role": "user", "content": f"【核心法律锚点】:\n{anchor_laws}"}
    #     ],
    #     temperature=0.7, # 稍微给点温度，让虚构的案情更加多样化
    #     response_format={"type": "json_object"}
    # )
    # return json.loads(response.choices[0].message.content)
    
    # ---------------- 模拟成功生成的 Ground Truth 数据 ----------------
    return {
      "user_query": "我 2021年3月1日入职，月薪 8000 元。昨天（2026年5月10日）公司以‘部门取消’为由直接让我走人，没提前通知，我能要多少钱？",
      "ground_truth": {
        "correct_planner_queue": [
            "判定解除劳动合同的合法性与性质", 
            "核算劳动者在本单位的工作年限", 
            "计算代通知金（N+1）或违法解除赔偿金（2N）"
        ],
        "key_facts": ["入职时间：2021-03-01", "离职时间：2026-05-10", "月薪：8000", "解除原因：客观情况发生重大变化且未提前30日通知"],
        "python_code": "months = 5.5\nbase = 8000\nresult = (months + 1) * base",
        "expected_result": 52000.0
      }
    }