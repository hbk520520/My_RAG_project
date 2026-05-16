import re
import time
import json
import logging
from typing import Dict, Any, Tuple

# 模拟之前的 DeepSeek 基础模型调用
# from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger("SemanticRouter")

class IntentClassifier:
    def __init__(self, api_key: str = "mock_key"):
        # self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        # ==========================================
        # L0 防线：极速正则表达式引擎 (0 成本)
        # ==========================================
        # 1. 闲聊模式 (极其死板但极度高效)
        self.chitchat_regex = re.compile(
            r"^(你好|在吗|哈喽|hello|hi|嗨|你是谁|谁开发的|谢谢|拜拜|再见|早安|晚安)[！。？~\s]*$", 
            re.IGNORECASE
        )
        
        # 2. 简单问答模式 (匹配典型的事实查询句式)
        self.simple_qa_regex = re.compile(
            r"^(什么是|解释一下|告诉我|.*的概念|.*的定义是|.*出台时间是|.*?属于什么法)[\?？\s]*$"
        )

        # 启发式阈值：超过此字数，强制认为案情复杂，不走简单 QA
        self.complex_length_threshold = 80 

    def _llm_fallback_route(self, query: str) -> str:
        """
        L1 防线：大模型兜底路由
        只有当正则引擎无法判定时才触发，将 Token 消耗降到最低
        """
        system_prompt = """
        你是一个毫无法感的意图分类路由器。
        请将用户的输入严格分类为以下三种之一，只输出 JSON，不要任何废话。
        
        1. "CHITCHAT" (闲聊)：与法律毫无关系的打招呼、夸赞或无意义字符。
        2. "SIMPLE_QA" (简单问答)：单一法律事实查询，不需要结合背景，如“劳动合同法第十四条规定了什么”、“法定退休年龄是多少”。
        3. "COMPLEX_TASK" (复杂推演)：包含了用户具体的背景故事、多项事实条件、需要进行逻辑推演或计算的咨询，如“我在公司干了三年，昨天没去被开除了，能赔多少？”
        
        输出格式：{"intent": "选择的意图"}
        """
        # 真实环境中这里调用 API
        # response = self.client.chat.completions.create(
        #     model="deepseek-chat",
        #     messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query}],
        #     temperature=0.0, response_format={"type": "json_object"}
        # )
        # return json.loads(response.choices[0].message.content).get("intent", "COMPLEX_TASK")
        
        # 为了本地测试直接 mock LLM 的判断逻辑
        logger.debug("触发 LLM 兜底路由...")
        time.sleep(0.3) # 模拟 API 延迟
        if "多少钱" in query or "赔" in query or len(query) > 30:
            return "COMPLEX_TASK"
        return "SIMPLE_QA"

    def route(self, query: str) -> Tuple[str, str]:
        """
        核心网关：先走正则和启发式规则，再走 LLM
        返回: (意图类型, 命中防线来源)
        """
        query_stripped = query.strip()
        
        # ==========================================
        # 规则 1：长度启发式 (极速拦截超长案情)
        # ==========================================
        if len(query_stripped) >= self.complex_length_threshold:
            return "COMPLEX_TASK", "L0_Heuristics_Length"
            
        # ==========================================
        # 规则 2：正则引擎拦截闲聊
        # ==========================================
        if self.chitchat_regex.match(query_stripped):
            return "CHITCHAT", "L0_Regex"
            
        # ==========================================
        # 规则 3：正则引擎拦截标准问答
        # ==========================================
        if self.simple_qa_regex.match(query_stripped):
            return "SIMPLE_QA", "L0_Regex"
            
        # ==========================================
        # 规则 4：防线被击穿，移交 LLM 裁决
        # ==========================================
        intent = self._llm_fallback_route(query_stripped)
        return intent, "L1_LLM"

# ==========================================
# 阶段 2：单元测试与性能评估 (Test Suite)
# ==========================================
if __name__ == "__main__":
    router = IntentClassifier()
    
    test_cases = [
        # 闲聊组
        "你好！",
        "你是谁开发的人工智能？",
        # 简单 QA 组
        "什么是竞业限制？",
        "劳动法第十四条的规定是什么",
        # 复杂推演组 (必定触发长文本拦截)
        "律师你好，我 2021 年入职这家公司，前天老板口头通知我明天不用来了，但是一直没给我发解除劳动合同的书面通知，而且我的社保他们从上个月就断缴了，请问我能主张 2N 的赔偿金还是 N+1？",
        # 模糊地带 (正则无法覆盖，需要 LLM 兜底)
        "试用期被辞退能赔多少钱？"
    ]
    
    print("-" * 60)
    print(f"{'用户 Query':<35} | {'判定意图':<15} | {'路由耗时':<10} | {'命中防线'}")
    print("-" * 60)
    
    for tc in test_cases:
        start_time = time.perf_counter()
        intent, source = router.route(tc)
        cost_time = (time.perf_counter() - start_time) * 1000 # 转换为毫秒
        
        # 格式化输出对齐
        tc_display = tc[:32] + "..." if len(tc) > 32 else tc
        print(f"{tc_display:<35} | {intent:<15} | {cost_time:6.2f} ms | {source}")