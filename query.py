"""
入口路由层 —— 整个系统的"前台接待"
=============================
每一条用户消息都先经过这里。不用大模型，只用正则和一个小分类器判断意图，
闲聊就礼貌回复，简单问题走 RAG，复杂案情才唤醒后面的 Agent 引擎。

技术栈: scikit-learn (LogisticRegression) / sentence-transformers / re
"""
import re
import time
import json
import logging
import numpy as np
from typing import Dict, Tuple, Any
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer

# ---- 路由阈值，低于这个数就走简单路径 ----
BGE_MODEL_PATH = "./RAG_data/bge-legal-v1"
COMPLEX_SEMANTIC_THRESHOLD = 0.40
COMPLEX_LENGTH_THRESHOLD = 80

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
logger = logging.getLogger("UnifiedRouter")

# ---- 统一智能路由网关 (query.py 专用，与 soul.py 的缓存包装器配对) ----
class UnifiedQueryRouter_Query:
    def __init__(self):
        logger.info("正在把嵌入模型和分类器加载进显存...")
        self.embedder = SentenceTransformer(BGE_MODEL_PATH, device='cuda')
        self.classifier = LogisticRegression(class_weight='balanced')

        # L0 层：零成本正则 —— 闲聊、法条、简单句式直接拦截
        self.greetings = re.compile(
            r'^(你好|在吗|哈喽|hello|hi|嗨|你是谁|谢谢|拜拜|再见|早安|晚安)[！。？~\s]*$',
            re.IGNORECASE
        )
        self.exact_law_pattern = re.compile(r'《.*法》第\d+条')
        self.simple_qa_regex = re.compile(
            r"^(什么是|解释一下|告诉我|.*的概念|.*的定义是|.*出台时间是|.*?属于什么法)[\?？\s]*$"
        )

        # 用几条手工标注的样本热一下分类器，后面靠真实数据持续更新
        self._train_semantic_classifier()

    def _train_semantic_classifier(self):
        """给语义分类器喂几条种子样本，让它知道简单和复杂大概长什么样"""
        dummy_queries = [
            ("故意杀人罪判几年", 0),
            ("劳动法赔偿标准是什么", 0),
            ("法定退休年龄是多少", 0),
            ("老板欠薪跑路，我还没签合同，现在公司资产被转移了怎么维权", 1),
            ("我在公司干了三年，老板突然辞退我不给补偿金，手里只有工牌", 1),
        ]
        texts, labels = zip(*dummy_queries)
        vectors = self.embedder.encode(list(texts), normalize_embeddings=True)
        self.classifier.fit(vectors, labels)
        logger.info("分类器种子训练完了，可以用了。")

    # ==========================================================
    # L2 兜底：LLM 裁决（这里用模拟实现，生产环境替换为 API 调用）
    # ==========================================================
    def _llm_fallback_route(self, query: str) -> str:
        """
        当正则和轻量模型都无法给出高置信度判断时，
        调用 LLM 做最终意图裁决，只输出 JSON:
        {"intent": "CHITCHAT" | "SIMPLE_QA" | "COMPLEX_TASK"}
        """
        logger.debug("触发 LLM 兜底路由...")
        # 真实调用示例（请替换为你的 DeepSeek / OpenAI 客户端）：
        # system_prompt = (
        #     "你是一个法律意图分类器。将用户输入分为：\n"
        #     "1. CHITCHAT (闲聊)\n"
        #     "2. SIMPLE_QA (简单事实问答)\n"
        #     "3. COMPLEX_TASK (复杂案情推演)\n"
        #     "只输出JSON: {\"intent\": \"分类结果\"}"
        # )
        # response = client.chat.completions.create(
        #     model="deepseek-chat",
        #     messages=[{"role":"system","content":system_prompt},
        #               {"role":"user","content":query}],
        #     temperature=0.0,
        #     response_format={"type": "json_object"}
        # )
        # return json.loads(response.choices[0].message.content)["intent"]

        # 本地模拟逻辑，仅用于演示
        time.sleep(0.3)  # 模拟网络延迟
        if "赔" in query or "怎么办" in query or len(query) > 30:
            return "COMPLEX_TASK"
        return "SIMPLE_QA"

    # ==========================================================
    # 主路由函数：三层漏斗
    # ==========================================================
    def route(self, query: str) -> Dict[str, Any]:
        """
        返回统一的结构：
        {
            "intent": "CHITCHAT" / "SIMPLE_QA" / "COMPLEX_TASK",
            "source": "L0_Regex" / "L0_Heuristics" / "L1_Semantic" / "L2_LLM",
            "vector": np.array   (仅当需要进一步检索时提供),
            "response": str      (仅闲聊时直接返回回复)
        }
        """
        query = query.strip()
        logger.info(f"📨 流量接入: '{query}'")

        # ==================== L0 层：零成本正则 + 启发式 ====================
        # 1. 闲聊拦截
        if self.greetings.match(query):
            logger.info("⚡ [L0] 正则命中：闲聊")
            return {
                "intent": "CHITCHAT",
                "source": "L0_Regex",
                "response": "您好，我是法务智能体，请描述您的法律问题。"
            }

        # 2. 精确法条查询 -> 下发到简单 RAG
        if self.exact_law_pattern.search(query):
            logger.info("⚡ [L0] 正则命中：精确法条查询")
            return {
                "intent": "SIMPLE_QA",
                "source": "L0_Regex",
                "vector": self.embedder.encode(query, normalize_embeddings=True)
            }

        # 3. 标准简单问答句式
        if self.simple_qa_regex.match(query):
            logger.info("⚡ [L0] 正则命中：简单问答句式")
            return {
                "intent": "SIMPLE_QA",
                "source": "L0_Regex",
                "vector": self.embedder.encode(query, normalize_embeddings=True)
            }

        # 4. 长度启发式：超长文本直接判为复杂案情
        if len(query) >= COMPLEX_LENGTH_THRESHOLD:
            logger.info(f"⚡ [L0] 长度启发式({len(query)} chars) -> 复杂案情")
            return {
                "intent": "COMPLEX_TASK",
                "source": "L0_Heuristics",
                "vector": self.embedder.encode(query, normalize_embeddings=True)
            }

        # ==================== L1 层：轻量语义分类器 ====================
        query_vector = self.embedder.encode(query, normalize_embeddings=True)
        complex_prob = self.classifier.predict_proba(query_vector.reshape(1, -1))[0][1]
        logger.info(f"📊 [L1] 语义复杂概率: {complex_prob:.2%}")

        if complex_prob >= COMPLEX_SEMANTIC_THRESHOLD:
            logger.info("🚨 [L1] 高概率复杂问题 -> 进入复杂流程")
            return {
                "intent": "COMPLEX_TASK",
                "source": "L1_Semantic",
                "vector": query_vector
            }
        elif complex_prob < 0.10:   # 非常低的复杂概率，果断视为简单
            logger.info("✅ [L1] 低概率，按简单问题处理")
            return {
                "intent": "SIMPLE_QA",
                "source": "L1_Semantic",
                "vector": query_vector
            }

        # ==================== L2 层：LLM 最终裁决（灰色地带） ====================
        logger.info("🌐 [L2] 语义分类器不确定，转交 LLM 裁决...")
        llm_intent = self._llm_fallback_route(query)
        return {
            "intent": llm_intent,
            "source": "L2_LLM",
            "vector": query_vector
        }

    # ==========================================================
    # 兼容第一段代码的 process 接口（可选）
    # ==========================================================
    def process(self, query: str) -> Dict[str, Any]:
        result = self.route(query)
        # 如果需要唤醒 Agent 图引擎，在这里根据 intent 进一步处理
        if result["intent"] == "COMPLEX_TASK":
            logger.info("⏳ 复杂意图，启动 Plan-and-Replan 智能体引擎...")
            try:
                # 延迟导入，避免循环依赖
                import sys, os
                sys.path.insert(0, os.path.join(os.path.dirname(__file__), "multiple-search"))
                from soul import build_plan_replan_agent, Reasoner, AgenticNodesOperator
                from soul import LegalDenseGraphBuilder as SoulGraphBuilder

                # 构建 Agent（使用 soul.py 内的图引擎和操作器）
                graph_engine = SoulGraphBuilder(embedding_dim=768)
                agentic_ops = AgenticNodesOperator(
                    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
                    base_url="https://api.deepseek.com"
                )

                # 复用 router 自身的 embedder 作为检索编码器
                def embed_fn(text):
                    return self.embedder.encode(text, normalize_embeddings=True)

                reasoner = Reasoner(
                    legal_graph=graph_engine,
                    embedding_fn=embed_fn,
                    agentic_ops=agentic_ops
                )

                agent = build_plan_replan_agent(reasoner, agentic_ops)

                initial_state = {
                    "user_query": query,
                    "task_queue": [],
                    "past_observations": [],
                    "final_report": "",
                    "global_facts": [],
                    "retry_context": {},
                    "recursion_depth": 0
                }

                final_state = agent.invoke(initial_state)
                result["agent_report"] = final_state.get("final_report", "")
                result["agent_observations"] = final_state.get("past_observations", [])
                result["status"] = "agent_execution"

            except Exception as e:
                logger.error(f"Agent 引擎调用失败: {e}")
                result["status"] = "agent_error"
                result["agent_error"] = str(e)
        else:
            result["status"] = "simple_rag" if result["intent"] == "SIMPLE_QA" else "chitchat"
        return result


# ==========================================
# 测试
# ==========================================
if __name__ == "__main__":
    router = UnifiedQueryRouter_Query()

    test_queries = [
        "你好！",
        "你是谁开发的？",
        "什么是竞业限制？",
        "劳动法第十四条的规定是什么",
        "《劳动法》第39条",
        "律师你好，我2021年入职这家公司，前天老板口头通知我明天不用来了，但是一直没给我发解除劳动合同的书面通知，而且我的社保他们从上个月就断缴了，请问我能主张2N的赔偿金还是N+1？",
        "试用期被辞退能赔多少钱？",
        "我在公司干了三年，老板突然辞退我不给补偿金，我手里只有工牌，这合法吗？"
    ]

    print("\n" + "=" * 80)
    print(f"{'用户 Query':<55} {'意图':<15} {'来源':<15} {'耗时(ms)':<10}")
    print("=" * 80)

    for q in test_queries:
        start = time.perf_counter()
        res = router.route(q)
        elapsed = (time.perf_counter() - start) * 1000
        display_q = q[:52] + "..." if len(q) > 52 else q
        print(f"{display_q:<55} {res['intent']:<15} {res['source']:<15} {elapsed:6.2f}")