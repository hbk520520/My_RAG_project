import json
import time
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict

import openai  # DeepSeek API 兼容 OpenAI 接口

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LegalBenchmark")

# ============================================================================
# 0. 配置 DeepSeek 裁判模型
# ============================================================================
DEEPSEEK_API_KEY = "your-deepseek-api-key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
JUDGE_MODEL = "deepseek-chat"      # 可使用 deepseek-reasoner 获得更强逻辑，但成本更高
judge_client = openai.OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def llm_judge(prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
    """调用 DeepSeek 裁判模型，返回文本结果"""
    try:
        resp = judge_client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Judge model call failed: {e}")
        return "ERROR"


# ============================================================================
# 1. 数据结构定义
# ============================================================================
@dataclass
class TestSample:
    """一条测试样本"""
    query: str                         # 用户问题（可能含噪声/口语化）
    ground_truth_answer: str           # 标准答案（由 Evol-Instruct + 法官模型交叉验证生成）
    relevant_doc_ids: List[int]        # 应该被检索到的文档节点 ID（用于 HR/MRR）
    noise_doc_ids: List[int]           # 不应被检索到但可能被误召回的文档 ID
    required_hops: int = 1             # 回答问题最少需要的推理跳数（用于图NIAH）
    optional_trajectory: List[str] = None  # 可选的最优执行轨迹描述（用于轨迹评测）


@dataclass
class RetrievalResult:
    """检索器返回的结构"""
    top_k_ids: List[int]               # 系统返回的 Top-K 文档 ID 列表（有序）
    top_k_scores: List[float]          # 对应相似度分数


@dataclass
class TrajectoryLog:
    """智能体执行轨迹（从 LangGraph state history 提取）"""
    plan: List[str]                    # 初始规划（任务队列）
    executed_steps: List[Dict]         # 每一步的详细信息 {step, node, action, result_summary}
    retry_count: int                   # 总重试/回退次数
    final_answer: str                  # 最终输出


# ============================================================================
# 2. 评测核心类
# ============================================================================
class LegalBenchmark:
    def __init__(self,
                 retriever_fn: Callable[[str, int], RetrievalResult],
                 agent_runner_fn: Callable[[str], TrajectoryLog],
                 k_values: List[int] = None):
        """
        :param retriever_fn: 检索函数，输入 query 和 k，返回 RetrievalResult
        :param agent_runner_fn: 运行完整的智能体流程，返回 TrajectoryLog
        :param k_values: 用于计算 HR@K 的 K 值列表，默认 [1,3,5,10]
        """
        self.retrieve = retriever_fn
        self.run_agent = agent_runner_fn
        self.k_values = k_values or [1, 3, 5, 10]

    # ------------------------------------------------------------------
    # 2.1 检索质量评测 (HR@K, MRR, Noise Ratio)
    # ------------------------------------------------------------------
    def evaluate_retrieval(self, samples: List[TestSample]) -> Dict[str, float]:
        """返回 HR@K, MRR, Noise Ratio"""
        metrics = defaultdict(list)

        for sample in samples:
            for k in self.k_values:
                result = self.retrieve(sample.query, k)
                retrieved_ids = result.top_k_ids

                # Hit Rate @K
                hit = any(rid in sample.relevant_doc_ids for rid in retrieved_ids[:k])
                metrics[f"HR@{k}"].append(1.0 if hit else 0.0)

                # MRR (只计算第一个相关文档的倒数排名)
                for rank, rid in enumerate(retrieved_ids[:k], start=1):
                    if rid in sample.relevant_doc_ids:
                        metrics["MRR"].append(1.0 / rank)
                        break
                else:
                    metrics["MRR"].append(0.0)

            # Noise Ratio：检索结果中噪声文档的比例 (取最大K)
            max_k = max(self.k_values)
            result = self.retrieve(sample.query, max_k)
            noise_count = sum(1 for rid in result.top_k_ids if rid in sample.noise_doc_ids)
            metrics["NoiseRatio"].append(noise_count / max_k if max_k > 0 else 0.0)

        # 平均
        avg_metrics = {k: float(np.mean(v)) for k, v in metrics.items()}
        return avg_metrics

    # ------------------------------------------------------------------
    # 2.2 轨迹评测 (使用 DeepSeek 裁判)
    # ------------------------------------------------------------------
    def evaluate_trajectory(self, samples: List[TestSample]) -> Dict[str, float]:
        """对每个样本的智能体执行轨迹进行评分，返回平均轨迹得分"""
        traj_scores = []
        for sample in samples:
            traj_log = self.run_agent(sample.query)
            score = self._score_trajectory(sample, traj_log)
            traj_scores.append(score)
        return {"TrajectoryScore": float(np.mean(traj_scores))}

    def _score_trajectory(self, sample: TestSample, traj: TrajectoryLog) -> float:
        """使用 DeepSeek 裁判对一条轨迹打分 (0-1)"""
        # 构造裁判 prompt
        plan_str = "\n".join([f"- {p}" for p in traj.plan])
        steps_summary = "\n".join(
            [f"Step {s['step']} ({s['node']}): {s['action']} → {s['result_summary'][:100]}"
             for s in traj.executed_steps]
        )
        prompt = f"""你是一个法律智能体轨迹裁判。请根据以下信息对智能体的“思考过程”进行评分（0到1之间，保留两位小数）。

用户问题：{sample.query}
标准答案：{sample.ground_truth_answer}
理想执行规划（参考）：{sample.optional_trajectory if sample.optional_trajectory else "无"}
实际初始规划：
{plan_str}
实际执行步骤：
{steps_summary}
总重试/回退次数：{traj.retry_count}
最终输出：{traj.final_answer}

评分要求（综合考量）：
1. 初始规划是否合理且高效（无需多余步骤）？ (权重 0.3)
2. 执行过程是否直接？重试/无效跳转是否过多？ (权重 0.3)
3. 最终答案是否正确且完整？ (权重 0.4)

请只输出一个数字（例如 0.85），不要包含其他文字。"""
        resp = llm_judge(prompt, temperature=0.0, max_tokens=10)
        try:
            return float(resp)
        except:
            return 0.0

    # ------------------------------------------------------------------
    # 2.3 图结构大海捞针 (Graph-NIAH)
    # ------------------------------------------------------------------
    def evaluate_graph_niah(self, niah_samples: List[TestSample]) -> Dict[str, float]:
        """
        niah_samples: 需要多跳推理的特殊样本，其 relevant_doc_ids 由跳板节点序列构成。
        测试检索器能否在 K 个结果内覆盖所有跳板节点。
        返回：多跳召回完整度、跳板丢失率等。
        """
        total_hops = 0
        recovered_hops = 0
        perfect_paths = 0
        for sample in niah_samples:
            # 假设 relevant_doc_ids 按跳顺序给出： [node_A, node_B, node_C]
            required_hops = len(sample.relevant_doc_ids)
            total_hops += required_hops
            # 检索尽可能多的节点（例如 20），模拟 agent 的检索深度
            result = self.retrieve(sample.query, max(20, required_hops * 5))
            retrieved_set = set(result.top_k_ids)
            # 统计找回的跳板节点
            found_hops = sum(1 for nid in sample.relevant_doc_ids if nid in retrieved_set)
            recovered_hops += found_hops
            if found_hops == required_hops:
                perfect_paths += 1

        n = len(niah_samples)
        metrics = {
            "GraphNIAH_Recall": recovered_hops / total_hops if total_hops > 0 else 0.0,
            "GraphNIAH_PerfectPathRate": perfect_paths / n if n > 0 else 0.0
        }
        return metrics

    # ------------------------------------------------------------------
    # 2.4 综合评测入口
    # ------------------------------------------------------------------
    def run_full_benchmark(self,
                           test_samples: List[TestSample],
                           niah_samples: List[TestSample] = None) -> Dict[str, float]:
        logger.info("Starting benchmark ...")
        results = {}

        # 1. 检索质量
        ret_metrics = self.evaluate_retrieval(test_samples)
        results.update(ret_metrics)
        logger.info(f"Retrieval metrics: {ret_metrics}")

        # 2. 轨迹评测（成本较高，可以只对部分样本执行）
        # 实际中可随机抽样子集，这里全量运行
        traj_score = self.evaluate_trajectory(test_samples)
        results.update(traj_score)
        logger.info(f"Trajectory score: {traj_score}")

        # 3. Graph-NIAH (如果提供了专门样本)
        if niah_samples:
            niah_metrics = self.evaluate_graph_niah(niah_samples)
            results.update(niah_metrics)
            logger.info(f"Graph-NIAH metrics: {niah_metrics}")

        return results


# ============================================================================
# 3. 示例：如何接入你的系统（模拟接口）
# ============================================================================
# 你需要实现以下两个函数，对接你自己的图谱/agent

def mock_retrieve(query: str, k: int) -> RetrievalResult:
    """模拟检索器"""
    # 实际应调用 legal_graph.index.search(embedding(query), k)
    # 返回节点 ID
    return RetrievalResult(
        top_k_ids=[101, 102, 103, 104, 105][:k],
        top_k_scores=[0.9, 0.8, 0.7, 0.6, 0.5][:k]
    )


def mock_run_agent(query: str) -> TrajectoryLog:
    """模拟运行你的 LangGraph 智能体，并返回轨迹"""
    # 实际应从你的 agent.invoke(state) 后的 state history 中提取信息
    return TrajectoryLog(
        plan=["检查劳动关系", "核实辞退理由", "计算赔偿金"],
        executed_steps=[
            {"step": 1, "node": "Executor", "action": "检索法律条文", "result_summary": "找到劳动合同法第47条"},
            {"step": 2, "node": "Executor", "action": "检索辞退事实", "result_summary": "确认口头辞退无书面通知"},
            {"step": 3, "node": "Executor", "action": "计算赔偿", "result_summary": "赔偿金=2*月工资*工作年限"}
        ],
        retry_count=1,
        final_answer="根据《劳动合同法》第47条、第87条，您应获得违法解除赔偿金共计xxx元。"
    )


# ============================================================================
# 4. 生成测试样本（基于 Evol-Instruct 思想）
# ============================================================================
def generate_test_samples() -> List[TestSample]:
    """
    实际中应由 Evol-Instruct 数据工厂生成，这里手工构造几条示例。
    每条样本都包含查询、标准答案、相关/噪声文档ID。
    """
    samples = []
    # 样本1：简单事实查询
    samples.append(TestSample(
        query="试用期最后一天被辞退，能拿多少钱？",
        ground_truth_answer="如果公司不能证明员工不符合录用条件，属于违法解除，应支付2N赔偿金。",
        relevant_doc_ids=[101, 102],   # 假设文档101是“试用期规定”，102是“违法解除赔偿”
        noise_doc_ids=[201, 202],      # 无关文档
        required_hops=2,
        optional_trajectory=["确认劳动关系", "核实解除理由", "计算赔偿数额"]
    ))
    # 样本2：含噪声口语化查询
    samples.append(TestSample(
        query="俺被老板口头炒了，没签过合同，他该给俺多少钱啊？",
        ground_truth_answer="未签订劳动合同，可主张双倍工资差额；口头辞退若无合法理由属于违法解除，需支付2N赔偿。",
        relevant_doc_ids=[103, 104, 105],   # 双倍工资、违法解除等
        noise_doc_ids=[203],
        required_hops=3,
        optional_trajectory=["确认劳动关系及合同状况", "查核未签合同的法律后果", "查核口头辞退的合法性", "合并计算赔偿数额"]
    ))
    return samples


def generate_niah_samples() -> List[TestSample]:
    """专门的多跳 Graph-NIAH 样本，每个样本的 relevant_doc_ids 是一条证据链"""
    # 假设在图数据库中存在节点 301 -> 302 -> 303 的路径
    niah_samples = [
        TestSample(
            query="某员工在集团公司A的子公司B工作满一年后，被调到子公司C，现在被C辞退，工龄如何连续计算？",
            ground_truth_answer="劳动关系连续，工龄从入职集团公司A起算，关联公司间调动工龄合并计算。",
            relevant_doc_ids=[301, 302, 303],  # 节点301: 劳动合同法实施条例第10条, 302: 关联公司认定, 303: 工龄连续计算案例
            noise_doc_ids=[],
            required_hops=3
        )
    ]
    return niah_samples


# ============================================================================
# 5. 执行评测
# ============================================================================
if __name__ == "__main__":
    # 实例化评测器，传入你的真实函数
    benchmark = LegalBenchmark(
        retriever_fn=mock_retrieve,
        agent_runner_fn=mock_run_agent,
        k_values=[1, 3, 5]
    )

    test_set = generate_test_samples()
    niah_set = generate_niah_samples()

    final_metrics = benchmark.run_full_benchmark(test_set, niah_set)

    print("\n" + "="*50)
    print("Legal Knowledge Graph Agent Benchmark Results")
    print("="*50)
    for k, v in final_metrics.items():
        print(f"{k:30s}: {v:.4f}")
    print("="*50)
    print("Note: TrajectoryScore is based on DeepSeek judge (0-1).")