"""Offline workflow example."""
import sys
import logging
import numpy as np
from .operators import AgenticNodesOperator
from .reasoning import Reasoner
from .workflow import build_plan_replan_agent
from dataset.graph import LegalDenseGraphBuilder
logger=logging.getLogger(__name__)
def main():
    # Windows 控制台默认 GBK，而观察记录里含 emoji（如 ⚠️/🐳），
    # 直接 print 会抛 UnicodeEncodeError。这里把 stdout 切到 UTF-8。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # 1. 模拟图谱（实际应用请加载真实数据）
    # 使用 from_config() 保证维度与 config.yaml 的 embedding.dimension 一致
    graph_engine = LegalDenseGraphBuilder.from_config(load_memory=False)
    from dataset.graph import StubEncoder
    graph_engine.set_encoder(StubEncoder(graph_engine.dim))
    dummy_nodes = [
        {"id": 1, "content": "试用期不符合录用条件可解除合同。", "type": "Raw", "metadata": {"source": "劳动法"}},
        {"id": 2, "content": "违法解除劳动合同按经济补偿标准的二倍支付赔偿金。", "type": "Raw", "metadata": {"source": "劳动法"}},
    ]
    dummy_emb = np.random.randn(len(dummy_nodes), graph_engine.dim).astype(np.float32)
    dummy_emb = dummy_emb / np.linalg.norm(dummy_emb, axis=1, keepdims=True)
    graph_engine.build_initial_graph_batch(dummy_nodes, dummy_emb)

    # 2. 模拟 Embedding 函数
    def mock_embedding(text: str) -> np.ndarray:
        v = np.random.randn(graph_engine.dim).astype(np.float32)
        return v / np.linalg.norm(v)

    # 3. 模拟 AgenticNodesOperator（包含新增方法）
    class MockAgenticOps(AgenticNodesOperator):
        """阶段 3：改为覆写 _call_messages（Prompt 已统一走 prompts.py 构造器）"""

        def _call_messages(self, messages, require_json=False, temperature=0.1):
            system_prompt = messages[0]["content"] if messages else ""
            logger.info(f"Mock LLM called with: {system_prompt[:50]}...")
            if require_json:
                if "Causal Extraction:" in system_prompt:
                    return '{"propositions": [], "causal_relations": []}'
                if "事实调查官" in system_prompt:
                    return '{"rationale": "Mock充足", "status": "sufficient", "extracted_facts": ["Mock事实1"]}'
                if "重规划引擎" in system_prompt:
                    return ('{"task_queue": [{"task_desc": "Mock全局检索", '
                            '"engine": "GLOBAL_DENSE_WORMHOLE", "rationale": "Mock"}]}')
                # Meta-Planner：返回双层蓝图
                return ('{"skeleton": {"nodes": [{"id": "1", "abstract": "核实劳动关系", "deps": []}, '
                        '{"id": "2", "abstract": "核实解除合法性", "deps": ["1"]}, '
                        '{"id": "3", "abstract": "计算赔偿金额", "deps": ["2"]}]}, '
                        '"concretion": {"concretions": {"1": "核实劳动关系", '
                        '"2": "核实辞退是否合法", "3": "计算赔偿金额"}}}')
            if "Extractor" in system_prompt:
                return "Mock 提取事实：公司口头辞退，属于违法解除。"
            if "Reasoner" in system_prompt:
                return "Mock 推理：根据事实，应支付双倍赔偿金。"
            if "Generator" in system_prompt:
                return "Mock 最终报告：您可以获得2N赔偿金。"
            return "Mock 回答"

    mock_ops = MockAgenticOps(api_key="mock")

    # 4. 创建 Reasoner
    reasoner = Reasoner(legal_graph=graph_engine, embedding_fn=mock_embedding, agentic_ops=mock_ops)

    # 5. 构建并运行智能体
    from agent_tools.sandbox.executor import run_in_process
    agent = build_plan_replan_agent(reasoner, mock_ops, code_runner=lambda code, **kwargs: run_in_process(code))

    initial_state = {
        "user_query": "试用期最后一天被辞退，能拿多少赔偿？",
        "task_queue": [],
        "past_observations": [],
        "final_report": "",
        "global_facts": [],
        "retry_context": {},
        "recursion_depth": 0
    }

    final_state = agent.invoke(initial_state)

    print("\n" + "="*50)
    print("最终法律意见书：")
    print(final_state.get("final_report", "无报告生成"))
    print("\n全部观察记录：")
    for obs in final_state["past_observations"]:
        print(obs)
    print("\n全局事实：")
    for f in final_state.get("global_facts", []):
        print(f"  - {f}")

if __name__ == "__main__":
    main()
