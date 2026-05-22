"""
双层蓝图 Schema —— Meta-Planner 输出的"施工图纸"
============================================
S_q 是抽象推理骨架（DAG），描述做什么但不提具体实体。
C_q 把抽象的节点映射到当前案件的具体查询。
附带 DAG 环检测、拓扑排序、并行组识别，最后展平为 Executor 能吃的队列。

技术栈: Pydantic v2 (BaseModel/Field/validator) / 图算法 (拓扑排序)
"""
from typing import List, Dict, Optional, Set
from pydantic import BaseModel, Field, validator


class DAGNode(BaseModel):
    """S_q 里的一步：要做什么事（抽象版），以及得等哪些前置步骤完成"""
    id: str = Field(..., description="节点编号")
    abstract: str = Field(..., description="抽象任务，不提具体人名公司名")
    deps: List[str] = Field(default_factory=list, description="依赖的前置节点 id")


class SkeletonGraph(BaseModel):
    """S_q：抽象推理骨架 DAG"""
    nodes: List[DAGNode] = Field(..., description="DAG 中的所有任务节点")

    @validator("nodes")
    def validate_dag(cls, nodes: List[DAGNode]) -> List[DAGNode]:
        """验证无环 + 依赖节点存在"""
        node_ids = {n.id for n in nodes}
        for node in nodes:
            for dep in node.deps:
                if dep not in node_ids:
                    raise ValueError(f"节点 '{node.id}' 依赖不存在的节点 '{dep}'")
        # 简单环检测：拓扑排序
        in_degree = {n.id: len(n.deps) for n in nodes}
        zero_in = [nid for nid, d in in_degree.items() if d == 0]
        visited = set()
        while zero_in:
            nid = zero_in.pop()
            visited.add(nid)
            for node in nodes:
                if nid in node.deps:
                    in_degree[node.id] -= 1
                    if in_degree[node.id] == 0:
                        zero_in.append(node.id)
        if len(visited) != len(nodes):
            raise ValueError("DAG 中存在环路！")
        return nodes

    def topological_order(self) -> List[str]:
        """返回拓扑排序后的节点 ID 列表"""
        in_degree = {n.id: len(n.deps) for n in self.nodes}
        zero_in = sorted([nid for nid, d in in_degree.items() if d == 0])
        order = []
        while zero_in:
            nid = zero_in.pop(0)
            order.append(nid)
            for node in self.nodes:
                if nid in node.deps:
                    in_degree[node.id] -= 1
                    if in_degree[node.id] == 0:
                        zero_in.append(node.id)
        return order

    def get_parallel_groups(self) -> List[List[str]]:
        """返回可并行执行的节点组（每组的节点互不依赖）"""
        remaining = set(n.id for n in self.nodes)
        groups = []
        while remaining:
            group = [
                nid for nid in remaining
                if all(dep not in remaining for dep in self._deps_of(nid))
            ]
            if not group:
                break
            groups.append(sorted(group))
            remaining -= set(group)
        return groups

    def _deps_of(self, node_id: str) -> Set[str]:
        for n in self.nodes:
            if n.id == node_id:
                return set(n.deps)
        return set()


class Concretization(BaseModel):
    """C_q：将抽象任务实例化为具体的检索查询"""
    concretions: Dict[str, str] = Field(
        ...,
        description="节点ID → 具体查询文本的映射"
    )


class DoubleLayerPlan(BaseModel):
    """双层蓝图 P_q = {S_q, C_q}"""
    skeleton: SkeletonGraph = Field(..., description="S_q: 抽象推理骨架 DAG")
    concretion: Concretization = Field(..., description="C_q: 抽象→具体映射")

    def to_flat_task_queue(self, respect_deps: bool = True) -> List[Dict]:
        """
        将双层蓝图展平为可执行的任务队列。
        respect_deps=True: 按拓扑序排列
        respect_deps=False: 按节点 ID 顺序排列
        """
        if respect_deps:
            order = self.skeleton.topological_order()
        else:
            order = [n.id for n in self.skeleton.nodes]

        queue = []
        for nid in order:
            if nid in self.concretion.concretions:
                queue.append({
                    "task_desc": self.concretion.concretions[nid],
                    "engine": "GRAPH_TRAVERSAL",
                    "rationale": f"双层蓝图节点 {nid}: {self._abstract_of(nid)}"
                })
        return queue

    def _abstract_of(self, node_id: str) -> str:
        for n in self.skeleton.nodes:
            if n.id == node_id:
                return n.abstract
        return ""


# ============================================================================
# 辅助：从 LLM JSON 输出构造 DoubleLayerPlan
# ============================================================================
def parse_double_layer_plan(raw_json: dict) -> DoubleLayerPlan:
    """
    解析 LLM 返回的 JSON 为双层蓝图。
    兼容多种 LLM 输出格式。
    """
    # 格式1: {"skeleton": {"nodes": [...]}, "concretion": {"concretions": {...}}}
    if "skeleton" in raw_json and "concretion" in raw_json:
        return DoubleLayerPlan(**raw_json)

    # 格式2: 旧的扁平格式 {"task_queue": ["步骤1", ...]}
    if "task_queue" in raw_json:
        tasks = raw_json["task_queue"]
        nodes = []
        concretions = {}
        for i, t in enumerate(tasks):
            nid = str(i + 1)
            if isinstance(t, dict):
                desc = t.get("task_desc", str(t))
            else:
                desc = str(t)
            nodes.append(DAGNode(id=nid, abstract=desc[:30], deps=[]))
            concretions[nid] = desc
        return DoubleLayerPlan(
            skeleton=SkeletonGraph(nodes=nodes),
            concretion=Concretization(concretions=concretions)
        )

    # 格式3: 空
    raise ValueError(f"无法解析双层蓝图: {list(raw_json.keys())}")
