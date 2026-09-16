"""
确定性结构因果模型（SCM）执行器
==============================
因果评测的地基。三个作用：

1. **统一产出 L1/L2/L3 的标准答案**
   L1 = 观测推演，L2 = 对变量施加 $do(\\cdot)$ 后推演，L3 = 翻转已发生变量后推演。
   三者走**同一套结构方程**，所以标准答案天然自洽 —— 不会出现"L1 的答案和 L3 的答案
   互相矛盾"这种会把被测模型冤判成错的情况。

2. **让 $do(\\cdot)$ 有真正的语义，而不是改改题干**
   实现方式就是**图手术（graph surgery）**：被干预的变量固定取值、其结构方程
   不再参与计算。所以这不是"提示词层面的假装干预"，而是真的按 DAG 重算下游。

3. **让每个因果边可审计**
   每条边与每个方程都**强制**携带 `article_refs`（法条锚点）。没有锚点的边
   直接拒绝构建 —— 满足"最严合规"下的可溯源要求，也让评测题 0 层能自动
   校验"题目引用的法条是否存在且当时有效"。

反事实的实现说明（重要）：
  标准三步是 ① 溯因(abduction) ② 干预(action) ③ 预测(prediction)。
  本 SCM 是**确定性**模型，外生变量就是"噪声"本身，构造题目时已经给定，
  因此①是平凡的一步，③等价于用翻转后的取值重跑结构方程。
  故 `counterfactual()` 就实现为 `compute(assignments, do=flip)`。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

from .schemas import LawRef

# 变量角色
ROLE_TREATMENT = "treatment"      # 处理/因
ROLE_MEDIATOR = "mediator"        # 中介
ROLE_OUTCOME = "outcome"          # 果
ROLE_CONFOUNDER = "confounder"    # 混杂因子
ROLE_EXOGENOUS = "exogenous"      # 外生（由题目给定，无结构方程）

# 因果边的两类性质（建模时必须分清，否则会把"事实关联"冒充成"法律规则"）
KIND_LEGAL = "legal"        # 规范性因果：由法条规定（如"违法解除→二倍赔偿"），**必须带法条锚点**
KIND_FACTUAL = "factual"    # 事实性因果：案情内的事实关联（如"工作强度高→加班多"），无法律依据，只需说明
_KINDS = (KIND_LEGAL, KIND_FACTUAL)


class SCMValidationError(ValueError):
    """SCM 定义不合法（缺法条锚点、成环、方程缺失等）"""


@dataclass(frozen=True)
class Variable:
    """
    一个因果变量。

    :param role: **语义角色**（treatment/mediator/outcome/confounder/exogenous），
                 仅用于可读性与混杂因子分析。
    """

    name: str
    label: str
    role: str = ROLE_EXOGENOUS
    domain: Optional[Tuple[Any, ...]] = None   # 允许取值；None=连续/自由
    unit: str = ""
    description: str = ""


@dataclass(frozen=True)
class Edge:
    """
    因果边 src → dst。

    :param kind: KIND_LEGAL 的边**必须**带法条锚点；KIND_FACTUAL 的边只需给 rationale。
                 这个区分很重要：不是每条因果边都是法律规则，
                 把事实关联也硬挂一个法条会是伪溯源。
    """

    src: str
    dst: str
    rationale: str
    article_refs: Tuple[LawRef, ...] = ()
    kind: str = KIND_LEGAL


@dataclass(frozen=True)
class Equation:
    """
    结构方程：`target = fn(parent_values)`。

    :param formula: 人类可读公式，会出现在因果链里，供模型/裁判阅读
    :param kind:    同上。法律计算用 KIND_LEGAL（必须锚定法条）；
                    纯算术汇总（如"总额=各项之和"）用 KIND_FACTUAL。
    """

    target: str
    fn: Callable[[Mapping[str, Any]], Any]
    formula: str
    article_refs: Tuple[LawRef, ...] = ()
    describe: Optional[Callable[[Mapping[str, Any], Any], str]] = None
    kind: str = KIND_LEGAL


class StructuralCausalModel:
    """确定性 SCM：给定外生取值 → 唯一确定全部变量的值"""

    def __init__(self,
                 name: str,
                 title: str,
                 variables: Sequence[Variable],
                 edges: Sequence[Edge],
                 equations: Sequence[Equation],
                 outcome: str,
                 treatment: str = "",
                 outcome_renderer: Optional[Callable[[Mapping[str, Any]], str]] = None,
                 notes: str = ""):
        self.name = name
        self.title = title
        self.notes = notes
        self.variables = {v.name: v for v in variables}
        self.edges = list(edges)
        self.equations = {e.target: e for e in equations}
        self.outcome = outcome
        # 处理变量（可选）。声明后才能用 has_confounder() 判断该图是否真有混杂因子 ——
        # 这决定 L3 的「隔离混杂因子」一项是否适用（没有混杂因子就不该考它）。
        self.treatment = treatment
        self._outcome_renderer = outcome_renderer
        self._parents: Dict[str, List[str]] = {v: [] for v in self.variables}
        self._children: Dict[str, List[str]] = {v: [] for v in self.variables}
        for e in self.edges:
            self._parents.setdefault(e.dst, []).append(e.src)
            self._children.setdefault(e.src, []).append(e.dst)
        self._validate()

    # ================================================================ 校验
    def _validate(self) -> None:
        if not self.variables:
            raise SCMValidationError("SCM 必须至少声明一个变量")

        # 1) 边的端点必须是已声明变量
        for e in self.edges:
            for endpoint in (e.src, e.dst):
                if endpoint not in self.variables:
                    raise SCMValidationError(f"边 {e.src}→{e.dst} 引用了未声明的变量 {endpoint}")

        # 2) 规范性边必须有法条锚点；所有边都必须有依据说明
        for e in self.edges:
            if e.kind not in _KINDS:
                raise SCMValidationError(
                    f"边 {e.src}→{e.dst} 的 kind={e.kind!r} 非法，只能是 {_KINDS}"
                )
            if e.kind == KIND_LEGAL and not e.article_refs:
                raise SCMValidationError(
                    f"规范性因果边 {e.src}→{e.dst} 缺少法条锚点。"
                    f"若该关联并非法条规定，请显式标为 kind='factual'"
                )
            if not (e.rationale or "").strip():
                raise SCMValidationError(f"因果边 {e.src}→{e.dst} 缺少 rationale（依据说明）")

        # 3) 每个变量要么"有结构方程"（内生、模型算出来），
        #    要么是"根变量"（无方程、必须由题目给定取值）。
        #    注意：根变量的角色可以是 exogenous，也可以是 confounder ——
        #    混杂因子若由外部给定（不在模型内被决定），它就是根变量。
        for name, var in self.variables.items():
            has_eq = name in self.equations
            if var.role == ROLE_EXOGENOUS and has_eq:
                raise SCMValidationError(
                    f"变量 {name} 标为 exogenous（外生/由题目给定）却又有结构方程，语义矛盾"
                )
            if not has_eq and name == self.outcome:
                raise SCMValidationError(f"outcome={name} 必须有结构方程（结果不能由题目直接给定）")

        # 4) 方程名必须对应已声明变量 + 规范性方程必须有法条锚点
        for target, eq in self.equations.items():
            if target not in self.variables:
                raise SCMValidationError(f"方程目标 {target} 未声明为变量")
            if eq.kind not in _KINDS:
                raise SCMValidationError(f"方程 {target} 的 kind={eq.kind!r} 非法")
            if eq.kind == KIND_LEGAL and not eq.article_refs:
                raise SCMValidationError(
                    f"规范性方程 {target} 缺少法条锚点"
                    f"（若只是算术汇总，请显式标为 kind='factual'）"
                )
            if not (eq.formula or "").strip():
                raise SCMValidationError(f"方程 {target} 缺少 formula")

        # 5) 无环
        if len(self.topological_order()) != len(self.variables):
            raise SCMValidationError("SCM 的因果图存在环路")

        # 6) outcome 已声明且有结构方程
        if self.outcome not in self.variables:
            raise SCMValidationError(f"outcome={self.outcome} 未声明为变量")
        if self.outcome not in self.equations:
            raise SCMValidationError(f"outcome={self.outcome} 必须有结构方程")

        # 7) treatment（若声明）必须是已声明变量，且不等于 outcome
        if self.treatment:
            if self.treatment not in self.variables:
                raise SCMValidationError(f"treatment={self.treatment} 未声明为变量")
            if self.treatment == self.outcome:
                raise SCMValidationError("treatment 与 outcome 不能是同一个变量")

    # ================================================================ 图结构
    def parents(self, name: str) -> List[str]:
        return list(self._parents.get(name, []))

    def children(self, name: str) -> List[str]:
        return list(self._children.get(name, []))

    def ancestors(self, name: str) -> Set[str]:
        seen: Set[str] = set()
        stack = list(self.parents(name))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self.parents(cur))
        return seen

    def descendants(self, name: str) -> Set[str]:
        seen: Set[str] = set()
        stack = list(self.children(name))
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(self.children(cur))
        return seen

    def has_confounder(self) -> bool:
        """
        本 SCM 的 treatment→outcome 之间是否存在混杂因子。

        未声明 treatment 时保守返回 True（不做"不适用"假设，避免漏考）。
        """
        if not self.treatment:
            return True
        return bool(self.confounders(self.treatment, self.outcome))

    def confounders(self, treatment: str, outcome: str) -> Set[str]:
        """
        同时是 treatment 与 outcome 的祖先、且不是 outcome 的后代的变量 → 混杂因子。

        因果评测用它自动标注「本题是否存在混杂因子」—— 决定 L3 的
        `confounder_isolated` 这一项是否适用（没有混杂因子就不该考它）。
        """
        anc_t = self.ancestors(treatment)
        anc_y = self.ancestors(outcome)
        shared = anc_t & anc_y
        return {v for v in shared if v not in self.descendants(treatment)}

    def topological_order(self) -> List[str]:
        indeg = {v: len(self.parents(v)) for v in self.variables}
        queue = [v for v, d in indeg.items() if d == 0]
        order: List[str] = []
        while queue:
            cur = queue.pop(0)
            order.append(cur)
            for ch in self.children(cur):
                indeg[ch] -= 1
                if indeg[ch] == 0:
                    queue.append(ch)
        return order

    def root_names(self) -> List[str]:
        """根变量（无结构方程）—— 必须由题目给定取值"""
        return [v for v in self.variables if v not in self.equations]

    def article_refs(self) -> List[LawRef]:
        """本 SCM 用到的全部法条锚点（去重，保持出现顺序）"""
        out: List[LawRef] = []
        seen: Set[str] = set()
        for e in self.edges:
            for ref in e.article_refs:
                if ref.key not in seen:
                    seen.add(ref.key)
                    out.append(ref)
        for eq in self.equations.values():
            for ref in eq.article_refs:
                if ref.key not in seen:
                    seen.add(ref.key)
                    out.append(ref)
        return out

    def edge_refs(self, src: str, dst: str) -> Tuple[LawRef, ...]:
        for e in self.edges:
            if e.src == src and e.dst == dst:
                return e.article_refs
        return ()

    # ================================================================ 计算
    def validate_assignments(self, assignments: Mapping[str, Any]) -> None:
        """校验外生/干预取值在声明域内"""
        for k, v in assignments.items():
            if k not in self.variables:
                raise SCMValidationError(f"取值含未声明变量 {k}")
            dom = self.variables[k].domain
            if dom is not None and v not in dom:
                raise SCMValidationError(
                    f"变量 {k} 取值 {v!r} 不在允许域 {dom} 内"
                )
        missing = [v for v in self.root_names() if v not in assignments]
        if missing:
            raise SCMValidationError(f"缺少根变量取值：{missing}")

    def compute(self, assignments: Mapping[str, Any],
                do: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """
        观测或干预下求全部变量取值。

        `do` 非空时执行**图手术**：被干预变量固定为其取值，其结构方程被跳过，
        下游按剩余方程重算 —— 这就是 $P(Y \\mid do(X=x))$ 的确定性版本。
        """
        merged = dict(assignments)
        if do:
            merged.update(do)
        self.validate_assignments({k: v for k, v in merged.items()
                                   if k in self.variables})
        # 干预值也可能不在域内，单独再校验一次
        for k, v in (do or {}).items():
            dom = self.variables[k].domain
            if dom is not None and v not in dom:
                raise SCMValidationError(f"do({k}={v!r}) 取值不在允许域 {dom} 内")

        values: Dict[str, Any] = dict(merged)
        for name in self.topological_order():
            if name in values:
                continue                      # 已被外生给定 或 被 do() 固定
            eq = self.equations[name]
            values[name] = eq.fn(values)
        return values

    def intervene(self, assignments: Mapping[str, Any],
                  do: Mapping[str, Any]) -> Dict[str, Any]:
        """L2：干预题"""
        return self.compute(assignments, do=do)

    def counterfactual(self, assignments: Mapping[str, Any],
                       flip: Mapping[str, Any]) -> Dict[str, Any]:
        """
        L3：反事实题。把已发生的变量翻转，重跑结构方程得到反事实结果。
        （确定性 SCM 下溯因平凡，详见模块头说明）
        """
        return self.compute(assignments, do=flip)

    # ================================================================ 因果链
    def chain(self, values: Mapping[str, Any]) -> List[str]:
        """
        把一次求值结果渲染成**有序因果推理链**（拓扑序），每步带上法条锚点。

        这条链就是 `LegalCausalTestCase.key_reasoning_points` 的来源 ——
        因为是机器按同一套方程生成的，L1/L2/L3 三层天然同源。
        """
        items: List[str] = []
        for name in self.topological_order():
            var = self.variables[name]
            val = values.get(name)
            eq = self.equations.get(name)
            if eq is None:                    # 外生变量
                items.append(f"{var.label}={self.fmt(val, var)}")
                continue
            if eq.describe is not None:
                text = eq.describe(values, val)
            else:
                text = f"{var.label}={self.fmt(val, var)}（{eq.formula}）"
            refs = "、".join(f"《{r.law_name}》{r.article_no}" for r in eq.article_refs)
            items.append(f"{text}；依据：{refs}" if refs else text)
        return items

    @staticmethod
    def fmt(value: Any, var: Variable) -> str:
        """变量取值的人类可读格式化（链文本、题干都要用）"""
        if isinstance(value, float):
            text = f"{value:g}"
        else:
            text = str(value)
        return f"{text}{var.unit}" if var.unit else text

    def render_outcome(self, values: Mapping[str, Any]) -> str:
        """渲染成自然语言结论（该题的标准答案文本）"""
        if self._outcome_renderer is None:
            var = self.variables[self.outcome]
            return f"{var.label}为 {self.fmt(values.get(self.outcome), var)}"
        return self._outcome_renderer(values)

    # ================================================================ 差异
    @staticmethod
    def diff(before: Mapping[str, Any],
             after: Mapping[str, Any]) -> Dict[str, Tuple[Any, Any]]:
        """两次求值中发生变化的变量 → {变量: (原值, 新值)}。

        用于 L3：反事实题里"到底哪些变量变了"，也是抗扰动题判定的基础。
        """
        changed: Dict[str, Tuple[Any, Any]] = {}
        for k in set(before) | set(after):
            b, a = before.get(k), after.get(k)
            if isinstance(b, float) and isinstance(a, float) and abs(b - a) < 1e-9:
                continue
            if b != a:
                changed[k] = (b, a)
        return changed

    def describe_graph(self) -> str:
        """文字版 DAG，便于人工审计"""
        lines = [f"{self.title} ({self.name})",
                 f"  结果变量: {self.outcome}"]
        for v in self.variables.values():
            lines.append(f"  · [{v.role}] {v.name} = {v.label}"
                         + (f"  域={v.domain}" if v.domain else ""))
        for e in self.edges:
            refs = "、".join(f"《{r.law_name}》{r.article_no}" for r in e.article_refs)
            tag = "法" if e.kind == KIND_LEGAL else "事"
            lines.append(f"  {e.src} → {e.dst}   [{tag}:{e.rationale}]"
                         + (f"  ← {refs}" if refs else ""))
        for t, eq in self.equations.items():
            refs = "、".join(f"《{r.law_name}》{r.article_no}" for r in eq.article_refs)
            lines.append(f"  {t} := {eq.formula}   ← {refs}")
        return "\n".join(lines)
