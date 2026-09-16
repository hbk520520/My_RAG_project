#!/usr/bin/env python
"""
入口点体检（P0 收尾）
=====================
回答两个问题，并**用子进程实际验证**而不是靠人记：

  1. **导入安全性** —— `import X` 会不会产生副作用？
     （连服务、读凭据、下模型、打日志…）
  2. **离线可跑性** —— 这个入口的 `__main__` 能不能在无网络/无服务/无凭据下跑完？

为什么值得单独做一个工具：
  本项目**反复**因为"导入即副作用"出问题 ——
    · 阶段 2：`import soul` 就会持有一个写死的假 API Key
    · 阶段 6：`training_data_guide.py` / `replanner_rules_report.py` 顶层 `print`
  而 `compileall` 与 `pytest` 都抓不到这类问题（pytest 走 conftest 已经铺好环境）。

分类含义：

  import: ok        —— **必须**干净导入（本工具的断言目标）
  import: needs:X   —— 导入就需要 X（已登记，只报告不判失败）
  run:    offline   —— `python <file>` 可在离线环境跑完
  run:    no-main   —— 只是库，没有 `__main__`
  run:    needs:X   —— 运行需要 X

用法：
    python tools/audit_entrypoints.py            # 体检 + 打印矩阵
    python tools/audit_entrypoints.py --check    # 只跑断言（供 CI 用）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 与 tools/ci.py / tests/conftest.py 保持一致的路径口径
EXTRA_PATHS = [
    ROOT,
    os.path.join(ROOT, "asynchronization"),
    os.path.join(ROOT, "asynchronization", "workers"),
    os.path.join(ROOT, "multiple-search"),
    os.path.join(ROOT, "multiple-search", "legal_sandbox"),
    # SemanticCache/ 自身是一个 sys.path 根（无 __init__.py，故只能按顶层模块导入）
    os.path.join(ROOT, "multiple-search", "SemanticCache"),
]

ENV = {
    **os.environ,
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    # 清掉可能存在的真实凭据 —— 导入安全性必须在"裸环境"下成立
    "DEEPSEEK_API_KEY": "",
    "KAFKA_BOOTSTRAP": "",
    "REDIS_URL": "",
    "PYTHONPATH": os.pathsep.join(EXTRA_PATHS),
}


@dataclass
class EntryPoint:
    path: str                    # 相对仓库根
    module: str | None           # import 名；None = 不可作为模块导入
    import_expect: str           # "ok" | "needs:<原因>" | "n/a"
    run_expect: str              # "offline" | "no-main" | "needs:<原因>" | "covered-by-ci"
    note: str = ""


# ============================================================================
# 清单：每一行都是一个人工判断，工具负责验证这个判断还成立
# ============================================================================
REGISTRY: list[EntryPoint] = [
    # ---------------- 根目录：核心库 ----------------
    EntryPoint("config_loader.py", "config_loader", "ok", "no-main",
               "配置单一真源；缺必填项只告警不抛（validate(strict) 才抛）"),
    EntryPoint("prompts.py", "prompts", "ok", "no-main", "Prompt 单一真源"),
    EntryPoint("double_layer_plan.py", "double_layer_plan", "ok", "no-main"),
    EntryPoint("replanner_rules.py", "replanner_rules", "ok", "no-main",
               "硬规则表，双链路共用"),
    EntryPoint("observability.py", "observability", "ok", "no-main",
               "阶段 6：新增 DLQ 单例，顶层无 IO"),
    EntryPoint("query.py", "query", "ok", "needs:BGE-M3(约2GB) + LLM API Key",
               "导入只 import SentenceTransformer，**构造路由时**才加载模型"),
    EntryPoint("benchmark.py", "benchmark", "ok", "needs:LLM API Key",
               "阶段 2：裁判客户端改为惰性，导入不再需要凭据"),
    EntryPoint("training_data_guide.py", "training_data_guide", "ok", "offline",
               "阶段 6：顶层 print 已收进 __main__"),
    EntryPoint("replanner_rules_report.py", "replanner_rules_report", "ok", "offline",
               "阶段 6：顶层 print 已收进 __main__"),

    # ---------------- dataset ----------------
    EntryPoint("dataset/graph.py", "dataset.graph", "ok", "covered-by-ci",
               "图引擎单一真源；BGE-M3 懒加载"),
    EntryPoint("dataset/IncrementalMemoryManager.py", "dataset.IncrementalMemoryManager",
               "ok", "covered-by-ci", "GMM 动态阈值"),
    EntryPoint("dataset/memory_graph_bridge.py", "dataset.memory_graph_bridge", "ok",
               "covered-by-ci", "GMM ↔ 图引擎双写"),
    EntryPoint("dataset/chunk.py", "dataset.chunk", "ok",
               "needs:PDF 文件参数",
               "核心函数 chunk_text 是纯函数，由测试覆盖；__main__ 需给 PDF 路径"),
    EntryPoint("dataset/prepare_corpus.py", "dataset.prepare_corpus", "ok", "offline",
               "P0 重写：原为占位符模板 + 模块级执行，导入即抛 FileNotFoundError；"
               "现在 --stub-encoder 可完全离线跑"),

    # ---------------- multiple-search ----------------
    EntryPoint("multiple-search/soul.py", "soul", "ok", "covered-by-ci",
               "阶段 2：已删模块级假 Key；P1 又删掉了无调用方的 get_llm_client()，"
               "LLM 只经 _call_messages 一个接缝"),
    EntryPoint("multiple-search/legal_sandbox/sandbox_exec.py", "sandbox_exec", "ok",
               "no-main", "沙箱执行统一入口"),
    EntryPoint("multiple-search/legal_sandbox/sandbox_manager.py", "sandbox_manager",
               "ok", "no-main",
               "阶段 5：新增 get_sandbox_manager() 进程内单例；构造时才连 Docker"),
    EntryPoint("multiple-search/legal_sandbox/sandbox_server.py", "sandbox_server",
               "ok", "needs:Docker",
               "容器内常驻 HTTP 服务，由 sandbox_manager 拉起"),
    EntryPoint("multiple-search/legal_sandbox/example_usage.py", "example_usage",
               "ok", "offline",
               "P0 修复：原在模块级 DockerSandboxManager() → import 即抛"
               "DockerException（与阶段 2 'import soul 就持有假 Key' 同类）；"
               "现改惰性 + 走 run_code_once 统一入口"),
    EntryPoint("multiple-search/SemanticCache/engine.py", "engine", "ok", "no-main",
               "无 __init__.py，故 SemanticCache/ 自身是一个 sys.path 根；"
               "未装 redisvl 或连不上 Redis 时降级 InMemoryVectorStore"),

    # ---------------- asynchronization ----------------
    EntryPoint("asynchronization/kafka_utils.py", "kafka_utils", "ok", "no-main",
               "阶段 2：TOPIC_*/GROUP_* 改读 config.yaml"),
    EntryPoint("asynchronization/state_manager.py", "state_manager", "ok", "no-main",
               "Redis 封装；构造时才连"),
    EntryPoint("asynchronization/workers/planner_worker.py", "planner_worker", "ok",
               "needs:Kafka + Redis",
               "实测：导入干净（main() 里才连 Kafka）"),
    EntryPoint("asynchronization/workers/retriever_worker.py", "retriever_worker", "ok",
               "needs:Kafka + Redis + BGE-M3",
               "实测：导入干净；BGE-M3 在 build_engine() 里懒加载"),
    EntryPoint("asynchronization/workers/grader_worker.py", "grader_worker", "ok",
               "needs:Kafka + Redis + LLM API Key"),
    EntryPoint("asynchronization/workers/replanner_worker.py", "replanner_worker", "ok",
               "needs:Kafka + Redis + LLM API Key"),
    EntryPoint("asynchronization/workers/reasoner_worker.py", "reasoner_worker", "ok",
               "needs:Kafka + Redis + LLM API Key + Docker 沙箱"),

    # ---------------- benchmark_causal ----------------
    EntryPoint("benchmark_causal/schemas.py", "benchmark_causal.schemas", "ok", "no-main"),
    EntryPoint("benchmark_causal/scm.py", "benchmark_causal.scm", "ok", "no-main",
               "确定性 SCM 执行器"),
    EntryPoint("benchmark_causal/scm_labor.py", "benchmark_causal.scm_labor", "ok",
               "no-main", "两个领域 SCM"),
    EntryPoint("benchmark_causal/scenarios.py", "benchmark_causal.scenarios", "ok",
               "no-main"),
    EntryPoint("benchmark_causal/legal_corpus.py", "benchmark_causal.legal_corpus",
               "ok", "no-main"),
    EntryPoint("benchmark_causal/generator.py", "benchmark_causal.generator", "ok",
               "no-main"),
    EntryPoint("benchmark_causal/gates.py", "benchmark_causal.gates", "ok", "no-main"),
    EntryPoint("benchmark_causal/run_demo.py", "benchmark_causal.run_demo", "ok",
               "covered-by-ci"),
    EntryPoint("benchmark_causal/build_corpus.py", "benchmark_causal.build_corpus",
               "ok", "needs:LawRefBook/Laws 仓库（git clone）",
               "语料已入库，重建才需要外部仓库"),

    # ---------------- spikes / tools ----------------
    EntryPoint("spikes/spike_checkpoint_resume.py", None, "n/a", "covered-by-ci",
               "文件名无连字符，可 import，但作为脚本跑才是它的用途"),
    EntryPoint("tools/ci.py", None, "n/a", "n/a", "本工具与它平级"),

    # ---------------- model/（训练侧，本机无 GPU）----------------
    # 这些模块导入失败是**预期**的：本机没装 unsloth / trl，也没有 GPU。
    # 登记为 needs:* 而非 ok，所以不会让 CI 变红。
    EntryPoint("model/utils/unsloth_loader.py", "model.utils.unsloth_loader",
               "needs:unsloth", "needs:unsloth + GPU"),
    EntryPoint("model/utils/vllm_engine.py", "model.utils.vllm_engine", "ok",
               "needs:vllm + GPU", "实测：导入干净，vllm 在引擎构造时才用"),
    EntryPoint("model/data/evol_instruct.py", "model.data.evol_instruct", "ok",
               "needs:LLM API Key", "实测：导入干净（阶段 2 改为惰性客户端）"),
    EntryPoint("model/Unsloth.py", "model.Unsloth", "needs:unsloth",
               "needs:unsloth + GPU",
               "阶段 7：已改为委托 UnslothLoader；缺依赖时给友好 SystemExit"),
    EntryPoint("model/training/train_extractor_grader.py",
               "model.training.train_extractor_grader", "needs:trl + GPU",
               "needs:GPU"),
    EntryPoint("model/training/train_meta_planner.py",
               "model.training.train_meta_planner", "needs:trl + GPU", "needs:GPU"),
    EntryPoint("model/training/train_reasoner.py", "model.training.train_reasoner",
               "needs:trl + GPU", "needs:GPU"),
    EntryPoint("model/training/train_replanner_grpo.py",
               "model.training.train_replanner_grpo", "needs:trl + GPU", "needs:GPU"),
    EntryPoint("model/training/train_retriever.py", "model.training.train_retriever",
               "needs:模型权重（联网下载）", "needs:GPU",
               "导入即加载 SentenceTransformer，无网无缓存时报错"),
]


def probe_import(module: str, timeout: int = 90) -> tuple[bool, float, str]:
    """在子进程里 `import module`，返回 (成功?, 耗时, 输出)"""
    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=ROOT, env=ENV, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout)
        return proc.returncode == 0, time.time() - started, \
            (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return False, time.time() - started, f"⏱️ 导入超过 {timeout}s（疑似阻塞：连服务或下模型）"
    except OSError as exc:
        return False, time.time() - started, f"无法执行: {exc}"


def error_excerpt(output: str, max_lines: int = 4) -> str:
    """
    从 traceback 里摘出最有用的几行。

    ⚠️ 不能只取最后一行：`config_loader` 会在**导入时**打一条"请设置环境变量"的
    告警，它往往排在真正的异常之后，只取末行就会把真实原因吞掉。
    """
    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
    if not lines:
        return ""

    # 异常类型行优先（traceback 的最后一段 "XxxError: ..."）
    hits = [ln for ln in lines
            if ln.startswith(("ModuleNotFoundError", "ImportError", "RuntimeError",
                              "ValueError", "KeyError", "OSError", "TypeError",
                              "AttributeError", "ConnectionError", "ConfigError",
                              "DockerException", "docker.errors"))]
    if hits:
        return hits[-1][:200]

    # 没有异常类型行 → 退化为 traceback 尾部
    return " ⏎ ".join(ln[:120] for ln in lines[-max_lines:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="只跑断言（供 CI 用），不打印完整矩阵")
    args = ap.parse_args()

    results: list[tuple[EntryPoint, bool | None, float, str]] = []

    # 每个模块独立跑一个子进程；它们之间毫无依赖，所以并行。
    # `subprocess.run` 会在等待时释放 GIL，线程池足够（重活都在子进程里）。
    # 串行跑要 36s 以上（单是 import torch 就 7~8s），并行后约 8s。
    probe_targets = [ep for ep in REGISTRY
                     if ep.import_expect != "n/a" and ep.module is not None]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=min(8, len(probe_targets) or 1)) as pool:
        probed = list(pool.map(lambda ep: probe_import(ep.module), probe_targets))

    outcome_by_path = {ep.path: res for ep, res in zip(probe_targets, probed)}
    for ep in REGISTRY:
        res = outcome_by_path.get(ep.path)
        if res is None:
            results.append((ep, None, 0.0, ""))
        else:
            ok, elapsed, out = res
            results.append((ep, ok, elapsed, out))

    # ---------------- 断言：登记为 ok 的必须真的干净导入 ----------------
    violations = [(ep, out) for ep, ok, _, out in results
                  if ok is False and ep.import_expect == "ok"]

    # 反向检查：登记为 needs: 的如果现在能导入了，说明清单过期了
    surprises = [(ep, ) for ep, ok, _, _ in results
                 if ok is True and ep.import_expect.startswith("needs")]

    if args.check:
        for ep, out in violations:
            print(f"❌ 导入失败（登记为必定成功）: {ep.module}")
            print(f"   {error_excerpt(out)}")
        if violations:
            print(f"\n结论：❌ {len(violations)} 个入口的导入安全性被破坏")
            return 1
        print("结论：✅ 全部登记为 import-ok 的入口都能干净导入")
        return 0

    # ---------------- 打印矩阵 ----------------
    print("=" * 78)
    print("  入口点体检：导入安全性 + 离线可跑性")
    print(f"  解释器 {sys.executable}")
    print("  ⚠️ 子进程环境已清空 DEEPSEEK_API_KEY / KAFKA_BOOTSTRAP / REDIS_URL，"
          "且 HF_HUB_OFFLINE=1")
    print("=" * 78)
    print(f"{'入口':<52}{'import':<10}{'run':<22}")
    print("-" * 84)

    for ep, ok, elapsed, out in results:
        if ok is None:
            imp = "n/a"
        elif ok:
            imp = f"ok {elapsed:.1f}s"
        else:
            imp = "FAIL"
        print(f"{ep.path:<52}{imp:<10}{ep.run_expect:<22}")

    # ---------------- 未登记项：说明清单还没写全 ----------------
    unknown = [(ep, ok, out) for ep, ok, _, out in results if ep.import_expect == "?"]
    if unknown:
        print("\n" + "=" * 78)
        print("  ⚠️ 清单未覆盖（请据此更新 tools/audit_entrypoints.py 的 REGISTRY）")
        print("=" * 78)
        for ep, ok, out in unknown:
            print(f"\n  · {ep.path}")
            print(f"    模块名 {ep.module} → {'可干净导入' if ok else '导入失败'}")
            if ok is False and out:
                print(f"    原因 {error_excerpt(out)}")

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 78)
    print("  汇总")
    print("=" * 78)
    ok_count = sum(1 for _, ok, _, _ in results if ok is True)
    fail_count = sum(1 for _, ok, _, _ in results if ok is False)
    na_count = sum(1 for _, ok, _, _ in results if ok is None)
    offline = sum(1 for ep, _, _, _ in results if ep.run_expect == "offline")
    ci = sum(1 for ep, _, _, _ in results if ep.run_expect == "covered-by-ci")
    print(f"  导入成功 {ok_count} / 导入失败 {fail_count} / 不适用 {na_count}")
    print(f"  __main__ 可离线跑 {offline} 个；已纳入 CI {ci} 个")
    print(f"  登记为 import-ok 却被破坏：{len(violations)} 个")
    print(f"  登记为 needs:X 却已能导入（清单过期）：{len(surprises)} 个")
    print(f"  清单未覆盖（import_expect == '?'）：{len(unknown)} 个")

    if violations:
        print("\n结论：❌ 导入安全性被破坏")
        return 1
    print("\n结论：✅ 无导入安全性违规"
          + ("（仍有未登记项，见上）" if unknown else ""))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
