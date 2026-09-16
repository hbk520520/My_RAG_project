#!/usr/bin/env python
"""
一键回归（M0-T4）
=================
把"改完代码该怎么验证"从**口头约定**变成**一条命令**。

    python tools/ci.py

跑的五类检查（任一失败即退出码非 0）：

  1. **导入安全性** —— 每个入口 `import` 都不得产生副作用
     （连服务 / 读凭据 / 下模型 / 打印）
  2. **compileall**  —— 语法/字节码层面能编译
  3. **pytest**      —— 全部单元 + 端到端沙盘测试
  4. **离线 demo**   —— 每个子系统的 `__main__` 真跑一遍（不是只 import）
  5. **P0 spike**    —— LangGraph 检查点/恢复能力仍然成立

为什么第 1 步不能省：`import` 即副作用是本项目**反复**出问题的一类 ——
  阶段 2 `import soul` 就持有写死的假 API Key；
  阶段 6 `training_data_guide.py` 顶层 `print`；
  P0 `example_usage.py` 模块级 `DockerSandboxManager()`。
  `compileall` 与 `pytest` 都抓不到（pytest 走 conftest 已经铺好环境）。

为什么需要第 4 步：`compileall` 通过 ≠ 能跑（本项目反复踩过）。
为什么必须**离线**：要 API key / 要下载 2GB 模型 / 要 Kafka 的检查，
不配叫回归测试 —— 它们的结果不可重复。

离线护栏：脚本强制设置 `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`。
一旦某个 demo 意外试图下载 BGE-M3，它会**立刻报错**而不是静默下 2GB。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 环境变量：既保证离线可复现，也保证中文输出不乱码
ENV = {
    **os.environ,
    "PYTHONIOENCODING": "utf-8",
    "PYTHONUTF8": "1",
    "HF_HUB_OFFLINE": "1",          # 禁止下载模型
    "TRANSFORMERS_OFFLINE": "1",
    "PYTHONPATH": os.pathsep.join([
        ROOT,
        os.path.join(ROOT, "asynchronization"),
        os.path.join(ROOT, "asynchronization", "workers"),
        os.path.join(ROOT, "multiple-search"),
        os.path.join(ROOT, "multiple-search", "legal_sandbox"),
    ]),
}

# 全部离线可跑的 demo（显式白名单，新增的请先确认无网络/凭据依赖再加进来）
OFFLINE_DEMOS = [
    ("图引擎（BGE-M3 + FAISS + igraph）", "dataset/graph.py"),
    ("记忆↔图 桥接（双写/脏同步/tombstone）", "dataset/memory_graph_bridge.py"),
    ("增量记忆（GMM 动态阈值）", "dataset/IncrementalMemoryManager.py"),
    ("Agent 状态机（LangGraph 全图）", "multiple-search/soul.py"),
    ("因果评测沙盘（合法语料 + SCM）", "benchmark_causal/run_demo.py"),
    ("沙箱工具返回契约（成功/失败/清理）",
     "multiple-search/legal_sandbox/example_usage.py"),
]
# 注：`dataset/prepare_corpus.py` 需要 --corpus-dir 入参，不适合放进这个白名单；
#     它的整条链路（扫描→分块→向量化→落盘）由 tests/test_corpus_ingest.py 覆盖。

COMPILE_EXCLUDE = (".git", "__pycache__", ".pytest_cache", "saved_loras",
                   "RAG_data", "training_data", "outputs", ".continue")

TAIL_LINES = 25


def _banner(text: str) -> None:
    print(f"\n{text}\n" + "-" * 72)


def _run(cmd, label: str, timeout: int = 900):
    """跑一条命令，返回 (ok, 耗时秒, 输出)。失败时打印输出尾部。"""
    started = time.time()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, env=ENV, capture_output=True,
                              text=True, encoding="utf-8", errors="replace",
                              timeout=timeout)
        elapsed = time.time() - started
        ok = proc.returncode == 0
        output = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        elapsed = time.time() - started
        ok = False
        output = f"⏱️ 超过 {timeout}s 未结束 —— 视为失败（可能是死循环或阻塞等待）"
    except OSError as exc:
        elapsed = time.time() - started
        ok = False
        output = f"无法执行: {exc}"

    mark = "✅" if ok else "❌"
    print(f"{mark} {label}  ({elapsed:.1f}s)")
    if not ok:
        tail = "\n".join(output.strip().splitlines()[-TAIL_LINES:])
        print("   ── 输出尾部 ──")
        for line in tail.splitlines():
            print(f"   | {line}")
    return ok, elapsed, output


def main() -> int:
    print("=" * 72)
    print("  一键回归 —— 导入安全 + compileall + pytest + 离线 demo + P0 spike")
    print(f"  解释器: {sys.executable}")
    print(f"  工作目录: {ROOT}")
    print("=" * 72)

    results = []

    # ---- 1. 入口点导入安全性 ----
    _banner("[1/5] 入口点导入安全性")
    results.append((*_run([sys.executable, "tools/audit_entrypoints.py", "--check"],
                          "audit: entrypoint imports", timeout=900),
                    "entrypoint imports"))

    # ---- 2. compileall ----
    _banner("[2/5] 语法编译检查")
    results.append((*_run(
        [sys.executable, "-m", "compileall", "-q",
         "-x", "|".join(COMPILE_EXCLUDE), "."],
        "compileall"), "compileall"))

    # ---- 3. pytest ----
    _banner("[3/5] 测试套件")
    results.append((*_run([sys.executable, "-m", "pytest", "-q"],
                          "pytest", timeout=1800), "pytest"))

    # ---- 4. 离线 demo ----
    _banner("[4/5] 离线端到端 demo（每个子系统的 __main__ 真跑一遍）")
    demo_ok = True
    for label, script in OFFLINE_DEMOS:
        ok, _, _ = _run([sys.executable, script], f"demo: {label}  [{script}]")
        demo_ok = demo_ok and ok
    results.append((demo_ok, 0.0, "", "offline demos"))

    # ---- 5. P0 spike ----
    _banner("[5/5] P0 spike：LangGraph 检查点 / 跨进程恢复")
    results.append((*_run([sys.executable, "spikes/spike_checkpoint_resume.py"],
                          "spike: checkpoint & resume", timeout=300),
                    "P0 spike"))

    # ---- 汇总 ----
    _banner("汇总")
    failed = []
    for ok, elapsed, _out, name in results:
        print(f"  {'✅ 通过' if ok else '❌ 失败'}   {name}")
        if not ok:
            failed.append(name)

    if failed:
        print(f"\n结论：❌ {len(failed)} 项失败 —— {', '.join(failed)}")
        return 1

    print("\n结论：✅ 全绿")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
