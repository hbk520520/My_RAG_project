"""
pytest 公共装置
==============
本项目是多入口布局：根目录 / asynchronization/ / multiple-search/ /
multiple-search/legal_sandbox/ 各自都是 sys.path 根，生产代码靠每个文件顶部的
「路径引导」片段自救。测试侧在这里统一做一次，避免每个测试文件各抄一份。

另外 pytest 默认只把 tests/ 目录加进 sys.path，所以这个引导是必需的
（不是冗余）。
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 注意：这里「倒序插入」是为了让 ROOT 最终排在 sys.path 最前面。
# 与 asynchronization/entrypoint.sh 里给 Worker 设置的 PYTHONPATH 保持一致
# （根目录 + asynchronization + asynchronization/workers）。
_EXTRA_PATHS = [
    ROOT,
    os.path.join(ROOT, "asynchronization"),
    os.path.join(ROOT, "asynchronization", "workers"),
    os.path.join(ROOT, "multiple-search"),
    os.path.join(ROOT, "multiple-search", "legal_sandbox"),
]

for _p in reversed(_EXTRA_PATHS):
    if _p not in sys.path:
        sys.path.insert(0, _p)


@pytest.fixture
def repo_root() -> str:
    """仓库根目录绝对路径"""
    return ROOT


@pytest.fixture
def fake_container():
    """一个只记录 remove() 是否被调用的假容器，用于验证沙箱不泄漏"""
    class _FakeContainer:
        def __init__(self):
            self.removed = False
            self.force_used = None

        def remove(self, force=False):
            self.removed = True
            self.force_used = force

    return _FakeContainer()


# ===========================================================================
# P0 · 离线端到端沙盘夹具（见 tests/harness.py 的说明）
# ===========================================================================
@pytest.fixture
def programmable_llm():
    """可编程假 LLM —— 替代真实 API，并记录每次调用"""
    from tests.harness import ProgrammableLLM
    return ProgrammableLLM()


@pytest.fixture
def mock_kafka():
    """内存任务总线 —— 替代真实 Kafka"""
    from tests.harness import MockKafka
    return MockKafka()


@pytest.fixture
def offline_engine():
    """已注入 StubEncoder 的图引擎（纯本地，不下载 BGE-M3）"""
    from tests.harness import make_offline_graph_engine
    return make_offline_graph_engine()
