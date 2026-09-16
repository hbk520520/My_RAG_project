"""
P0 收尾 · 语料入库链路 + 导入安全性
====================================
覆盖三件 P0 才发现的问题：

  1. `dataset/prepare_corpus.py` 原先是一份**从未填写的模板**
     （`CORPUS_DIR = "/path/to/your/corpus/directory"`），而且所有逻辑写在
     **模块级** —— `import` 就会在 `os.listdir()` 处抛 `FileNotFoundError`。
     现在函数化 + `__main__` 保护，这里锁住它的行为。
  2. `dataset/chunk.py` 是 `prepare_corpus` 复用的分块器，此前**完全没有测试**
     （它是纯函数，不需要 PDF 就能测）。
  3. `example_usage.py` 原先在模块级执行 `DockerSandboxManager()`，没有 Docker
     的环境连 `import` 都失败 —— 与阶段 2 的"import soul 就持有假 API Key"同类。

设计要点（与 docs/CONTRACTS.md 对齐）：
  · 节点 ID **必须是 int** —— 图引擎顶点名规范是 `str(int)`，`as_num_id()` 会拒绝非数字
  · `nodes.jsonl` 与 `vectors.npy` 的**行序必须严格一一对应**，否则
    `build_initial_graph_batch(nodes, embeddings)` 会以极难定位的方式失败
"""
import json
import os

import numpy as np
import pytest

from dataset.chunk import chunk_text
from dataset.prepare_corpus import (NODES_FILENAME, VECTORS_FILENAME,
                                    build_nodes, discover_files, prepare_corpus,
                                    save, vectorize)


# ===========================================================================
# 夹具
# ===========================================================================
class _FixedEncoder:
    """确定性编码器：同一文本必得同一向量，长度固定"""

    def __init__(self, dim: int = 8):
        self.dim = dim
        self.calls = 0

    def __call__(self, text: str) -> np.ndarray:
        import hashlib

        self.calls += 1
        h = int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest(),
                           "big")
        v = np.random.RandomState(h % (2 ** 31)).randn(self.dim).astype(np.float32)
        return v / np.linalg.norm(v)


def _write_corpus(dir_path, files: dict) -> str:
    os.makedirs(dir_path, exist_ok=True)
    for name, content in files.items():
        with open(os.path.join(dir_path, name), "w", encoding="utf-8") as fh:
            fh.write(content)
    return str(dir_path)


def _read_text(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# ===========================================================================
# chunk_text（prepare_corpus 复用的分块器）
# ===========================================================================
def test_chunk_text_merges_short_paragraphs():
    text = "短段落一。\n\n短段落二。"
    assert chunk_text(text, min_chunk_chars=100) == ["短段落一。\n短段落二。"]


def test_chunk_text_keeps_paragraphs_when_long_enough():
    a, b = "甲" * 120, "乙" * 120
    assert chunk_text(f"{a}\n\n{b}", min_chunk_chars=100) == [a, b]


def test_chunk_text_sentence_mode():
    chunks = chunk_text("第一句。第二句！第三句？", mode="sentence", min_chunk_chars=1)
    assert chunks == ["第一句。", "第二句！", "第三句？"]


def test_chunk_text_rejects_unknown_mode():
    with pytest.raises(ValueError, match="paragraph"):
        chunk_text("任意文本", mode="char")


def test_chunk_text_max_chunk_chars_splits_by_sentence():
    text = "甲" * 50 + "。" + "乙" * 50 + "。"
    chunks = chunk_text(text, min_chunk_chars=1, max_chunk_chars=60)
    assert len(chunks) == 2, "超长块应按句号再切"
    assert all(len(c) <= 60 for c in chunks)


def test_chunk_text_empty_input_gives_no_chunks():
    assert chunk_text("   \n\n  ", min_chunk_chars=10) == []


# ===========================================================================
# discover_files
# ===========================================================================
def test_discover_files_rejects_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError, match="语料目录不存在"):
        discover_files(str(tmp_path / "definitely_missing"))


def test_discover_files_rejects_dir_without_supported_files(tmp_path):
    (tmp_path / "readme.pdf").write_text("x", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="没有找到任何"):
        discover_files(str(tmp_path))


def test_discover_files_is_sorted_and_filters_suffixes(tmp_path):
    _write_corpus(tmp_path, {"c.md": "c", "a.txt": "a", "b.text": "b",
                             "skip.pdf": "no"})
    names = [os.path.basename(p) for p in discover_files(str(tmp_path))]
    assert names == ["a.txt", "b.text", "c.md"], "必须排序，否则 ID 不可复现"


# ===========================================================================
# build_nodes
# ===========================================================================
def test_build_nodes_assigns_contiguous_int_ids_from_one(tmp_path):
    corpus = _write_corpus(tmp_path / "laws", {
        "a.txt": "甲" * 120 + "\n\n" + "乙" * 120,
        "b.txt": "丙" * 120,
    })
    files = discover_files(corpus)
    nodes, texts = build_nodes(files, min_chunk_chars=100)

    assert [n["id"] for n in nodes] == [1, 2, 3]
    assert all(isinstance(n["id"], int) for n in nodes), "ID 必须是 int（图引擎顶点名规范）"
    assert len(nodes) == len(texts)
    assert [n["content"] for n in nodes] == texts
    assert nodes[0]["metadata"]["source_file"] == "a.txt"
    assert nodes[0]["metadata"]["chunk_index"] == 0
    assert nodes[1]["metadata"]["chunk_index"] == 1, "同一文件内的块要连续编号"
    assert nodes[2]["metadata"]["source_file"] == "b.txt"


def test_build_nodes_is_deterministic(tmp_path):
    corpus = _write_corpus(tmp_path / "laws", {"a.txt": "甲" * 120 + "\n\n" + "乙" * 120})
    first, _ = build_nodes(discover_files(corpus), min_chunk_chars=100)
    second, _ = build_nodes(discover_files(corpus), min_chunk_chars=100)
    assert first == second, "同样输入必须得到同样 ID（否则向量与节点会对不上）"


def test_build_nodes_on_empty_file_yields_nothing(tmp_path):
    corpus = _write_corpus(tmp_path / "laws", {"empty.txt": "   "})
    nodes, texts = build_nodes(discover_files(corpus))
    assert nodes == [] and texts == []


# ===========================================================================
# vectorize
# ===========================================================================
def test_vectorize_shapes_and_order():
    enc = _FixedEncoder(dim=8)
    matrix = vectorize(["a", "b", "c"], encoder=enc)
    assert matrix.shape == (3, 8)
    assert matrix.dtype == np.float32
    assert enc.calls == 3, "每个文本编码一次，不多不少"


def test_vectorize_accepts_all_three_return_shapes():
    dim = 8

    bare = vectorize(["x"], encoder=lambda t: np.ones(dim, dtype=np.float32))
    dense = vectorize(["x"], encoder=lambda t: {"dense": np.ones(dim, dtype=np.float32)})
    native = vectorize(["x"], encoder=lambda t: {
        "dense_vecs": np.ones((1, dim), dtype=np.float32),
        "lexical_weights": [{}],
    })
    for m in (bare, dense, native):
        assert m.shape == (1, dim)


def test_vectorize_rejects_dict_without_dense_keys():
    with pytest.raises(ValueError, match="dense"):
        vectorize(["x"], encoder=lambda t: {"sparse": {}})


def test_vectorize_rejects_inconsistent_dimensions():
    """维度随文本长度变化 → 必须在源头报出可定位的错误，而不是让 numpy 抛"""
    with pytest.raises(ValueError, match="长度不一致"):
        vectorize(["a", "bbbb"], encoder=lambda t: np.zeros(len(t), dtype=np.float32))


def test_vectorize_empty_input():
    matrix = vectorize([], encoder=_FixedEncoder())
    assert matrix.shape == (0, 0)


# ===========================================================================
# save
# ===========================================================================
def test_save_writes_aligned_artifacts(tmp_path):
    nodes = [{"id": 1, "content": "甲", "type": "Raw", "metadata": {}},
             {"id": 2, "content": "乙", "type": "Raw", "metadata": {}}]
    vectors = np.eye(2, 4, dtype=np.float32)

    paths = save(nodes, vectors, str(tmp_path / "out"))

    assert os.path.basename(paths["nodes"]) == NODES_FILENAME
    assert os.path.basename(paths["vectors"]) == VECTORS_FILENAME

    with open(paths["nodes"], encoding="utf-8") as fh:
        loaded = [json.loads(line) for line in fh if line.strip()]
    assert loaded == nodes, "jsonl 必须一行一个节点、保持原序"

    assert np.load(paths["vectors"]).shape == (2, 4)


def test_save_refuses_mismatched_counts(tmp_path):
    """行数不一致一旦写出，下游 build_initial_graph_batch 会以极难定位的方式失败"""
    nodes = [{"id": 1, "content": "甲", "type": "Raw", "metadata": {}}]
    with pytest.raises(ValueError, match="不一致"):
        save(nodes, np.zeros((3, 4), dtype=np.float32), str(tmp_path / "out"))


# ===========================================================================
# prepare_corpus 端到端
# ===========================================================================
def test_prepare_corpus_end_to_end_offline(tmp_path):
    corpus = _write_corpus(tmp_path / "laws", {
        "a.md": "# 标题\n\n" + "甲" * 120 + "\n\n" + "乙" * 120,
        "b.txt": "丙" * 120,
    })
    out = str(tmp_path / "out")

    stats = prepare_corpus(corpus, out, encoder=_FixedEncoder(dim=16))

    assert stats["files"] == 2
    assert stats["nodes"] == 3
    assert stats["dim"] == 16

    with open(stats["nodes_path"], encoding="utf-8") as fh:
        nodes = [json.loads(line) for line in fh if line.strip()]
    vectors = np.load(stats["vectors_path"])

    assert int(vectors.shape[0]) == len(nodes) == 3
    assert vectors.shape[1] == 16
    assert [n["id"] for n in nodes] == [1, 2, 3]


def test_prepare_corpus_output_is_reproducible(tmp_path):
    corpus = _write_corpus(tmp_path / "laws", {"a.txt": "甲" * 120 + "\n\n" + "乙" * 120})

    s1 = prepare_corpus(corpus, str(tmp_path / "o1"), encoder=_FixedEncoder(dim=8))
    s2 = prepare_corpus(corpus, str(tmp_path / "o2"), encoder=_FixedEncoder(dim=8))

    assert _read_text(s1["nodes_path"]) == _read_text(s2["nodes_path"])
    assert np.array_equal(np.load(s1["vectors_path"]), np.load(s2["vectors_path"]))


def test_prepare_corpus_reports_missing_dir_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="语料目录不存在"):
        prepare_corpus(str(tmp_path / "nope"), str(tmp_path / "out"),
                       encoder=_FixedEncoder())


def test_prepare_corpus_cli_without_dir_returns_usage_error():
    """CLI 缺参数时必须给用法提示、返回 2，而不是抛堆栈"""
    from dataset.prepare_corpus import main

    assert main([]) == 2
    assert main(["--corpus-dir", ""]) == 2


# ===========================================================================
# 导入安全性（P0 修复的两个模块）
# ===========================================================================
def test_example_usage_imports_without_docker():
    """
    回归：原先在模块级执行 `DockerSandboxManager()`（会真的连 Docker daemon），
    没有 Docker 的环境连 `import` 都失败。
    """
    import example_usage

    assert callable(example_usage.node_execute_code)
    assert callable(example_usage.terminate_session)
    # 惰性：导入后不该已经建好管理器
    assert example_usage._manager_cache is None


def test_prepare_corpus_has_no_import_side_effects(tmp_path):
    """回归：原先所有逻辑在模块级，import 就会 os.listdir 一个占位符路径"""
    import importlib

    import dataset.prepare_corpus as pc

    # 重新导入一次不应产生任何文件
    before = os.listdir(tmp_path)
    importlib.reload(pc)
    assert os.listdir(tmp_path) == before
    assert pc.NODES_FILENAME == "nodes.jsonl"
