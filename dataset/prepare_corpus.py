"""
语料入库 —— 把「一堆法律文本文件」变成「图引擎能吃的节点 + 向量」
================================================================
产出两个文件，喂给 `LegalDenseGraphBuilder.build_initial_graph_batch()`：

    <out>/nodes.jsonl   每行一个节点 {"id": int, "content": str, "type": "Raw", "metadata": {...}}
    <out>/vectors.npy   float32 矩阵，形状 (N, dim)，行序与 nodes.jsonl **严格一一对应**

用法：
    python dataset/prepare_corpus.py --corpus-dir <目录> --out <输出目录>
    # 或走环境变量
    set CORPUS_DIR=<目录> & python dataset/prepare_corpus.py --out <输出目录>

    --stub-encoder   离线模式：用 StubEncoder 代替 BGE-M3（不联网、不下 2GB 模型）

⚠️ 本文件的历史（P0 修复）：
   原先它是**一份从未填写的模板** —— `CORPUS_DIR = "/path/to/your/corpus/directory"`、
   `OUTPUT_VECTORS_FILE = ""`，而且**所有逻辑都在模块级**（没有 `__main__` 保护），
   于是 `import dataset.prepare_corpus` 会在 `os.listdir()` 处直接抛
   `FileNotFoundError`，同时还要 `device='cuda'`。既不可导入也不可运行。
   现在改为：函数化 + `__main__` 保护 + 路径必须显式给出 + 离线可跑。

⚠️ 设计要点：
  · **节点 ID 必须是整数** —— 图引擎的顶点名规范是 `str(int)`，
    `dataset/graph.as_num_id()` 会拒绝非数字 ID（见 docs/CONTRACTS.md）。
  · 分块复用 `dataset/chunk.py::chunk_text`，不另写一份切分逻辑。
  · 文件名排序后再处理，保证**同样输入必得同样 ID**（可复现）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

# 路径引导：本文件在 dataset/ 下
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from dataset.chunk import chunk_text

SUPPORTED_SUFFIXES = (".txt", ".md", ".text")
NODES_FILENAME = "nodes.jsonl"
VECTORS_FILENAME = "vectors.npy"


# ===========================================================================
# 1. 扫描 + 分块
# ===========================================================================
def discover_files(corpus_dir: str) -> List[str]:
    """
    列出语料目录下支持的文件，**按文件名排序**。

    排序是必须的：节点 ID 依赖遍历顺序，不排序的话同样输入会得到不同 ID，
    语料与向量文件的对应关系就无法复现。
    """
    if not os.path.isdir(corpus_dir):
        raise FileNotFoundError(
            f"语料目录不存在：{corpus_dir!r}。"
            f"请用 --corpus-dir 指定，或设置环境变量 CORPUS_DIR。"
        )

    files = [os.path.join(corpus_dir, name)
             for name in sorted(os.listdir(corpus_dir))
             if name.lower().endswith(SUPPORTED_SUFFIXES)]
    if not files:
        raise FileNotFoundError(
            f"语料目录 {corpus_dir!r} 下没有找到任何 "
            f"{'/'.join(SUPPORTED_SUFFIXES)} 文件。"
        )
    return files


def build_nodes(files: List[str],
                min_chunk_chars: int = 100,
                chunk_mode: str = "paragraph") -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    把文件列表切成节点。返回 (nodes, texts)，两者等长且顺序一致。

    ID 全局递增且从 1 开始 —— 与图引擎的内部 ID 空间同口径。
    """
    nodes: List[Dict[str, Any]] = []
    texts: List[str] = []
    next_id = 1

    for path in files:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()

        chunks = chunk_text(raw, mode=chunk_mode, min_chunk_chars=min_chunk_chars)
        for idx, chunk in enumerate(chunks):
            nodes.append({
                "id": next_id,
                "content": chunk,
                "type": "Raw",
                "metadata": {
                    "source_file": os.path.basename(path),
                    "chunk_index": idx,
                    "length": len(chunk),
                },
            })
            texts.append(chunk)
            next_id += 1

    return nodes, texts


# ===========================================================================
# 2. 向量化
# ===========================================================================
def _default_encode_text() -> Callable[[str], Any]:
    """
    默认编码器 = 图引擎自己的（即 BGE-M3，**首次调用时才加载**）。

    这里刻意不 import 顶层 BGE-M3 —— 保持"只在真需要编码时才拉模型"。
    """
    from dataset.graph import LegalDenseGraphBuilder

    engine = LegalDenseGraphBuilder.from_config()
    return engine.encode_text


def vectorize(texts: List[str],
              encoder: Optional[Callable[[str], Any]] = None) -> "np.ndarray":
    """
    把文本列表编码成 (N, dim) float32 矩阵。

    :param encoder: 单条文本 → 向量。可为 None（用 BGE-M3）。
                    传入 `StubEncoder` 即可完全离线。
    """
    import numpy as np

    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    encode_text = encoder or _default_encode_text()

    vecs: List["np.ndarray"] = []
    expected: Optional[int] = None
    for text in texts:
        out = encode_text(text)
        # 兼容三种返回形态：{"dense": ...} / {"dense_vecs": ...} / 裸向量
        if isinstance(out, dict):
            if "dense" in out:
                vec = out["dense"]
            elif "dense_vecs" in out:
                vec = out["dense_vecs"][0]
            else:
                raise ValueError(
                    f"编码器返回的字典里既没有 'dense' 也没有 'dense_vecs'：{list(out)}")
        else:
            vec = out

        arr = np.asarray(vec, dtype=np.float32).ravel()

        # 失败要响：**逐条**校验长度。如果拖到 np.vstack 才炸，
        # 得到的是 numpy 那句"除拼接轴以外维度必须一致"，完全看不出是谁的问题。
        if expected is None:
            expected = int(arr.shape[0])
        elif arr.shape[0] != expected:
            raise ValueError(
                f"编码器返回的向量长度不一致：先 {expected} 维，"
                f"后 {arr.shape[0]} 维（文本 {text[:30]!r}…）"
            )
        vecs.append(arr)

    matrix = np.vstack(vecs).astype(np.float32)

    if matrix.ndim != 2 or matrix.shape[0] != len(texts):
        raise ValueError(
            f"向量矩阵形状 {matrix.shape} 与文本数 {len(texts)} 不匹配。"
        )
    return matrix


# ===========================================================================
# 3. 落盘
# ===========================================================================
def save(nodes: List[Dict[str, Any]], vectors: "np.ndarray", out_dir: str) -> Dict[str, str]:
    """写 nodes.jsonl + vectors.npy，返回两个文件的绝对路径"""
    import numpy as np

    if len(nodes) != int(vectors.shape[0]):
        # 这一条一旦不成立，后续 `build_initial_graph_batch(nodes, embeddings)`
        # 会以极难定位的方式失败（FAISS 维度/行数报错）。宁可在源头拦。
        raise ValueError(
            f"节点数 {len(nodes)} 与向量行数 {vectors.shape[0]} 不一致，拒绝写出。"
        )

    os.makedirs(out_dir, exist_ok=True)
    nodes_path = os.path.join(out_dir, NODES_FILENAME)
    vectors_path = os.path.join(out_dir, VECTORS_FILENAME)

    with open(nodes_path, "w", encoding="utf-8") as fh:
        for node in nodes:
            fh.write(json.dumps(node, ensure_ascii=False) + "\n")

    np.save(vectors_path, vectors)
    return {"nodes": nodes_path, "vectors": vectors_path}


def prepare_corpus(corpus_dir: str, out_dir: str,
                   encoder: Optional[Callable[[str], Any]] = None,
                   min_chunk_chars: int = 100,
                   chunk_mode: str = "paragraph") -> Dict[str, Any]:
    """
    完整流程：扫描 → 分块 → 向量化 → 落盘。

    :return: 统计信息（节点数、维度、两个产物路径）
    """
    files = discover_files(corpus_dir)
    nodes, texts = build_nodes(files, min_chunk_chars=min_chunk_chars,
                               chunk_mode=chunk_mode)
    vectors = vectorize(texts, encoder=encoder)
    paths = save(nodes, vectors, out_dir)

    return {
        "files": len(files),
        "nodes": len(nodes),
        "dim": int(vectors.shape[1]) if vectors.ndim == 2 and vectors.size else 0,
        "nodes_path": paths["nodes"],
        "vectors_path": paths["vectors"],
    }


# ===========================================================================
# 4. 命令行
# ===========================================================================
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="语料入库：文本文件 → 节点 + 向量")
    ap.add_argument("--corpus-dir", default=os.environ.get("CORPUS_DIR"),
                    help="语料目录（也可用环境变量 CORPUS_DIR）")
    ap.add_argument("--out", default=os.path.join(_THIS_DIR, "corpus_out"),
                    help="输出目录（默认 dataset/corpus_out）")
    ap.add_argument("--min-chunk-chars", type=int, default=100,
                    help="短块向上合并的阈值")
    ap.add_argument("--chunk-mode", default="paragraph",
                    choices=["paragraph", "sentence"])
    ap.add_argument("--stub-encoder", action="store_true",
                    help="离线模式：用 StubEncoder 代替 BGE-M3（不联网、不下模型）")
    args = ap.parse_args(argv)

    if not args.corpus_dir:
        # 失败要响：不猜路径，直接说清楚该怎么给
        print("❌ 缺少语料目录。请用 --corpus-dir 指定，或设置环境变量 CORPUS_DIR。",
              file=sys.stderr)
        print("   示例：python dataset/prepare_corpus.py --corpus-dir ./laws "
              "--out ./dataset/corpus_out --stub-encoder", file=sys.stderr)
        return 2

    encoder = None
    if args.stub_encoder:
        from dataset.graph import StubEncoder

        class _Stub:
            """把 StubEncoder 适配成「单条文本 → 裸向量」"""

            def __init__(self, dim: int):
                self._enc = StubEncoder(dim)

            def __call__(self, text: str):
                return self._enc.encode([text])["dense_vecs"][0]

        from config_loader import cfg
        encoder = _Stub(cfg.get("embedding", "dimension", default=1024))

    print(f"[准备] 语料目录 {args.corpus_dir}")
    stats = prepare_corpus(args.corpus_dir, args.out, encoder=encoder,
                           min_chunk_chars=args.min_chunk_chars,
                           chunk_mode=args.chunk_mode)
    print(f"[完成] {stats['files']} 个文件 → {stats['nodes']} 个节点，维度 {stats['dim']}")
    print(f"       nodes   → {stats['nodes_path']}")
    print(f"       vectors → {stats['vectors_path']}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
