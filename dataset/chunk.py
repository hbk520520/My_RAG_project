"""
文本分块 —— 把 PDF 拆成语义完整的片段
==================================
不做粗暴的定长切割，而是按段落/句子自然边界分块。
太短的向上合并，保证每一块都有独立可读的语义。

技术栈: PyMuPDF (fitz) / re
"""
import re
import fitz
from typing import List, Optional

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    从 PDF 中提取全部文本（按页顺序）。
    使用 PyMuPDF，兼容中文，自动处理大部分编码问题。
    """
    doc = fitz.open(pdf_path)
    full_text = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")  # 也可用 "blocks" 按区块排序
        full_text.append(text)
    doc.close()
    return "\n".join(full_text)


def chunk_text(
    text: str,
    mode: str = "paragraph",          # "paragraph" 或 "sentence"
    min_chunk_chars: int = 100,       # 最小块字符数（短块向上合并）
    max_chunk_chars: Optional[int] = None,  # 最大块字符数（若超长则不再细分）
    strip_whitespace: bool = True,
) -> List[str]:
    """
    将文本按自然边界分割为语义完整的块。
    - mode="paragraph": 按空行（两个及以上换行）分割，再根据 min_chunk_chars 合并过短的段落。
    - mode="sentence": 按中英文句号、问号、感叹号等分割，再合并短句。
    返回块列表。
    """
    if strip_whitespace:
        text = text.strip()

    # ----- 1. 初步切分 -----
    if mode == "paragraph":
        # 按连续换行（至少一个空行）分割
        raw_chunks = re.split(r"\n\s*\n", text)
    elif mode == "sentence":
        # 按句末标点切分（保留标点在前面一句）
        raw_chunks = re.split(r"(?<=[。！？.!?])\s*", text)
    else:
        raise ValueError("mode must be 'paragraph' or 'sentence'")

    # 去除空白块
    raw_chunks = [c.strip() for c in raw_chunks if c.strip()]

    # ----- 2. 合并过短的块（保证语义完整，避免碎片化）-----
    merged_chunks = []
    buffer = ""
    for chunk in raw_chunks:
        if not buffer:
            buffer = chunk
        else:
            # 如果当前缓冲区长度不足 min_chunk_chars，继续向后合并
            if len(buffer) < min_chunk_chars:
                buffer += "\n" + chunk
            else:
                # 缓冲区已足够长，保存并开始新块
                merged_chunks.append(buffer)
                buffer = chunk

    # 处理最后的缓冲区
    if buffer:
        # 若最后一个块太短，且前面有块，则并入前一块
        if len(buffer) < min_chunk_chars and merged_chunks:
            merged_chunks[-1] += "\n" + buffer
        else:
            merged_chunks.append(buffer)

    # ----- 3. 可选的最大长度处理（尽量避免，仅当用户指定时）-----
    if max_chunk_chars:
        final_chunks = []
        for chunk in merged_chunks:
            if len(chunk) > max_chunk_chars:
                # 如果超长，再次按句子拆分，但仍保证 min_chunk_chars
                sub_chunks = chunk_text(
                    chunk, mode="sentence",
                    min_chunk_chars=min_chunk_chars,
                    max_chunk_chars=None  # 避免无限递归
                )
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        return final_chunks

    return merged_chunks


# ============================================================================
# 使用示例
# ============================================================================
if __name__ == "__main__":
    pdf_file = "example.pdf"  # 替换为你的 PDF 路径

    # 1. 提取纯文本
    raw_text = extract_text_from_pdf(pdf_file)
    print(f"提取文本总长度：{len(raw_text)} 字符")

    # 2. 按段落分块（自动合并小于100字的段落，不设最大长度限制）
    chunks = chunk_text(raw_text, mode="paragraph", min_chunk_chars=100)

    print(f"\n共生成 {len(chunks)} 个块：")
    for i, ch in enumerate(chunks[:5]):  # 只打印前5块预览
        print(f"\n--- Chunk {i+1} (len={len(ch)}) ---")
        print(ch[:200] + "..." if len(ch) > 200 else ch)