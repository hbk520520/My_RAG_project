"""
法条语料构建器 —— 从开源法条库生成项目内统一格式的 JSONL
=========================================================
数据源：LawRefBook/Laws（https://github.com/LawRefBook/Laws）
  · `db.sqlite3` 的 `law` 表提供元数据，含 **valid_from / valid_to / ver**
    —— 这是判定「法条在案件发生日是否有效」的关键字段
  · 同目录下的 markdown 提供逐条正文（`## 第X章` 分章、`第X条` 分条）

为什么从外部库构建、而不是手工维护：
  1. 法条正文庞大，手工维护必然是错的；
  2. 需要 **时效区间** 才能做「法不溯及既往」判定，自建数据没有这个字段；
  3. 《著作权法》第五条：法律、法规等文件不受著作权保护，故正文可自由再分发。

合规说明：
  · 本脚本只提取**法规正文与元数据**，不提取该仓库的整理结构/脚本；
  · 生成的 JSONL 里保留 `source` 字段记录来源与 commit，便于溯源审计；
  · 已废止的法条**一并保留**（`valid_to` 早于 2099 即视为已失效），
    因为「模型会不会引用失效法条」本身就是评测项。

用法：
    set LAWS_REPO=D:\\path\\to\\LawRefBook-Laws
    python -m benchmark_causal.build_corpus
    或： python benchmark_causal/build_corpus.py --repo <path> --out <path>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from typing import Dict, Iterable, List, Optional

# 本项目第一阶段只需要劳动争议相关的最小集合
DEFAULT_LAW_NAMES = [
    "劳动合同法",              # valid 2013-07-01 ~ 现行
    "劳动法",                  # valid 2018-12-29 ~ 现行
    "劳动合同法实施条例",
    "劳动争议调解仲裁法",
    "民法典 合同编",
    "民法典 侵权责任编",
    "最高人民法院关于审理劳动争议案件适用法律问题的解释（一）",
    "最高人民法院关于适用《民法典》时间效力的若干规定",   # 法不溯及既往的权威锚点
    "合同法",                  # 已废止（valid_to=2019-12-31）—— 用于"引用失效法条"测试
]

# 视为"现行有效"的哨兵日期
OPEN_END = "2099-12-31"

_CN_DIGITS = "〇零一二三四五六七八九十百千两"
_ARTICLE_RE = re.compile(rf"^第([{_CN_DIGITS}\d]+)条\s*(.*)$")
_CHAPTER_RE = re.compile(r"^#{2,6}\s*(第[^\s]*[章节])\s*(.*)$")
_HEADING_RE = re.compile(r"^#{1,6}\s+")
_INFO_END = "<!-- INFO END -->"

_CN_NUM = {"〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
           "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def cn_to_int(text: str) -> Optional[int]:
    """
    中文数字 → int（支持「八十七」「一百零三」「一千二百」这类法律条号写法）。

    只实现条号需要的范围（1–9999），不做通用中文数字解析。
    """
    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    total, section, number = 0, 0, 0
    for ch in text:
        if ch in _CN_NUM:
            number = _CN_NUM[ch]
        elif ch == "十":
            section += (number or 1) * 10
            number = 0
        elif ch == "百":
            section += (number or 1) * 100
            number = 0
        elif ch == "千":
            section += (number or 1) * 1000
            number = 0
        else:
            return None
    return total + section + number


def parse_markdown_law(md_text: str) -> List[Dict[str, str]]:
    """
    把一部法律的 markdown 拆成条目。

    结构约定（该库实际格式）：
        # 法律标题
        （发布/修正说明若干行）
        <!-- INFO END -->
        ## 第一章 总 则
        第一条 正文……
        第二条 正文……
        （续行：不以「第X条」开头，属于上一条正文）
    """
    if _INFO_END in md_text:
        md_text = md_text.split(_INFO_END, 1)[1]

    articles: List[Dict[str, str]] = []
    chapter = ""
    current_no: Optional[str] = None
    buffer: List[str] = []

    def flush():
        if current_no is not None:
            body = "\n".join(buffer).strip()
            if body:
                articles.append({
                    "article_no": current_no,
                    "article_index": str(cn_to_int(current_no.strip("第条")) or 0),
                    "chapter": chapter,
                    "text": body,
                })

    for raw_line in md_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            if buffer:
                buffer.append("")
            continue

        chap = _CHAPTER_RE.match(line)
        if chap:
            flush()
            current_no, buffer = None, []
            chapter = (chap.group(1) + " " + chap.group(2)).strip()
            continue

        art = _ARTICLE_RE.match(line.strip())
        if art:
            flush()
            current_no = f"第{art.group(1)}条"
            buffer = [art.group(2).strip()] if art.group(2).strip() else []
            continue

        if _HEADING_RE.match(line):        # 其他级别的标题（如 #### 小节）不计入正文
            continue

        if current_no is not None:
            buffer.append(line)

    flush()
    return articles


def _git_commit(repo: str) -> str:
    try:
        out = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


_MD_DATE_RE = re.compile(r"^(?P<name>.+?)\((?P<date>\d{4}-\d{2}-\d{2})\)?$")


def _index_markdown(repo: str) -> Dict[str, List[Dict[str, str]]]:
    """
    扫描仓库，建立 {法名: [{date, path}, ...]} 索引。

    该库的 markdown 文件名为 `法名(发布日期).md`，但也有少数不带日期、
    或 `db.sqlite3.filename` 为空/与磁盘不一致的情况，所以不直接信 filename，
    改为按「法名 + 发布日期」匹配 —— 同一部法有多个版本时能正确区分。
    """
    index: Dict[str, List[Dict[str, str]]] = {}
    for path in _walk_md(repo):
        stem = os.path.splitext(os.path.basename(path))[0]
        m = _MD_DATE_RE.match(stem)
        name = (m.group("name") if m else stem).strip()
        date = (m.group("date") if m else "").strip()
        index.setdefault(name, []).append({"date": date, "path": path})
    return index


def _resolve_md(index: Dict[str, List[Dict[str, str]]],
                law_name: str, publish: str) -> Optional[str]:
    """按 (法名, 发布日期) 定位 markdown；日期缺失时退回文件名前缀匹配。"""
    candidates = list(index.get(law_name, []))
    if not candidates:
        # 兜底：法名是另一名称的前缀/后缀（例如「民法典 合同编」vs「民法典合同编」）
        for key, items in index.items():
            if law_name in key or key in law_name:
                candidates.extend(items)
    if not candidates:
        return None
    if publish:
        for c in candidates:
            if c["date"] == publish:
                return c["path"]
    # 没有精确日期：优先取不带日期的，否则取第一个
    for c in candidates:
        if not c["date"]:
            return c["path"]
    return candidates[0]["path"]


def build(repo: str, law_names: Iterable[str]) -> List[Dict]:
    db_path = os.path.join(repo, "db.sqlite3")
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"找不到 {db_path}。请先克隆法条库：\n"
            f"  git clone --depth 1 https://github.com/LawRefBook/Laws <dir>"
        )

    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    commit = _git_commit(repo)
    index = _index_markdown(repo)

    records: List[Dict] = []
    missing: List[str] = []

    for name in law_names:
        rows = cur.execute(
            "SELECT * FROM law WHERE name = ? ORDER BY valid_from", (name,)
        ).fetchall()
        if not rows:
            missing.append(name)
            continue

        for row in rows:
            md_path = _resolve_md(index, name, row["publish"] or "")
            if not md_path:
                missing.append(f"{name} (未找到 markdown)")
                continue

            with open(md_path, encoding="utf-8") as f:
                articles = parse_markdown_law(f.read())
            if not articles:
                missing.append(f"{name} (markdown 未解析出条目: {md_path})")
                continue

            for art in articles:
                records.append({
                    "law_id": row["id"],
                    "law_name": name,
                    "law_level": row["level"],
                    "filename": os.path.relpath(md_path, repo).replace(os.sep, "/"),
                    "publish": row["publish"] or "",
                    "valid_from": row["valid_from"] or "",
                    "valid_to": row["valid_to"] or OPEN_END,
                    "version": row["ver"],
                    "article_no": art["article_no"],
                    "article_index": int(art["article_index"]),
                    "chapter": art["chapter"],
                    "text": art["text"],
                    "source": f"LawRefBook/Laws@{commit[:12]}",
                })

    if missing:
        print("[warn] 以下法律未提取到：", file=sys.stderr)
        for m in missing:
            print("   -", m, file=sys.stderr)
    return records


def _walk_md(repo: str):
    for root, _dirs, files in os.walk(repo):
        if ".git" in root.split(os.sep):
            continue
        for fn in files:
            if fn.endswith(".md"):
                yield os.path.join(root, fn)


def main(argv=None) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    default_out = os.path.join(here, "data", "legal_corpus.jsonl")

    ap = argparse.ArgumentParser(description="构建法条语料 JSONL")
    ap.add_argument("--repo", default=os.environ.get("LAWS_REPO", ""),
                    help="LawRefBook/Laws 仓库本地路径（或设环境变量 LAWS_REPO）")
    ap.add_argument("--out", default=default_out, help="输出 JSONL 路径")
    args = ap.parse_args(argv)

    if not args.repo:
        print("错误：需要 --repo 或环境变量 LAWS_REPO 指向已克隆的法条库", file=sys.stderr)
        return 2

    records = build(args.repo, DEFAULT_LAW_NAMES)
    records.sort(key=lambda r: (r["law_name"], r["valid_from"], r["article_index"]))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_law: Dict[str, int] = {}
    for r in records:
        key = f"{r['law_name']} ({r['valid_from'] or '?'}~{r['valid_to']})"
        by_law[key] = by_law.get(key, 0) + 1

    expired = sum(1 for r in records if r["valid_to"] < OPEN_END)
    print(f"已写出 {len(records)} 条 → {args.out}")
    for k, v in sorted(by_law.items()):
        print(f"   {v:>5} 条  {k}")
    print(f"其中已失效法条 {expired} 条（用于「引用失效法条」测试）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
