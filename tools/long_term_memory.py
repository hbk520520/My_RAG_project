"""Inspect local project memory or import a PageIndex JSON tree plus its source PDF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv=None):
    from config_loader import cfg
    from dataset.long_term_memory import MemoryStore

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path(cfg.get("long_term_memory", "path")))
    parser.add_argument("--project-id", default=cfg.get("long_term_memory", "project_id"))
    parser.add_argument("--space-id", default=cfg.get("long_term_memory", "embedding_space_id"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inspect")
    sub.add_parser("documents")
    structure = sub.add_parser("structure")
    structure.add_argument("--document-id", required=True)
    structure.add_argument("--version")
    pages = sub.add_parser("pages")
    pages.add_argument("--document-id", required=True)
    pages.add_argument("--version")
    pages.add_argument("--pages", nargs="+", required=True, type=int)
    cost = sub.add_parser("cost")
    cost.add_argument("--document-id", required=True)
    importer = sub.add_parser("import-pageindex")
    importer.add_argument("--tree", type=Path, help="Optional PageIndex export; otherwise detect PDF structure locally")
    importer.add_argument("--pdf", required=True, type=Path)
    importer.add_argument("--document-id", required=True)
    importer.add_argument("--generate-summaries", action="store_true", help="Generate missing summaries with configured LLM")
    importer.add_argument("--stub-encoder", action="store_true",
                          help="Offline demo only; use an isolated path and --space-id stub-v1")
    args = parser.parse_args(argv)
    path = args.path if args.path.is_absolute() else ROOT / args.path
    if args.command == "inspect":
        if not path.is_file():
            raise FileNotFoundError(path)
        revision, snapshot = MemoryStore(path, args.project_id, read_only=True).load()
        print(json.dumps({"project_id": args.project_id, "revision": revision,
                          "nodes": len(snapshot["vertices"]) if snapshot else 0,
                          "summaries": sum(v["type"] == "Summary" for v in snapshot["vertices"]) if snapshot else 0},
                         ensure_ascii=False, indent=2))
        return 0
    if args.command in ("documents", "structure", "pages", "cost"):
        from dataset.document_store import DocumentReader
        reader = DocumentReader.from_file(path, args.project_id)
        if args.command == "documents":
            result = reader.list_documents()
        elif args.command == "structure":
            result = reader.read_structure(args.document_id, version=args.version)
        elif args.command == "cost":
            from dataset.document_index import diagnose_cost
            result = diagnose_cost(reader.read_structure(args.document_id))
        else:
            result = reader.read_pages(args.document_id, args.pages, version=args.version)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    from dataset.graph import LegalDenseGraphBuilder, StubEncoder
    from dataset.page_index_memory import import_tree
    from dataset.document_index import read_pdf_structure
    if args.stub_encoder and not args.space_id.startswith("stub-"):
        raise ValueError("Offline encoder requires an explicit stub-* embedding space")
    engine = LegalDenseGraphBuilder.from_config(load_memory=False)
    engine.enable_memory(path, args.project_id, args.space_id)
    if args.stub_encoder:
        engine.set_encoder(StubEncoder(engine.dim))
    pages, detected = read_pdf_structure(args.pdf)
    tree = json.loads(args.tree.read_text(encoding="utf-8")) if args.tree else detected
    summary_fn = None
    if args.generate_summaries:
        from causal_evidence.pipeline import causal_completion
        def summary_fn(texts):
            response = causal_completion([{"role": "system", "content":
                'Return JSON {"summary": "..."}. Summarize only the supplied text for document navigation. '
                'Do not add facts or causal claims; this summary is not source evidence.'},
                {"role": "user", "content": json.dumps(texts, ensure_ascii=False)}])
            return json.loads(response)["summary"]
    result = import_tree(engine, pages, tree, source_file=args.pdf.name,
                         document_id=args.document_id, summary_fn=summary_fn)
    result["toc_source"] = tree.get("toc_source", "imported") if isinstance(tree, dict) else "imported"
    result["diagnostics"] = tree.get("diagnostics", []) if isinstance(tree, dict) else []
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
