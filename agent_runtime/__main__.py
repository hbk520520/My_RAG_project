"""Explicit operations; all default files live under the D-drive project root."""
import argparse
import json


def main(argv=None):
    parser = argparse.ArgumentParser(description="Durable single-host Legal Agent")
    parser.add_argument("--db", default=".runtime/runs.sqlite")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("query")
    run.add_argument("--project-id", default="legal-rag")
    run.add_argument("--kafka", action="store_true")
    for name in ("status", "events", "cancel", "checkpoints", "retry"):
        sub.add_parser(name).add_argument("run_id")
    resume = sub.add_parser("resume")
    resume.add_argument("run_id")
    resume.add_argument("answer")
    rewind = sub.add_parser("rewind")
    rewind.add_argument("run_id")
    rewind.add_argument("checkpoint_id")
    sub.add_parser("recover")
    backup = sub.add_parser("backup")
    backup.add_argument("destination")
    restore = sub.add_parser("restore")
    restore.add_argument("source")
    restore.add_argument("destination")
    serve = sub.add_parser("serve")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--kafka", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "restore":
        from .backup import restore_bundle
        print(json.dumps({"database": str(restore_bundle(args.source, args.destination))}))
        return 0
    from .store import RunStore
    from .service import AgentService
    service = AgentService(RunStore(args.db))
    if args.command == "run":
        row = service.submit({"query": args.query, "project_id": args.project_id}, transport="kafka" if args.kafka else "local")
        result = row if args.kafka else service.execute(row["run_id"])
    elif args.command == "resume":
        row = service.resume(args.run_id, args.answer)
        result = service.execute(row["run_id"])
    elif args.command == "rewind":
        result = service.rewind(args.run_id, args.checkpoint_id)
    elif args.command == "status":
        result = service.store.get(args.run_id)
    elif args.command == "cancel":
        result = service.store.cancel(args.run_id)
    elif args.command == "events":
        result = service.store.events(args.run_id)
    elif args.command == "checkpoints":
        result = service.checkpoints(args.run_id)
    elif args.command == "retry":
        service.store.requeue(args.run_id, retry=True)
        result = service.execute(args.run_id)
    elif args.command == "recover":
        result = service.recover()
    elif args.command == "backup":
        result = {"backup": str(service.backup(args.destination))}
    else:
        import uvicorn
        from .api import create_app
        uvicorn.run(create_app(service, execute_local=not args.kafka), host=args.host, port=args.port)
        return 0
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
