"""Command-line entry point.

  python cli.py init [--reset]
  python cli.py ingest https://github.com/owner/name
  python cli.py search owner__name "where are failed jobs retried?"
  python cli.py ask    owner__name "where are failed jobs retried?"
  python cli.py chunks path/to/file.py        # preview how a file gets chunked
"""
import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(prog="repo-chat")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init"); s.add_argument("--reset", action="store_true")
    s = sub.add_parser("ingest"); s.add_argument("url")
    s = sub.add_parser("search"); s.add_argument("repo"); s.add_argument("question"); s.add_argument("-k", type=int)
    s = sub.add_parser("ask"); s.add_argument("repo"); s.add_argument("question"); s.add_argument("-k", type=int)
    s = sub.add_parser("chunks"); s.add_argument("file")
    a = p.parse_args()

    if a.cmd == "init":
        from app.db import init_schema
        init_schema(reset=a.reset)
        print("Schema ready.")

    elif a.cmd == "ingest":
        from app.db import init_schema
        from app.ingest import ingest_repo
        init_schema()
        print(json.dumps(ingest_repo(a.url), indent=2))

    elif a.cmd == "search":
        from app.retrieve import search
        for i, h in enumerate(search(a.repo, a.question, a.k), 1):
            print(f"[{i}] {h.score:.3f}  {h.citation}  {h.kind} {h.symbol or ''}")

    elif a.cmd == "ask":
        from app.answer import ask
        res = ask(a.repo, a.question, a.k)
        print(res["answer"], "\n\nSources:")
        for s in res["sources"]:
            print(f"  [{s['n']}] {s['citation']}  {s['kind']} {s['symbol'] or ''}")

    elif a.cmd == "chunks":
        from app.chunker import chunk_file
        path = Path(a.file)
        for c in chunk_file(path.as_posix(), path.read_bytes()):
            print(f"{c.start_line:>5}-{c.end_line:<5} {c.kind:<8} {c.symbol or ''}  ({len(c.content)} chars)")


if __name__ == "__main__":
    main()
