"""Clone a repo, chunk every indexable file, embed, and store in Postgres."""
from __future__ import annotations

import hashlib
import re
import subprocess
import time
from pathlib import Path

from . import config, embedder
from .chunker import Chunk, chunk_file, detect_language, is_indexable
from .db import connect_vector


def repo_name(url: str) -> str:
    """https://github.com/owner/name(.git) -> owner__name"""
    m = re.search(r"github\.com[/:]([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if not m:
        raise ValueError(f"Not a GitHub repo URL: {url}")
    return f"{m.group(1)}__{m.group(2)}"


def clone_or_pull(url: str) -> tuple[Path, str]:
    dest = config.REPOS_DIR / repo_name(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest / ".git").exists():
        subprocess.run(["git", "-C", str(dest), "pull", "--ff-only", "-q"], check=True)
    else:
        subprocess.run(["git", "clone", "--depth", "1", "-q", url, str(dest)], check=True)
    sha = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                         check=True, capture_output=True, text=True).stdout.strip()
    return dest, sha


def iter_files(root: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in config.SKIP_DIRS for part in rel.parts):
            continue
        if p.name in config.SKIP_FILES or p.name.endswith((".min.js", ".min.css", ".map")):
            continue
        if not is_indexable(p) or p.stat().st_size > config.MAX_FILE_BYTES:
            continue
        yield p, rel.as_posix()


def ingest_local(root: Path, url: str, sha: str | None = None) -> dict:
    """Index a directory that's already on disk. Re-indexing replaces the old data."""
    t0 = time.time()
    files: list[tuple[str, str | None, str, list[Chunk]]] = []
    for p, rel in iter_files(root):
        src = p.read_bytes()
        if b"\x00" in src[:1024]:  # binary
            continue
        files.append((rel, detect_language(p), hashlib.sha256(src).hexdigest(), chunk_file(rel, src)))

    all_chunks = [c for *_, chunks in files for c in chunks]
    if not all_chunks:
        raise RuntimeError("No indexable files found.")
    vectors = embedder.embed_documents([c.embed_text() for c in all_chunks])

    with connect_vector() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM repos WHERE url = %s", (url,))  # cascades to files/chunks
        cur.execute(
            "INSERT INTO repos (url, name, commit_sha) VALUES (%s, %s, %s) RETURNING id",
            (url, repo_name(url) if "github.com" in url else root.name, sha),
        )
        repo_id = cur.fetchone()[0]

        vi = 0
        for rel, lang, h, chunks in files:
            cur.execute(
                "INSERT INTO files (repo_id, path, language, content_hash) VALUES (%s,%s,%s,%s) RETURNING id",
                (repo_id, rel, lang, h),
            )
            file_id = cur.fetchone()[0]
            rows = []
            for c in chunks:
                rows.append((repo_id, file_id, c.path, c.symbol, c.kind,
                             c.start_line, c.end_line, c.content, vectors[vi]))
                vi += 1
            if rows:
                cur.executemany(
                    """INSERT INTO chunks (repo_id, file_id, path, symbol, kind, start_line,
                                           end_line, content, embedding)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                    rows,
                )
        conn.commit()

    return {"repo_id": repo_id, "files": len(files), "chunks": len(all_chunks),
            "seconds": round(time.time() - t0, 1)}


def ingest_repo(url: str) -> dict:
    root, sha = clone_or_pull(url)
    return ingest_local(root, url, sha) | {"commit": sha[:10]}
