"""Retrieval. Stage 1 is vector-only; hybrid (vector + full-text, fused) comes next."""
from __future__ import annotations

from dataclasses import dataclass

from . import config, embedder
from .db import connect_vector


@dataclass
class Hit:
    chunk_id: int
    path: str
    symbol: str | None
    kind: str
    start_line: int
    end_line: int
    content: str
    score: float

    @property
    def citation(self) -> str:
        return f"{self.path}:{self.start_line}-{self.end_line}"


def get_repo_id(conn, repo: str) -> int:
    """Accepts a repo id, full URL, or the owner__name slug."""
    row = conn.execute(
        "SELECT id FROM repos WHERE url = %s OR name = %s OR id::text = %s",
        (repo, repo, repo),
    ).fetchone()
    if row is None:
        raise LookupError(f"Repo not indexed: {repo}")
    return row[0]


def vector_search(repo: str, query: str, k: int | None = None) -> list[Hit]:
    k = k or config.TOP_K
    qvec = embedder.embed_query(query)
    with connect_vector() as conn:
        repo_id = get_repo_id(conn, repo)
        rows = conn.execute(
            """SELECT id, path, symbol, kind, start_line, end_line, content,
                      1 - (embedding <=> %s) AS score
               FROM chunks
               WHERE repo_id = %s
               ORDER BY embedding <=> %s
               LIMIT %s""",
            (qvec, repo_id, qvec, k),
        ).fetchall()
    return [Hit(*r) for r in rows]


def search(repo: str, query: str, k: int | None = None) -> list[Hit]:
    return vector_search(repo, query, k)
