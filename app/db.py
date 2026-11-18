import psycopg
from pgvector.psycopg import register_vector

from . import config

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS repos (
    id          SERIAL PRIMARY KEY,
    url         TEXT UNIQUE NOT NULL,
    name        TEXT NOT NULL,
    commit_sha  TEXT,
    indexed_at  TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS files (
    id            SERIAL PRIMARY KEY,
    repo_id       INT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    path          TEXT NOT NULL,
    language      TEXT,
    content_hash  TEXT NOT NULL,          -- used later for incremental re-indexing
    UNIQUE (repo_id, path)
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    repo_id     INT NOT NULL REFERENCES repos(id) ON DELETE CASCADE,
    file_id     INT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
    path        TEXT NOT NULL,
    symbol      TEXT,                     -- e.g. "QueueWorker.retry"
    kind        TEXT NOT NULL,            -- function | class | method | module | window
    start_line  INT NOT NULL,
    end_line    INT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector({dim}) NOT NULL,
    -- Full-text column for keyword search (hybrid retrieval comes in the next stage).
    tsv         tsvector GENERATED ALWAYS AS (
                    to_tsvector('simple', coalesce(symbol, '') || ' ' || path || ' ' || content)
                ) STORED
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_repo_idx ON chunks (repo_id);
"""


def connect() -> psycopg.Connection:
    conn = psycopg.connect(config.DATABASE_URL, autocommit=False)
    return conn


def init_schema(reset: bool = False) -> None:
    with psycopg.connect(config.DATABASE_URL, autocommit=True) as conn:
        if reset:
            conn.execute("DROP TABLE IF EXISTS chunks, files, repos CASCADE")
        conn.execute(SCHEMA.format(dim=config.EMBED_DIM))


def connect_vector() -> psycopg.Connection:
    conn = connect()
    register_vector(conn)
    return conn
