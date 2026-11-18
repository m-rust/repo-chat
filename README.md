# Repo Chat

Ask questions about any GitHub repository and get answers with `file:line` citations.

**Stage 1 (this version):** clone → chunk code by function/class with tree-sitter → embed locally → store in Postgres + pgvector → vector top-k → LLM answer with citations.

## Run it (Windows PowerShell)

```powershell
docker compose up -d
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env      # then set LLM_BASE_URL / LLM_API_KEY / LLM_MODEL
python cli.py init
python cli.py ingest https://github.com/rq/rq
python cli.py search rq__rq "how are failed jobs retried?"
python cli.py ask rq__rq "how are failed jobs retried?"
```

The first run downloads the embedding model (~130 MB) and the tree-sitter grammars.

API: `uvicorn app.api:app --reload`, then open http://localhost:8000/docs
- `POST /repos` `{"url": "..."}`: index a repo
- `POST /search` `{"repo": "...", "question": "..."}`: retrieval only (no LLM)
- `POST /ask` `{"repo": "...", "question": "..."}`: answer + sources

Debug chunking for a single file: `python cli.py chunks path\to\file.py`

## Layout

```
app/
  chunker.py   tree-sitter splitting: definitions → chunks, comments attached, line-window fallback
  ingest.py    clone, file filtering, hashing, embed, insert
  embedder.py  sentence-transformers (BAAI/bge-small-en-v1.5, 384-d)
  db.py        schema: repos → files → chunks (vector + tsvector columns)
  retrieve.py  vector search (cosine, HNSW index)
  answer.py    prompt with <source n=...> snippets, OpenAI-compatible LLM call
  api.py       FastAPI
cli.py
```

## Design decisions so far

| Decision | Why |
|---|---|
| Chunk by function/class, not fixed size | A chunk is a complete unit of meaning and can be cited precisely |
| Comments above a definition stay with it | Docstrings/JSDoc describe the code in plain language, which is what queries look like |
| Embed `File + Symbol` header + code | Code bodies often don't mention their own file or function name |
| Big classes split into methods, giant functions into line windows | Keeps chunks under ~4000 chars without cutting mid-function where avoidable |
| Postgres + pgvector, not a separate vector DB | One store for vectors, full-text and metadata; filters and joins in SQL |
| `tsv` column and `content_hash` already in the schema | Ready for hybrid search and incremental re-indexing |
| Local embeddings, any OpenAI-compatible LLM | No embedding API cost; swap between OpenAI, Gemini and Ollama by editing `.env` |

## Next stages

1. **Evaluation set:** 30 questions → expected file; measure recall@5 *before* changing anything
2. **Hybrid retrieval:** full-text + vector, merged with reciprocal rank fusion
3. **Incremental re-indexing:** skip unchanged files using `content_hash`
4. **Background ingestion:** Redis queue, so `POST /repos` returns immediately
