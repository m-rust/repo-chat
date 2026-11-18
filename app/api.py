from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from . import answer, ingest, retrieve
from .db import init_schema


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_schema()
    yield


app = FastAPI(title="Repo Chat", lifespan=lifespan)


class IngestRequest(BaseModel):
    url: str


class AskRequest(BaseModel):
    repo: str          # URL, owner__name slug, or repo id
    question: str
    k: int | None = None


@app.post("/repos")
def add_repo(req: IngestRequest):
    # Synchronous for now — this blocks for big repos. Moving it to a
    # background queue is a later stage.
    try:
        return ingest.ingest_repo(req.url)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/ask")
def ask(req: AskRequest):
    try:
        return answer.ask(req.repo, req.question, req.k)
    except LookupError as e:
        raise HTTPException(404, str(e))


@app.post("/search")
def search(req: AskRequest):
    """Retrieval only, no LLM — handy for debugging and for evaluation."""
    try:
        hits = retrieve.search(req.repo, req.question, req.k)
    except LookupError as e:
        raise HTTPException(404, str(e))
    return [{"citation": h.citation, "symbol": h.symbol, "kind": h.kind,
             "score": round(h.score, 3)} for h in hits]
