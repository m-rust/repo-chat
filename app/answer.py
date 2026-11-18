"""Build the prompt from retrieved chunks and ask the LLM, with numbered citations."""
from __future__ import annotations

from openai import OpenAI

from . import config
from .retrieve import Hit, search

SYSTEM_PROMPT = """You answer questions about a code repository using ONLY the source \
snippets provided in <source n="..."> tags. Rules:
- Cite every claim with the snippet's n in square brackets, e.g. [2] or [1][3].
- Name the specific files, functions and classes involved.
- If the snippets don't contain the answer, say so plainly and say what you'd look for \
instead. Do not guess about code you weren't shown.
- Be concise. Use short code excerpts only when they help."""


def _client() -> OpenAI:
    return OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY or "none")


def build_context(hits: list[Hit]) -> str:
    parts = []
    for i, h in enumerate(hits, 1):
        # XML-style tags rather than ``` fences: markdown docs in the repo contain
        # their own fences, which would break the snippet boundaries.
        sym = f' symbol="{h.symbol}"' if h.symbol else ""
        parts.append(f'<source n="{i}" location="{h.citation}" kind="{h.kind}"{sym}>\n'
                     f"{h.content}\n</source>")
    return "\n\n".join(parts)


def ask(repo: str, question: str, k: int | None = None) -> dict:
    hits = search(repo, question, k)
    if not hits:
        return {"answer": "Nothing indexed for this repo yet.", "sources": []}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n\n{build_context(hits)}\n\nQuestion: {question}"},
    ]
    resp = _client().chat.completions.create(model=config.LLM_MODEL, messages=messages, temperature=0)
    return {
        "answer": resp.choices[0].message.content,
        "sources": [
            {"n": i, "citation": h.citation, "symbol": h.symbol, "kind": h.kind,
             "score": round(h.score, 3)}
            for i, h in enumerate(hits, 1)
        ],
    }
