import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://repochat:repochat@localhost:5433/repochat")

EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(os.getenv("EMBED_DIM", "384"))

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

REPOS_DIR = Path(os.getenv("REPOS_DIR", "./data/repos"))
TOP_K = int(os.getenv("TOP_K", "8"))

# Chunking limits (characters). A definition bigger than MAX_CHUNK_CHARS is split
# into its children (e.g. a class into its methods); below MIN_CHUNK_CHARS small
# neighbouring pieces are merged so we don't index one-line fragments.
MAX_CHUNK_CHARS = 4000
MIN_CHUNK_CHARS = 200

# Files we never index.
SKIP_DIRS = {
    ".git", "node_modules", "vendor", "dist", "build", "out", "target", "__pycache__",
    ".venv", "venv", ".next", ".idea", ".vscode", "coverage", ".mypy_cache", ".pytest_cache",
}
SKIP_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock", "Cargo.lock",
    "composer.lock", "Gemfile.lock", "go.sum",
}
MAX_FILE_BYTES = 300_000  # skip huge/generated files
