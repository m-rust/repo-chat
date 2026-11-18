"""Split source files into retrieval chunks.

Strategy
--------
1. Code in a supported language is parsed with tree-sitter and split along
   definitions: each function / class / method becomes its own chunk.
2. Comments directly above a definition are attached to it, so docstrings and
   JSDoc travel with the code they describe.
3. Code between definitions (imports, constants, top-level calls) is grouped
   into "module" chunks.
4. A definition larger than MAX_CHUNK_CHARS is split into its children
   (a class into its methods). If it has no children (a giant function),
   it falls back to line windows.
5. Anything we can't parse (unknown language, docs, config) is split into
   overlapping line windows.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tree_sitter_language_pack import get_parser

from . import config

EXT_TO_LANG = {
    ".py": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx",
    ".go": "go",
    ".java": "java",
    ".php": "php",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".hpp": "cpp", ".cxx": "cpp",
    ".rs": "rust",
    ".rb": "ruby",
}

# Text files we index with line windows (no parser).
TEXT_EXTS = {
    ".md", ".txt", ".rst", ".yaml", ".yml", ".toml", ".json", ".sql", ".sh",
    ".ini", ".cfg", ".env.example", ".html", ".css", ".scss", ".vue", ".svelte",
}
TEXT_NAMES = {"Dockerfile", "Makefile", "Procfile"}

# Node types that count as a "definition" worth its own chunk.
DEFINITION_TYPES = {
    # python
    "function_definition", "class_definition", "decorated_definition",
    # js / ts
    "function_declaration", "generator_function_declaration", "class_declaration",
    "abstract_class_declaration", "method_definition", "interface_declaration",
    "type_alias_declaration", "enum_declaration",
    # go
    "method_declaration", "type_declaration",
    # java
    "constructor_declaration",
    # php
    "trait_declaration",
    # c / cpp
    "class_specifier", "struct_specifier",
    # rust
    "function_item", "impl_item", "struct_item", "enum_item", "trait_item", "mod_item",
    # ruby
    "method", "singleton_method", "class", "module",
}

FUNCTION_VALUE_TYPES = {"arrow_function", "function_expression", "function"}
COMMENT_TYPES = {"comment", "line_comment", "block_comment"}


@dataclass
class Chunk:
    path: str
    symbol: str | None
    kind: str          # function | class | method | module | window
    start_line: int    # 1-based, inclusive
    end_line: int      # 1-based, inclusive
    content: str

    def embed_text(self) -> str:
        """Text we embed: a short header gives the model the file and symbol name,
        which the raw code body often doesn't mention."""
        header = f"File: {self.path}"
        if self.symbol:
            header += f"\nSymbol: {self.symbol} ({self.kind})"
        return f"{header}\n\n{self.content}"


def detect_language(path: Path) -> str | None:
    return EXT_TO_LANG.get(path.suffix.lower())


def is_indexable(path: Path) -> bool:
    return (
        detect_language(path) is not None
        or path.suffix.lower() in TEXT_EXTS
        or path.name in TEXT_NAMES
    )


# ---------------------------------------------------------------- helpers

def _text(node, src: bytes) -> str:
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _unwrap(node):
    """export_statement / decorated_definition wrap the real declaration."""
    if node.type == "export_statement":
        inner = node.child_by_field_name("declaration")
        return inner or node
    if node.type == "decorated_definition":
        inner = node.child_by_field_name("definition")
        return inner or node
    return node


def _is_definition(node) -> bool:
    n = _unwrap(node)
    if n.type in DEFINITION_TYPES:
        return True
    # const handler = () => {...}  /  const handler = function () {...}
    if n.type in ("lexical_declaration", "variable_declaration"):
        for decl in n.named_children:
            if decl.type == "variable_declarator":
                value = decl.child_by_field_name("value")
                if value is not None and value.type in FUNCTION_VALUE_TYPES:
                    return True
    return False


def _name(node, src: bytes) -> str | None:
    n = _unwrap(node)
    named = n.child_by_field_name("name")
    if named is not None:
        return _text(named, src)
    if n.type in ("lexical_declaration", "variable_declaration"):
        for decl in n.named_children:
            if decl.type == "variable_declarator":
                nm = decl.child_by_field_name("name")
                if nm is not None:
                    return _text(nm, src)
    if n.type == "impl_item":  # rust: impl Foo / impl Trait for Foo
        t = n.child_by_field_name("type")
        if t is not None:
            return _text(t, src)
    if n.type == "type_declaration":  # go: type Foo struct
        for c in n.named_children:
            nm = c.child_by_field_name("name")
            if nm is not None:
                return _text(nm, src)
    # c / cpp: dig through declarators to the identifier
    decl = n.child_by_field_name("declarator")
    while decl is not None:
        if decl.type in ("identifier", "field_identifier", "qualified_identifier",
                         "destructor_name", "operator_name"):
            return _text(decl, src)
        decl = decl.child_by_field_name("declarator")
    return None


def _kind(node, parent_symbol: str | None) -> str:
    t = _unwrap(node).type
    if any(k in t for k in ("class", "struct", "interface", "trait", "impl", "enum", "module", "mod_item", "type_")):
        return "class"
    return "method" if parent_symbol else "function"


def _body(node):
    n = _unwrap(node)
    body = n.child_by_field_name("body")
    return body if body is not None else n


def _lines(lines: list[str], start: int, end: int) -> str:
    """start/end are 0-based row indices, inclusive."""
    return "\n".join(lines[start:end + 1])


def window_chunks(path: str, lines: list[str], start: int = 0, end: int | None = None,
                  symbol: str | None = None, size: int = 60, overlap: int = 10) -> list[Chunk]:
    """Overlapping fixed-size line windows over rows [start, end]."""
    end = len(lines) - 1 if end is None else end
    out: list[Chunk] = []
    i = start
    while i <= end:
        # Grow the window up to `size` lines, but stop early at MAX_CHUNK_CHARS
        # (markdown/minified lines can be very long).
        j, chars = i, len(lines[i])
        while j < end and j - i + 1 < size and chars + len(lines[j + 1]) + 1 <= config.MAX_CHUNK_CHARS:
            j += 1
            chars += len(lines[j]) + 1
        content = _lines(lines, i, j)
        if content.strip():
            out.append(Chunk(path, symbol, "window", i + 1, j + 1, content))
        if j == end:
            break
        i = max(j + 1 - overlap, i + 1)
    return out


# ---------------------------------------------------------------- main logic

class _Splitter:
    def __init__(self, path: str, src: bytes):
        self.path = path
        self.src = src
        self.lines = src.decode("utf-8", errors="replace").split("\n")
        self.chunks: list[Chunk] = []

    def _emit_range(self, start: int, end: int, symbol: str | None, kind: str) -> None:
        content = _lines(self.lines, start, end)
        if not content.strip():
            return
        if len(content) > config.MAX_CHUNK_CHARS:
            for w in window_chunks(self.path, self.lines, start, end, symbol=symbol):
                w.kind = kind if kind != "module" else "window"
                self.chunks.append(w)
        else:
            self.chunks.append(Chunk(self.path, symbol, kind, start + 1, end + 1, content))

    def _flush_loose(self, loose: list, symbol: str | None, kind: str) -> None:
        """Group consecutive non-definition nodes into chunks of up to MAX_CHUNK_CHARS."""
        if not loose:
            return
        group_start = loose[0].start_point[0]
        group_end = loose[0].end_point[0]
        for node in loose[1:]:
            candidate = _lines(self.lines, group_start, node.end_point[0])
            if len(candidate) > config.MAX_CHUNK_CHARS:
                self._emit_range(group_start, group_end, symbol, kind)
                group_start = node.start_point[0]
            group_end = node.end_point[0]
        self._emit_range(group_start, group_end, symbol, kind)
        loose.clear()

    def split(self, container, parent_symbol: str | None = None) -> None:
        loose: list = []
        loose_kind = "class" if parent_symbol else "module"
        for child in container.named_children:
            if not _is_definition(child):
                loose.append(child)
                continue

            # Pull comments that sit directly above this definition into it.
            start_row = child.start_point[0]
            leading: list = []
            while loose and loose[-1].type in COMMENT_TYPES and loose[-1].end_point[0] >= start_row - 1:
                leading.insert(0, loose.pop())
                start_row = leading[0].start_point[0]
            self._flush_loose(loose, parent_symbol, loose_kind)

            name = _name(child, self.src)
            symbol = f"{parent_symbol}.{name}" if parent_symbol and name else (name or parent_symbol)
            end_row = child.end_point[0]
            size = len(_lines(self.lines, start_row, end_row))

            if size <= config.MAX_CHUNK_CHARS:
                self._emit_range(start_row, end_row, symbol, _kind(child, parent_symbol))
                continue

            # Too big: descend into its body if it contains definitions.
            body = _body(child)
            if any(_is_definition(c) for c in body.named_children):
                # Header (signature + anything before the body) as its own small chunk.
                body_start = body.start_point[0]
                if body_start > start_row:
                    self._emit_range(start_row, body_start - 1, symbol, "class")
                self.split(body, parent_symbol=symbol)
            else:
                for w in window_chunks(self.path, self.lines, start_row, end_row, symbol=symbol):
                    w.kind = _kind(child, parent_symbol)
                    self.chunks.append(w)

        self._flush_loose(loose, parent_symbol, loose_kind)


def chunk_file(path: str, src: bytes) -> list[Chunk]:
    lang = detect_language(Path(path))
    lines = src.decode("utf-8", errors="replace").split("\n")
    if lang is None:
        return window_chunks(path, lines)
    try:
        tree = get_parser(lang).parse(src)
    except Exception:
        return window_chunks(path, lines)

    splitter = _Splitter(path, src)
    splitter.split(tree.root_node)
    chunks = splitter.chunks
    return chunks or window_chunks(path, lines)
