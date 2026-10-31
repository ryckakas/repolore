import hashlib
import re
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sqlite_vec

from repolore.chunking import Chunk
from repolore.embedding import Vectors

SCHEMA = """
create table meta (key text primary key, value text not null);
create table chunks (
    rowid integer primary key,
    id text not null unique,
    url text not null,
    repo text not null,
    kind text not null,
    visibility text not null,
    title text not null,
    section text,
    author text,
    created_at text,
    state text,
    ai_assisted integer not null,
    header text not null,
    text text not null
);
create virtual table chunks_vec using vec0(
    embedding float[{dim}] distance_metric=cosine,
    repo text,
    kind text,
    author text
);
create virtual table chunks_fts using fts5(
    header, text, content='chunks', content_rowid='rowid', tokenize="unicode61 tokenchars '_'"
);
"""
TABLES = ("meta", "chunks", "chunks_vec", "chunks_fts")
NO_AUTHOR = ""


class StoreError(Exception):
    pass


@dataclass(frozen=True)
class Filters:
    repo: str | None = None
    kind: str | None = None
    author: str | None = None


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    url: str
    repo: str
    kind: str
    title: str
    section: str | None
    author: str | None
    created_at: str | None
    header: str
    text: str
    score: float


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    return _with_vectors(sqlite3.connect(path))


def connect_read_only(path: Path) -> sqlite3.Connection:
    return _with_vectors(sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True))


def _with_vectors(db: sqlite3.Connection) -> sqlite3.Connection:
    # Some Python builds (python.org's macOS installer, GitHub's macOS runners) leave SQLite extension loading out.
    if not hasattr(db, "enable_load_extension"):
        db.close()
        raise StoreError(
            "This Python's sqlite3 can't load extensions, which the vector index needs. "
            "Run repolore on a uv-managed Python: `uv python install 3.12`, then `uv sync --managed-python`."
        )
    db.enable_load_extension(True)
    sqlite_vec.load(db)
    db.enable_load_extension(False)
    return db


def cache_key(model_id: str, text: str) -> str:
    return hashlib.sha256(f"{model_id}\0{text}".encode()).hexdigest()


class EmbeddingCache:
    """Vectors keyed by model and text, shared by every index so unchanged chunks are never embedded twice."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db
        db.execute("create table if not exists embeddings (key text primary key, vector blob not null)")

    def get(self, keys: Iterable[str]) -> dict[str, Vectors]:
        found: dict[str, Vectors] = {}
        for key in keys:
            row = self._db.execute("select vector from embeddings where key = ?", (key,)).fetchone()
            if row:
                found[key] = np.frombuffer(row[0], dtype=np.float32)
        return found

    def put(self, keys: Sequence[str], vectors: Vectors) -> None:
        with self._db:
            self._db.executemany(
                "insert or replace into embeddings (key, vector) values (?, ?)",
                [(key, vector.astype(np.float32).tobytes()) for key, vector in zip(keys, vectors, strict=True)],
            )


def write_index(
    db: sqlite3.Connection,
    chunks: Sequence[Chunk],
    vectors: Vectors,
    model_id: str,
    dim: int,
    settings: Mapping[str, str] | None = None,
) -> None:
    with db:
        for table in TABLES:
            db.execute(f"drop table if exists {table}")
        db.executescript(SCHEMA.format(dim=dim))
        meta = {**(settings or {}), "model_id": model_id, "dim": str(dim)}
        db.executemany("insert into meta (key, value) values (?, ?)", meta.items())
        for rowid, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True), start=1):
            _insert_chunk(db, rowid, chunk, vector)
        db.execute("insert into chunks_fts (chunks_fts) values ('rebuild')")


def _insert_chunk(db: sqlite3.Connection, rowid: int, chunk: Chunk, vector: Vectors) -> None:
    document = chunk.document
    created_at = document.created_at.isoformat() if document.created_at else None
    db.execute(
        "insert into chunks values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            rowid,
            chunk.id,
            document.url,
            document.repo,
            document.kind,
            document.visibility,
            document.title,
            chunk.section,
            document.author,
            created_at,
            document.state,
            document.ai_assisted,
            chunk.header,
            chunk.text,
        ),
    )
    db.execute(
        "insert into chunks_vec (rowid, embedding, repo, kind, author) values (?, ?, ?, ?, ?)",
        (rowid, vector.astype(np.float32).tobytes(), document.repo, document.kind, document.author or NO_AUTHOR),
    )


def index_model(db: sqlite3.Connection) -> tuple[str, int]:
    meta = index_settings(db)
    return meta["model_id"], int(meta["dim"])


def changed_settings(db: sqlite3.Connection, settings: Mapping[str, str]) -> dict[str, tuple[str, str]]:
    try:
        previous = index_settings(db)
    except StoreError:
        return {}
    return {key: (previous[key], value) for key, value in settings.items() if previous.get(key, value) != value}


def index_settings(db: sqlite3.Connection) -> dict[str, str]:
    try:
        return dict(db.execute("select key, value from meta").fetchall())
    except sqlite3.OperationalError as error:
        raise StoreError("The index is empty; run `repolore build` first.") from error


def nearest(db: sqlite3.Connection, vector: Vectors, k: int, filters: Filters) -> list[Hit]:
    conditions = _conditions(filters)
    where = "".join(f" and {column} = :{column}" for column in conditions)
    rows = db.execute(
        f"""
        with hits as (
            select rowid, distance from chunks_vec
            where embedding match :vector and k = :k{where}
        )
        select c.id, c.url, c.repo, c.kind, c.title, c.section, c.author, c.created_at, c.header, c.text,
            hits.distance
        from hits join chunks c on c.rowid = hits.rowid
        order by hits.distance
        """,
        {"vector": vector.astype(np.float32).tobytes(), "k": k, **conditions},
    ).fetchall()
    return [Hit(*row[:-1], score=1 - row[-1]) for row in rows]


def keyword_matches(db: sqlite3.Connection, query: str, k: int, filters: Filters) -> list[Hit]:
    words = re.findall(r"\w+", query)
    if not words:
        return []
    conditions = _conditions(filters)
    where = "".join(f" and c.{column} = :{column}" for column in conditions)
    # Each word is quoted so FTS5 never reads the user's text as query syntax; OR keeps questions from needing
    # every word to match, and BM25 ranks the chunks matching the rarer words first.
    terms = " OR ".join(f'"{word}"' for word in words)
    rows = db.execute(
        f"""
        select c.id, c.url, c.repo, c.kind, c.title, c.section, c.author, c.created_at, c.header, c.text,
            bm25(chunks_fts)
        from chunks_fts join chunks c on c.rowid = chunks_fts.rowid
        where chunks_fts match :terms{where}
        order by bm25(chunks_fts)
        limit :k
        """,
        {"terms": terms, "k": k, **conditions},
    ).fetchall()
    return [Hit(*row[:-1], score=-row[-1]) for row in rows]


def _conditions(filters: Filters) -> dict[str, str]:
    columns = {"repo": filters.repo, "kind": filters.kind, "author": filters.author}
    return {column: value for column, value in columns.items() if value is not None}


def indexed_urls(db: sqlite3.Connection) -> set[str]:
    return {url for (url,) in db.execute("select distinct url from chunks")}


def repo_counts(db: sqlite3.Connection) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for repo, kind, count in db.execute(
        "select repo, kind, count(*) from chunks group by repo, kind order by repo, kind"
    ):
        counts.setdefault(repo, {})[kind] = count
    return counts
