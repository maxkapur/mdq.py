import sqlite3
import sys
from argparse import ArgumentParser
from hashlib import file_digest
from pathlib import Path
from typing import NamedTuple

import numpy as np
import sqlite_vec
from fastembed import TextEmbedding
from platformdirs import user_cache_path
from rich.console import Console

query_prefix = "query: "

console = Console(stderr=True)
conn_path = user_cache_path("mdq", ensure_exists=True) / "cache.db"

embed_model = TextEmbedding("BAAI/bge-small-en-v1.5")


def main():
    options = get_options()

    conn = initialize_db()

    paths = flat_paths(options)

    updated_paths = updated(paths, conn)
    embed_docs = [path.read_text() for path, _, _ in updated_paths]

    with console.status(f"Embed {len(embed_docs)} documents"):
        embeddings = [e.astype(np.float32) for e in embed_model.embed(embed_docs)]

    with conn:
        # Insert any new paths
        conn.executemany(
            "INSERT OR IGNORE INTO document(path, digest, mtime) VALUES (?, ?, ?)",
            [(p.path_str, p.digest, p.mtime) for p in updated_paths],
        )
        # Update all path digests
        conn.executemany(
            "UPDATE document SET digest=?, mtime=? WHERE path=?",
            [(p.digest, p.mtime, p.path_str) for p in updated_paths],
        )

        conn.executemany(
            "INSERT INTO embedding(digest, vec) VALUES (?, ?)",
            [(p.digest, vec) for p, vec in zip(updated_paths, embeddings)],
        )

    query_embed = np.array(*embed_model.embed([options.query]), dtype=np.float32)

    question_marks = ",".join("?" * len(paths))
    print(
        conn.execute(
            f"""
            SELECT path, distance
                FROM document
                JOIN embedding
                ON document.digest = embedding.digest
                WHERE document.path IN ({question_marks})
                AND embedding.vec MATCH ?
                AND k = ?
                ORDER BY distance
            """,
            [str(p.absolute()) for p in paths] + [query_embed, options.n_matches],
        ).fetchall()
    )


def get_options(args=None):
    parser = ArgumentParser()
    parser.add_argument(
        "-p",
        "--paths",
        default=[Path()],
        nargs="+",
        type=Path,
    )
    parser.add_argument("-q", "--query", default=None)
    parser.add_argument(
        "-e", "--extensions", nargs="*", default=["md", "markdown", "txt"]
    )
    parser.add_argument("-k", "-n", "--n-matches", default=4, type=int)

    options = parser.parse_args(args)
    query_str = options.query or sys.stdin.readline()
    options.query = query_prefix + query_str.strip()

    options.extensions = set("." + e.lstrip(".") for e in options.extensions)

    return options


def initialize_db():
    conn = sqlite3.connect(str(conn_path))

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    with conn:
        conn.execute("PRAGMA strict = ON")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS document(
                path TEXT UNIQUE,
                digest TEXT,
                mtime FLOAT
            )
        """)

        embedding_size = embed_model.get_embedding_size(embed_model.model_name)

        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS embedding USING vec0(
                digest TEXT,
                vec FLOAT[{embedding_size}]
            )
        """)

    return conn


def flat_paths(options):
    """List giving each file path implied by the paths given in options."""

    def inner():
        for path in options.paths:
            if path.is_file():
                yield path
            elif path.is_dir():
                for ext in options.extensions:
                    yield from path.glob(f"**/*{ext}")
            else:
                raise ValueError(path)

    return list(inner())


class DocumentMetadata(NamedTuple):
    path: Path
    digest: str
    mtime: float

    @property
    def path_str(self):
        return str(self.path.absolute())


def updated(paths, conn):
    """List of just the file paths that have been updated."""

    def inner():
        for path in paths:
            new_mtime = path.stat().st_mtime

            res = conn.execute(
                "SELECT path, digest, mtime FROM document WHERE path=?",
                [str(path.absolute())],
            )
            if fetched := res.fetchone():
                _, _, old_mtime = fetched
                if new_mtime == old_mtime:
                    continue
            yield DocumentMetadata(path, digest(path), new_mtime)

    return list(inner())


def digest(path):
    with path.open("rb") as f:
        return file_digest(f, "sha-256").hexdigest()


if __name__ == "__main__":
    main()
