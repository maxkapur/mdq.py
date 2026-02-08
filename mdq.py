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

    # Paths to all text files implied by command line arguments (after globbing
    # directories)
    all_text_file_paths = get_text_file_paths(options)

    # Metadata for all text files *not* reflected in the documents table--either
    # missing an entry for that path, or the mtime (and possibly hash) is out of
    # date
    updated_text_file_metadatas = get_outdated_paths(all_text_file_paths, conn)

    with conn:
        # Insert any new paths
        conn.executemany(
            "INSERT OR IGNORE INTO document(path, digest, mtime) VALUES (?, ?, ?)",
            [(p.path_str, p.digest, p.mtime) for p in updated_text_file_metadatas],
        )
        # Update all path digests
        conn.executemany(
            "UPDATE document SET digest=?, mtime=? WHERE path=?",
            [(p.digest, p.mtime, p.path_str) for p in updated_text_file_metadatas],
        )

    # Subset of previous list: Just those for which the hash is absent from our
    # embeddings table, and thus we need to compute a new embedding
    need_embedding_metadatas = []
    for metadata in updated_text_file_metadatas:
        # See if a new embedding is needed (it could have had its timestamp
        # updated but identical content, or it could have been updated to have
        # its contents match those of an already-embedded document)
        if (
            conn.execute(
                "SELECT digest FROM embedding WHERE digest = ?", [metadata.digest]
            ).fetchone()
            is not None
        ):
            continue

        need_embedding_metadatas.append(metadata)

    with console.status(f"Embed {len(need_embedding_metadatas)} documents"):
        embed_docs = [
            metadata.path.read_text() for metadata in need_embedding_metadatas
        ]
        embeddings = [e.astype(np.float32) for e in embed_model.embed(embed_docs)]
    del embed_docs

    with conn:
        conn.executemany(
            "INSERT INTO embedding(digest, vec) VALUES (?, ?)",
            [
                (metadata.digest, vec)
                for metadata, vec in zip(
                    need_embedding_metadatas, embeddings, strict=True
                )
            ],
        )

    query_embed = np.array(*embed_model.embed([options.query]), dtype=np.float32)

    question_marks = ",".join("?" * len(all_text_file_paths))
    results = conn.execute(
        f"""
            SELECT path
                FROM document
                JOIN embedding
                ON document.digest = embedding.digest
                WHERE
                    document.path IN ({question_marks})
                    AND embedding.vec MATCH ?
                    AND k = ?
                ORDER BY distance
            """,
        [str(p.absolute()) for p in all_text_file_paths]
        + [query_embed, options.n_matches],
    ).fetchall()
    for (metadata,) in results:
        print(str(metadata))


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
                digest TEXT UNIQUE,
                vec FLOAT[{embedding_size}]
            )
        """)

    return conn


def get_text_file_paths(options):
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


def get_outdated_paths(paths, conn):
    """List of metadata for files that are outdated.

    A file is outdated if it is either entirely absent from our documents table,
    or the file's mtime has been updated since we last saw it.

    An outdated file *may* need to be read and embedded.
    """

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
                    # File hasn't changed since inclusion in the table, no
                    # embedding needed
                    continue

            yield DocumentMetadata(path, digest(path), new_mtime)

    return list(inner())


def digest(path):
    with path.open("rb") as f:
        return file_digest(f, "sha-256").hexdigest()


if __name__ == "__main__":
    main()
