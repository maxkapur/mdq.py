import sqlite3
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import sqlite_vec

import mdq
import mdq.file_utils


def handle(options: Namespace) -> None:
    """Handle parsed command line options for query mode (no subcommand)."""
    query_str = options.query or sys.stdin.readline()
    options.query = mdq.query_prefix + query_str.strip()

    options.extensions = set("." + e.lstrip(".") for e in options.extensions)

    conn = initialize_db()

    # Paths to all text files implied by command line arguments (after globbing
    # directories)
    all_text_file_paths = mdq.file_utils.get_text_file_paths(options)
    refresh_embeddings_db(all_text_file_paths, conn)

    for path_str in fetch_top_matches(all_text_file_paths, conn, options):
        print(str(path_str))


def initialize_db() -> sqlite3.Connection:
    """Get a connection to the sqlite database.

    Load the sqlite-vec extension. Create files and tables as needed.
    """

    conn = sqlite3.connect(str(mdq.conn_path))

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

        embedding_size = mdq.embed_model.get_embedding_size(mdq.embed_model.model_name)

        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS embedding USING vec0(
                digest TEXT UNIQUE,
                vec FLOAT[{embedding_size}]
            )
        """)

        # Temp space to hold embeddings for just the paths requested
        conn.execute("ATTACH DATABASE ':memory:' AS mem")
        conn.execute(f"""
            CREATE VIRTUAL TABLE mem.filtered USING vec0(
                path TEXT UNIQUE,
                vec FLOAT[{embedding_size}]
            )
        """)

    return conn


def refresh_embeddings_db(
    requested_paths: list[Path], conn: sqlite3.Connection
) -> None:
    """Update the database to reflect the current state of all requested paths.

    Update file hashes in the document table. For any unseen hashes, compute the
    embeddings and add them to the embeddings table.
    """

    # Metadata for all text files *not* reflected in the documents table--either
    # missing an entry for that path, or the mtime (and possibly hash) is out of
    # date
    updated_text_file_metadatas = mdq.file_utils.get_outdated_paths(
        requested_paths, conn
    )

    # Single transaction to prevent any document.digests from having no match in
    # the embedding table if program terminated during embeddings. TODO: enforce
    # this with a primary key constraint instead; update embedding first, then
    # documents.
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

        # Subset of previous list: Just those for which the hash is absent from
        # our embeddings table, and thus we need to compute a new embedding
        need_embedding_metadatas = []
        for metadata in updated_text_file_metadatas:
            # See if a new embedding is needed (it could have had its timestamp
            # updated but identical content, or it could have been updated to
            # have its contents match those of an already-embedded document)
            if (
                conn.execute(
                    "SELECT digest FROM embedding WHERE digest = ?", [metadata.digest]
                ).fetchone()
                is not None
            ):
                continue

            need_embedding_metadatas.append(metadata)

        with mdq.console.status(f"Embed {len(need_embedding_metadatas)} documents"):
            embed_docs = [
                metadata.path.read_text() for metadata in need_embedding_metadatas
            ]
            embeddings = [
                e.astype(np.float32) for e in mdq.embed_model.embed(embed_docs)
            ]

        conn.executemany(
            "INSERT INTO embedding(digest, vec) VALUES (?, ?)",
            [
                (metadata.digest, vec)
                for metadata, vec in zip(
                    need_embedding_metadatas, embeddings, strict=True
                )
            ],
        )


def fetch_top_matches(
    paths: list[Path], conn: sqlite3.Connection, options: Namespace
) -> list[str]:
    (query_embed,) = mdq.embed_model.embed([options.query])
    query_embed_arr = np.array(query_embed, dtype=np.float32)

    # Populate the temporary, in-memory table which joins only the paths
    # requested to their embeddings. This can't be done as a subquery because
    # sqlite-vec's indexing algorithm needs to rank the top k relative to an
    # entire column of embeddings. (For example, if you try to just join
    # document and embedding and select the top k with a WHERE statement on
    # document.path, the top k are selected *before* applying the WHERE
    # statement and you get fewer than k matches.)
    question_marks = ",".join("?" * len(paths))
    with conn:
        conn.execute(
            f"""
            INSERT INTO mem.filtered(path, vec)
                SELECT path, vec
                FROM document
                JOIN embedding
                ON document.digest = embedding.digest
                WHERE path IN ({question_marks})
            """,
            [str(p.absolute()) for p in paths],
        )

    res = conn.execute(
        """
        SELECT path
            FROM mem.filtered
            WHERE
                vec MATCH ?
                AND k = ?
            ORDER BY distance
        """,
        [query_embed_arr, options.n_matches],
    ).fetchall()
    return [path_str for (path_str,) in res]
