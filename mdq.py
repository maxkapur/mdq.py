import shlex
import sqlite3
import subprocess
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


def main(args=None):
    """Command line entry point.

    Parse command line arguments and delegate to the appropriate handler.
    """
    parser = ArgumentParser()

    # No subcommand: query documents
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
    parser.set_defaults(func=handle_query)

    # systemd subcommand
    systemd_parser = parser.add_subparsers().add_parser(
        "systemd", help="Generate systemd service file and enable service"
    )
    systemd_parser.add_argument(
        "-f", "--force", action="store_true", help="Override existing service file"
    )
    systemd_parser.set_defaults(func=systemd_integrate)

    options = parser.parse_args(args)
    options.func(options)


def handle_query(options):
    query_str = options.query or sys.stdin.readline()
    options.query = query_prefix + query_str.strip()

    options.extensions = set("." + e.lstrip(".") for e in options.extensions)

    conn = initialize_db()

    # Paths to all text files implied by command line arguments (after globbing
    # directories)
    all_text_file_paths = get_text_file_paths(options)
    refresh_embeddings_db(all_text_file_paths, conn)

    for (path_str,) in fetch_top_matches(
        options.n_matches, options.query, all_text_file_paths, conn
    ):
        print(str(path_str))


def refresh_embeddings_db(requested_paths, conn):
    # Metadata for all text files *not* reflected in the documents table--either
    # missing an entry for that path, or the mtime (and possibly hash) is out of
    # date
    updated_text_file_metadatas = get_outdated_paths(requested_paths, conn)

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
    for path_str in updated_text_file_metadatas:
        # See if a new embedding is needed (it could have had its timestamp
        # updated but identical content, or it could have been updated to have
        # its contents match those of an already-embedded document)
        if (
            conn.execute(
                "SELECT digest FROM embedding WHERE digest = ?", [path_str.digest]
            ).fetchone()
            is not None
        ):
            continue

        need_embedding_metadatas.append(path_str)

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


def fetch_top_matches(paths, conn, options):
    query_embed = np.array(*embed_model.embed([options.query]), dtype=np.float32)

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

    return conn.execute(
        """
        SELECT path
            FROM mem.filtered
            WHERE
                vec MATCH ?
                AND k = ?
            ORDER BY distance
        """,
        [query_embed, options.n_matches],
    ).fetchall()


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

        # Temp space to hold embeddings for just the paths requested
        conn.execute("ATTACH DATABASE ':memory:' AS mem")
        conn.execute(f"""
            CREATE VIRTUAL TABLE mem.filtered USING vec0(
                path TEXT UNIQUE,
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


def systemd_integrate(options):
    if not Path("/run/systemd/system").exists():
        raise ValueError("Host appears not to use systemd")

    dest = Path.home() / ".config" / "systemd" / "user" / "mdq.service"
    with console.status(f"Creating unit file {dest}"):
        if dest.exists() and not options.force:
            raise FileExistsError(str(dest))

        dest.parent.mkdir(parents=True, exist_ok=True)

        # Default setup: dummy query, k=0, default extensions in current (at
        # time of running `mdq systemd`) working directory
        cmd = shlex.join(
            [sys.executable, "-m", "mdq", "-q", "dummy", "-k", "0"],
        )
        body = f"""\
# Autogenerated by {__file__}
dest
[Unit]
Description=Run dummy mdq query to populate cache

[Service]
Type=oneshot
WorkingDirectory={str(Path().absolute())}
ExecStart={cmd}

[Install]
WantedBy=default.target
"""
        dest.write_text(body)

    cmd = ["systemctl", "--user", "daemon-reload"]
    with console.status(shlex.join(cmd)):
        subprocess.run(cmd).check_returncode()

    cmd = ["systemctl", "--user", "enable", "--now", "mdq"]
    with console.status(shlex.join(cmd)):
        subprocess.run(cmd).check_returncode()


if __name__ == "__main__":
    main()
