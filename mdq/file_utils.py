from argparse import Namespace
from hashlib import file_digest
from pathlib import Path
from sqlite3 import Connection
from typing import NamedTuple


def get_text_file_paths(options: Namespace) -> list[Path]:
    """Extract individual file paths from `options.paths` and `.extensions`."""

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
    """Container of data needed to check which files need new embeddings."""

    path: Path
    digest: str
    mtime: float

    @property
    def path_str(self) -> str:
        return str(self.path.absolute())


def get_outdated_paths(paths: list[Path], conn: Connection) -> list[DocumentMetadata]:
    """List of metadata for files that are outdated.

    A file is outdated if it is either entirely absent from our documents table,
    or the file's `.mtime` has been updated since we last saw it.

    All outdated files need to be read. Some, but not all, required computing a
    new embedding. A new embedding is *not* required if the file was updated
    such that its contents match something we've embedded before.
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


def digest(path: Path) -> str:
    with path.open("rb") as f:
        return file_digest(f, "sha-256").hexdigest()
