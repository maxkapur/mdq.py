import base64
from argparse import ArgumentParser
from pathlib import Path

import mdq.query
import mdq.systemd


def main(args: None | list[str] = None) -> None:
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
        help="Paths to files or directories to query; defaults to current working directory (see --extensions)",
    )
    parser.add_argument(
        "-q",
        "--query",
        default=None,
        help="Search query",
    )
    parser.add_argument(
        "-e",
        "--extensions",
        nargs="*",
        default=["md", "markdown", "txt"],
        help="Extensions to match when recursing through a directory",
    )
    parser.add_argument(
        "-k",
        "-n",
        "--n-matches",
        default=4,
        type=int,
        help="Number of matches to return",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        type=Path,
        help="Directory to store cache of embeddings and downloaded models",
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-Embedding-0.6B",
        type=str,
        help="Name of fastembed embedding model",
    )
    parser.add_argument(
        "--query-prefix",
        default="query: ",
        type=str,
        help="Query prefix; default value of 'query: ' can improve performance with default model",
    )
    parser.set_defaults(func=mdq.query.handle)

    # systemd subcommand
    systemd_parser = parser.add_subparsers().add_parser(
        "systemd",
        help="Generate systemd service file to embed files in current working directory on login",
    )
    systemd_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Override existing service file",
    )
    systemd_parser.set_defaults(func=mdq.systemd.handle)

    options = parser.parse_args(args)

    # Use this as identifier in tables since IDK what strings are allowed as
    # model identifiers
    options.model_name_b64 = base64.b64encode(options.model_name.encode())
    options.func(options)


if __name__ == "__main__":
    main()
