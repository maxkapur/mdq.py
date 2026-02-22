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
    )
    parser.add_argument("-q", "--query", default=None)
    parser.add_argument(
        "-e", "--extensions", nargs="*", default=["md", "markdown", "txt"]
    )
    parser.add_argument("-k", "-n", "--n-matches", default=4, type=int)
    parser.set_defaults(func=mdq.query.handle)

    # systemd subcommand
    systemd_parser = parser.add_subparsers().add_parser(
        "systemd", help="Generate systemd service file and enable service"
    )
    systemd_parser.add_argument(
        "-f", "--force", action="store_true", help="Override existing service file"
    )
    systemd_parser.set_defaults(func=mdq.systemd.handle)

    options = parser.parse_args(args)
    options.func(options)


if __name__ == "__main__":
    main()
