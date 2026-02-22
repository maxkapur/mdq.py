import subprocess
import sys
from pathlib import Path

import mypy.api

REPO_ROOT = Path(".").parent.parent


def test_mypy():
    stdout, stderr, returncode = mypy.api.run([str(REPO_ROOT)])
    sys.stdout.write(stdout)
    sys.stderr.write(stderr)
    assert returncode == 0


def test_ruff():
    subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check"]
    ).check_returncode()
    subprocess.run([sys.executable, "-m", "ruff", "check"]).check_returncode()
