import re
import subprocess

import pytest


@pytest.fixture(scope="session")
def mock_directory(tmp_path_factory):
    directory = tmp_path_factory.mktemp("session_cwd")

    (directory / "notes").mkdir()
    (directory / "notes" / "sasquatch.md").write_text(
        # https://en.wikipedia.org/wiki/Bigfoot
        "Bigfoot (/ˈbɪɡ.fʊt/), also commonly referred to as Sasquatch (/ˈsæs.kwɒtʃ/, SASS-kwotch; /ˈsæs.kwætʃ/, SASS-kwahtch), is a large, hairy, mythical creature said to inhabit forests in North America, particularly in the Pacific Northwest.[2][3][4] Bigfoot is featured in both American and Canadian folklore, and since the mid-20th century has become a cultural icon, permeating popular culture and becoming the subject of its own distinct subculture.[5][6]"
    )

    (directory / "notes" / "more notes").mkdir()
    (directory / "notes" / "more notes" / "senarathne.txt").write_text(
        # https://en.wikipedia.org/wiki/Nuwandhika_Senarathne
        "Nuwandhika Senarathne (born 23 June 1993) is a Sri Lankan singer and television personality who is most known as a playback singer, as well as a performer of Soprano Opera and Ghazals. She rose to fame as the 1st runner-up of the reality TV show Derana Dream Star Season IX.[1]"
    )

    (directory / "docs").mkdir()
    (directory / "docs" / "satron.md").write_text(
        # https://en.wikipedia.org/wiki/Satron
        "Satron is a hamlet in Swaledale, North Yorkshire, England. It lies 0.6 miles (1 km) south west of Gunnerside[1] on the opposite bank of the River Swale. It is in the civil parish of Muker,[2] but used to be in the ancient parish of Grinton.[3] From 1974 to 2023 it was part of the district of Richmondshire, it is now administered by the unitary North Yorkshire Council."
    )
    (directory / "docs" / "village.txt").write_text(
        # https://en.wikipedia.org/wiki/Brane%C5%A1ci_(%C4%8Cajetina)
        "Branešci is a village in the municipality of Čajetina, western Serbia. According to the 2011 census, the village has a population of 737 people.[2]"
    )

    return directory


# Integration tests of the examples from the README


def test_basic(mock_directory):
    cmd = ["mdq", "-q", "sasquatch"]
    res = subprocess.run(cmd, cwd=mock_directory, text=True, capture_output=True)

    assert re.match(
        """\
.*/notes/sasquatch.md
.*/docs/satron.md
.*/docs/village.txt
.*/notes/more notes/senarathne.txt
""",
        res.stdout,
    )


def test_stdin(mock_directory):
    cmd = ["mdq"]
    res = subprocess.run(
        cmd, cwd=mock_directory, text=True, capture_output=True, input="sasquatch"
    )

    assert re.match(
        """\
.*/notes/sasquatch.md
.*/docs/satron.md
.*/docs/village.txt
.*/notes/more notes/senarathne.txt
""",
        res.stdout,
    )


def test_globs(mock_directory):
    cmd = [
        "mdq",
        "-q",
        "sasquatch",
        "-p",
        *mock_directory.glob("./notes/*.md"),
        *mock_directory.glob("./docs/*.md"),
    ]
    res = subprocess.run(
        cmd, cwd=mock_directory, text=True, capture_output=True, input="sasquatch"
    )

    assert re.match(
        """\
.*/notes/sasquatch.md
.*/docs/satron.md
""",
        res.stdout,
    )
