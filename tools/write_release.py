"""
Standalone script for writing release doc::

    python tools/write_release <version>

Example::

    python tools/write_release.py 1.7.0

Needs to be run from the root of the repository and assumes
that the output is in `release` and wheels and sdist in
`release/installers`.

"""
import argparse
import re
from pathlib import Path

# Name of the notes directory
NOTES_DIR = "doc/source/release"
# Name of the output directory
OUTPUT_DIR = "release"
# Name of the output file
OUTPUT_FILE = "README.md"

def myst_to_gfm(text):
    """
    Translate MyST markdown to GitHub flavored markdown.

    Drops the front matter and directives without content, and renders
    roles such as {func}`numpy.sum` as code.
    """
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)
    text = re.sub(r"^```\{[\w:-]+\}.*\n```\n", "", text, flags=re.MULTILINE)

    def role(match):
        name, content = match.groups()
        content = re.sub(r"^(.*?)\s*<.*>$", r"\1", content).lstrip("~!")
        return content if name in ("ref", "doc") else f"`{content}`"

    return re.sub(r"(?<!`)\{([\w:]+)\}`([^`\n]+)`", role, text).lstrip()


def write_release(version):
    """
    Write the <version>-notes.md file to OUTPUT_DIR as GitHub
    flavored markdown, which can be used on github for the release.

    Parameters
    ----------
    version: str
       Release version, e.g., '2.3.2', etc.

    Returns
    -------
    None.

    """
    notes = Path(NOTES_DIR) / f"{version}-notes.md"
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(exist_ok=True)
    (outdir / OUTPUT_FILE).write_text(myst_to_gfm(notes.read_text()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "version",
        help="NumPy version of the release, e.g. 2.3.2, etc.")

    args = parser.parse_args()
    write_release(args.version)
