"""
Standalone script for writing release doc::

    python tools/write_release <version>

Example::

    python tools/write_release.py 1.7.0

Needs to be run from the root of the repository and assumes
that the output is in `release` and wheels and sdist in
`release/installers`.

Translation from rst to md markdown requires Pandoc, you
will need to rely on your distribution to provide that.

"""
import argparse
import os
import subprocess
from pathlib import Path

# Name of the notes directory
NOTES_DIR = "doc/source/release"
# Name of the output directory
OUTPUT_DIR = "release"
# Output base name, `.rst` or `.md` will be appended
OUTPUT_FILE = "README"

def write_release(version):
    """
    Copy the <version>-notes.rst file to the OUTPUT_DIR and use
    pandoc to translate it to markdown. That results in both
    README.rst and README.md files that can be used for on
    github for the release.

    Parameters
    ----------
    version: str
       Release version, e.g., '2.3.2', etc.

    Returns
    -------
    None.

    """
    notes = Path(NOTES_DIR) / f"{version}-notes.rst"
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(exist_ok=True)
    target_md = outdir / f"{OUTPUT_FILE}.md"
    target_rst = outdir / f"{OUTPUT_FILE}.rst"

    # translate README.rst to md for posting on GitHub
    os.system(f"cp {notes} {target_rst}")
    subprocess.run(
        ["pandoc", "-s", "-o", str(target_md), str(target_rst), "--wrap=preserve"],
        check=True,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "version",
        help="NumPy version of the release, e.g. 2.3.2, etc.")

    args = parser.parse_args()
    write_release(args.version)
