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
import textwrap
from hashlib import md5, sha256
from pathlib import Path

# Name of the notes directory
NOTES_DIR = "doc/source/release"
# Name of the output directory
OUTPUT_DIR = "release"
# Output base name, `.rst` or `.md` will be appended
OUTPUT_FILE = "README"

def compute_hash(wheel_dir, hash_func):
    """
    Compute hashes of files in wheel_dir.

    Parameters
    ----------
    wheel_dir: str
        Path to wheel directory from repo root.
    hash_func: function
        Hash function, i.e., md5, sha256, etc.

    Returns
    -------
    list_of_strings: list
        List of of strings. Each string is the hash
        followed by the file basename.

    """
    released = os.listdir(wheel_dir)
    checksums = []
    for fn in sorted(released):
        fn_path = Path(f"{wheel_dir}/{fn}")
        m = hash_func(fn_path.read_bytes())
        checksums.append(f"{m.hexdigest()}  {fn}")
    return checksums


def write_release(version):
    """
    Copy the <version>-notes.rst file to the OUTPUT_DIR, append
    the md5 and sha256 hashes of the wheels and sdist, and produce
    README.rst and README.md files.

    Parameters
    ----------
    version: str
       Release version, e.g., '2.3.2', etc.

    Returns
    -------
    None.

    """
    notes = Path(NOTES_DIR) / f"{version}-notes.rst"
    wheel_dir = Path(OUTPUT_DIR) / "installers"
    target_md = Path(OUTPUT_DIR) / f"{OUTPUT_FILE}.md"
    target_rst = Path(OUTPUT_DIR) / f"{OUTPUT_FILE}.rst"

    os.system(f"cp {notes} {target_rst}")

    with open(str(target_rst), 'a') as f:
        f.writelines(textwrap.dedent(
            """
            Checksums
            =========

            MD5
            ---
            ::

            """))
        f.writelines([f'    {c}\n' for c in compute_hash(wheel_dir, md5)])

        f.writelines(textwrap.dedent(
            """
            SHA256
            ------
            ::

            """))
        f.writelines([f'    {c}\n' for c in compute_hash(wheel_dir, sha256)])

    # translate README.rst to md for posting on GitHub
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
