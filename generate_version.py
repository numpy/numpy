# Note: This file has to live next to versioneer.py or it will not work
import argparse
import os

import versioneer


def write_version_info(path):
    vinfo = versioneer.get_versions()
    full_version = vinfo['version']
    git_revision = vinfo['full-revisionid']

    if os.environ.get("MESON_DIST_ROOT"):
        path = os.path.join(os.environ.get("MESON_DIST_ROOT"), path)

    with open(path, "w") as f:
        f.write("def get_versions():\n")
        f.write("    return {\n")
        f.write(f"        'full-revisionid': '{git_revision}',\n")
        f.write(f"        'version': '{full_version}'\n")
        f.write("}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o", "--outfile", type=str, help="Path to write version info to"
    )
    args = parser.parse_args()

    if not args.outfile.endswith(".py"):
        raise ValueError(
            f"Output file must be a Python file. "
            f"Got: {args.outfile} as filename instead"
        )

    write_version_info(args.outfile)


main()
