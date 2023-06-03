#!/usr/bin/env python
"""
check_license.py [MODULE]

Check the presence of a LICENSE.txt in the installed module directory,
and that it appears to contain text prevalent for a NumPy binary
distribution.

"""
import os
import sys
import re
import argparse


def check_text(text):
    ok = "Copyright (c)" in text and re.search(
        r"This binary distribution of \w+ also bundles the following software",
        text,
    )
    return ok


def main():
    p = argparse.ArgumentParser(usage=__doc__.rstrip())
    p.add_argument("module", nargs="?", default="numpy")
    args = p.parse_args()

    # Drop '' from sys.path
    sys.path.pop(0)

    # Find module path
    __import__(args.module)
    mod = sys.modules[args.module]

    # Check license text
    license_txt = os.path.join(os.path.dirname(mod.__file__), "LICENSE.txt")
    with open(license_txt, encoding="utf-8") as f:
        text = f.read()

    ok = check_text(text)
    if not ok:
        print(
            "ERROR: License text {} does not contain expected "
            "text fragments\n".format(license_txt)
        )
        print(text)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
