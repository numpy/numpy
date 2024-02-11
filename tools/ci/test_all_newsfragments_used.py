#!/usr/bin/env python3

import sys
import toml
import os

def main():
    path = toml.load("pyproject.toml")["tool"]["towncrier"]["directory"]

    fragments = os.listdir(path)
    fragments.remove("README.rst")
    fragments.remove("template.rst")

    if fragments:
        print("The following files were not found by towncrier:")
        print("    " + "\n    ".join(fragments))
        sys.exit(1)


if __name__ == "__main__":
    main()
