#!python3
""" Platform independent file copier script
"""

import shutil
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infiles", nargs='+', help="Paths to the input files")
    parser.add_argument("outdir", help="Path to the output directory")
    args = parser.parse_args()
    for infile in args.infiles:
        shutil.copy2(infile, args.outdir)


if __name__ == "__main__":
    main()
