#!/usr/bin/env python3
import argparse
import os
import sys

import tempita


def process_tempita(fromfile, outfile=None):
    """Process tempita templated file and write out the result.

    The template file is expected to end in `.c.in` or `.pyx.in`:
    E.g. processing `template.c.in` generates `template.c`.

    """
    if outfile is None:
        # We're dealing with a distutils build here, write in-place
        outfile = os.path.splitext(fromfile)[0]

    from_filename = tempita.Template.from_filename
    template = from_filename(fromfile, encoding=sys.getdefaultencoding())

    content = template.substitute()

    with open(outfile, 'w') as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "infile",
        type=str,
        help="Path to the input file"
    )
    parser.add_argument(
        "-o",
        "--outfile",
        type=str,
        help="Path to the output file"
    )
    parser.add_argument(
        "-i",
        "--ignore",
        type=str,
        help="An ignored input - may be useful to add a "
        "dependency between custom targets",
    )
    args = parser.parse_args()

    if not args.infile.endswith('.in'):
        raise ValueError(f"Unexpected extension: {args.infile}")

    outfile_abs = os.path.join(os.getcwd(), args.outfile)
    process_tempita(args.infile, outfile_abs)


if __name__ == "__main__":
    main()
