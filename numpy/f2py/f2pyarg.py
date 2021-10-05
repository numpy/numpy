#!/usr/bin/env python3

"""
argparse front-end to f2py

The concept is based around the idea that F2PY is overloaded in terms of
functionality:

1. Creating the wrapper `.c` files
2. Generating `.pyf` signature files
3. Compilation helpers
  a. This essentially means `numpy.distutils` for now

The three functionalities are largely independent of each other, hence the
implementation in terms of subparsers
"""

import argparse


parser = argparse.ArgumentParser(
    prog="f2py",
    description="""This program generates a Python C/API file
    (<modulename>module.c) that contains wrappers for given fortran
    functions so that they can be called from Python. With the -c
    option the corresponding extension modules are built.""",
    add_help=False,
)

build_helpers = parser.add_argument_group("build helpers")
generate_wrappers = parser.add_argument_group("wrappers and signature files")

# Common #
##########

parser.add_argument(
    "Fortran Files",
    metavar="<fortran files>",
    type=str,
    nargs="*",
    help="""Paths to fortran/signature files that will be scanned for
                   <fortran functions> in order to determine their signatures.""",
)

parser.add_argument("--2d-numpy", default=True, help="Use f2py with Numeric support")

# Wrappers/Signatures #
#######################

generate_wrappers.add_argument(
    # TODO: Seriously consider scrapping this naming convention
    "-h",
    "--hint-signature",
    type=str,
    nargs=1,
    help="""Write signatures of the fortran routines to file <filename> and
    exit. You can then edit <filename> and use it instead of
    <fortran files>. If <filename>==stdout then the signatures
    are printed to stdout.""",
)

parser.add_argument("--help", action="store_true", help="Print the help")


def process_args(args):
    if args.help:
        parser.print_help()
    elif getattr(args, "Fortran Files"):
        print("BOOM")
    else:
        parser.print_usage()


if __name__ == "__main__":
    args = parser.parse_args()
    process_args(args)
