#!/usr/bin/env python3

"""
argparse+logging front-end to f2py

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
import logging
import pathlib

###########
# Helpers #
###########


def check_fortran(fname):
    fpname = pathlib.Path(fname)
    if fpname.suffix.lower() in [".f90", ".f", ".f77"]:
        return fpname
    else:
        logger.warning(
            """Does not look like a standard fortran file ending in *.f90, *.f or
            *.f77, continuing against better judgement"""
        )
        return fpname


##########
# Parser #
##########


parser = argparse.ArgumentParser(
    prog="f2py",
    description="""This program generates a Python C/API file
    (<modulename>module.c) that contains wrappers for given fortran
    functions so that they can be called from Python. With the -c
    option the corresponding extension modules are built.""",
    add_help=False,  # Since -h is taken...
    # Format to look like f2py
    formatter_class=lambda prog: argparse.HelpFormatter(
        prog, max_help_position=100, width=80
    ),
)

build_helpers = parser.add_argument_group("build helpers")
generate_wrappers = parser.add_argument_group("wrappers and signature files")

# Common #
##########

# --help is still free
parser.add_argument("--help", action="store_true", help="Print the help")

# TODO: Remove?
parser.add_argument(
    "--2d-numpy",
    type=bool,
    default=True,
    help="Use f2py with Numeric support [DEFAULT]",
)

parser.add_argument(
    "Fortran Functions",
    metavar="<fortran functions>",
    type=str,
    nargs="*",
    default="ALL",
    help="""Names of fortran routines for which Python C/API
                   functions will be generated. Default is all that are found
                   in <fortran files>.""",
)

parser.add_argument(
    "Fortran Files",
    metavar="<fortran files>",
    type=check_fortran,  # Returns a pathlib.Path
    nargs="*",  # As many as present, 0 OK
    help="""Paths to fortran/signature files that will be scanned for
                   <fortran functions> in order to determine their signatures.""",
)

# Wrappers/Signatures #
#######################

generate_wrappers.add_argument(
    # TODO: Seriously consider scrapping this naming convention
    "-h",
    "--hint-signature",
    type=pathlib.Path,
    nargs=1,
    help=r"""
    Write signatures of the fortran routines to file <filename> and exit. You
    can then edit <filename> and use it instead of <fortran files>. If
    <filename>==stdout then the signatures are printed to stdout.
    """,
)


def process_args(args):
    if args.help:
        parser.print_help()
    elif getattr(args, "Fortran Files"):
        print("BOOM")
    else:
        parser.print_usage()


if __name__ == "__main__":
    logger = logging.getLogger("f2py_cli")
    logger.setLevel(logging.WARNING)
    args = parser.parse_args()
    process_args(args)
