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
import tempfile

from numpy.version import version as __version__

# F2PY imports
from . import crackfortran
from . import rules

##################
# Temp Variables #
##################

# TODO: Kill these np.distutil specific variables
npd_link = ['atlas', 'atlas_threads', 'atlas_blas', 'atlas_blas_threads',
            'lapack_atlas', 'lapack_atlas_threads', 'atlas_3_10',
            'atlas_3_10_threads', 'atlas_3_10_blas', 'atlas_3_10_blas_threads'
            'lapack_atlas_3_10', 'lapack_atlas_3_10_threads', 'flame', 'mkl',
            'openblas', 'openblas_lapack', 'openblas_clapack', 'blis',
            'lapack_mkl', 'blas_mkl', 'accelerate', 'openblas64_',
            'openblas64__lapack', 'openblas_ilp64', 'openblas_ilp64_lapack'
            'x11', 'fft_opt', 'fftw', 'fftw2', 'fftw3', 'dfftw', 'sfftw',
            'fftw_threads', 'dfftw_threads', 'sfftw_threads', 'djbfft', 'blas',
            'lapack', 'lapack_src', 'blas_src', 'numpy', 'f2py', 'Numeric',
            'numeric', 'numarray', 'numerix', 'lapack_opt', 'lapack_ilp64_opt',
            'lapack_ilp64_plain_opt', 'lapack64__opt', 'blas_opt',
            'blas_ilp64_opt', 'blas_ilp64_plain_opt', 'blas64__opt',
            'boost_python', 'agg2', 'wx', 'gdk_pixbuf_xlib_2',
            'gdk-pixbuf-xlib-2.0', 'gdk_pixbuf_2', 'gdk-pixbuf-2.0', 'gdk',
            'gdk_2', 'gdk-2.0', 'gdk_x11_2', 'gdk-x11-2.0', 'gtkp_x11_2',
            'gtk+-x11-2.0', 'gtkp_2', 'gtk+-2.0', 'xft', 'freetype2', 'umfpack',
            'amd']

###########
# Helpers #
###########


def check_fortran(fname: str):
    """Function which checks <fortran files>

    This is meant as a sanity check, but will not raise an error, just a
    warning.  It is called with ``type``

    Parameters
    ----------
    fname : str
        The name of the file

    Returns
    -------
    pathlib.Path
        This is the string as a path, irrespective of the suffix
    """
    fpname = pathlib.Path(fname)
    if fpname.suffix.lower() in [".f90", ".f", ".f77"]:
        return fpname
    else:
        logger.warning(
            """Does not look like a standard fortran file ending in *.f90, *.f or
            *.f77, continuing against better judgement"""
        )
        return fpname


def check_dir(dname: str):
    """Function which checks the build directory

    This is meant to ensure no odd directories are passed, it will fail if a
    file is passed

    Parameters
    ----------
    dname : str
        The name of the directory, by default it will be a temporary one

    Returns
    -------
    pathlib.Path
        This is the string as a path
    """
    if dname == "tempfile.mkdtemp()":
        dname = tempfile.mkdtemp()
        return pathlib.Path(dname)
    else:
        dpname = pathlib.Path(dname)
        if dpname.is_dir():
            return dpname
        else:
            raise RuntimeError(f"{dpname} is not a directory")


def check_dccomp(opt: str):
    """Function which checks for an np.distutils compliant c compiler

    Meant to enforce sanity checks, note that this just checks against distutils.show_compilers()

    Parameters
    ----------
    opt: str
        The compiler name, must be a distutils option

    Returns
    -------
    str
        This is the option as a string
    """
    cchoices = ["bcpp", "cygwin", "mingw32", "msvc", "unix"]
    if opt in cchoices:
        return opt
    else:
        raise RuntimeError(f"{opt} is not an distutils supported C compiler, choose from {cchoices}")


def check_npfcomp(opt: str):
    """Function which checks for an np.distutils compliant fortran compiler

    Meant to enforce sanity checks

    Parameters
    ----------
    opt: str
        The compiler name, must be a np.distutils option

    Returns
    -------
    str
        This is the option as a string
    """
    from numpy.distutils import fcompiler
    fcompiler.load_all_fcompiler_classes()
    fchoices = list(fcompiler.fcompiler_class.keys())
    if opt in fchoices[0]:
        return opt
    else:
        raise RuntimeError(f"{opt} is not an np.distutils supported compiler, choose from {fchoices}")


# TODO: Compatibility helper, kill later
# From 3.9 onwards should be argparse.BooleanOptionalAction
class BoolAction(argparse.Action):
    """A custom action to mimic Ruby's --[no]-blah functionality in f2py

    This is meant to use ``argparse`` with a custom action to ensure backwards
    compatibility with ``f2py``. Kanged `from here`_.

    .. note::

       Like in the old ``f2py``, it is not an error to pass both variants of
       the flag, the last one will be used

    .. from here:
        https://thisdataguy.com/2017/07/03/no-options-with-argparse-and-python/
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        """Initialization of the boolean flag

        Mimics the parent
        """
        super(BoolAction, self).__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        """The logical negation action

        Essentially this returns the semantically valid operation implied by
        --no
        """
        setattr(namespace, self.dest, False if option_string.contains("no") else True)


# TODO: Generalize or kill this np.distutils specific helper action class
class NPDLinkHelper(argparse.Action):
    """A custom action to work with f2py's --link-blah

    This is effectively the same as storing help_link

    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        """Initialization of the boolean flag

        Mimics the parent
        """
        super(NPDLinkHelper, self).__init__(option_strings, dest, nargs="*", **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        """The storage action

        Essentially, split the value on -, store in dest

        """
        items = getattr(namespace, self.dest) or []
        outvar = option_string.split("--link-")[1]
        if outvar in npd_link:
            # replicate the extend functionality
            items.extend(outvar)
        else:
            raise RuntimeError(f"{outvar} is not in {npd_link}")


##########
# Parser #
##########


parser = argparse.ArgumentParser(
    prog="f2py",
    description="""
    This program generates a Python C/API file (<modulename>module.c) that
    contains wrappers for given fortran functions so that they can be called
    from Python.

    With the -c option the corresponding extension modules are built.""",
    add_help=False,  # Since -h is taken...
    # Format to look like f2py
    formatter_class=lambda prog: argparse.RawDescriptionHelpFormatter(
        prog, max_help_position=100, width=85
    ),
    epilog=f"""
    Using the following macros may be required with non-gcc Fortran
  compilers:
    -DPREPEND_FORTRAN -DNO_APPEND_FORTRAN -DUPPERCASE_FORTRAN
    -DUNDERSCORE_G77

  When using -DF2PY_REPORT_ATEXIT, a performance report of F2PY
  interface is printed out at exit (platforms: Linux).

  When using -DF2PY_REPORT_ON_ARRAY_COPY=<int>, a message is
  sent to stderr whenever F2PY interface makes a copy of an
  array. Integer <int> sets the threshold for array sizes when
  a message should be shown.

    Version:     {__version__}
    numpy Version: {__version__}
    Requires:    Python 3.5 or higher.
    License:     NumPy license (see LICENSE.txt in the NumPy source code)
    Copyright 1999 - 2011 Pearu Peterson all rights reserved.
    https://web.archive.org/web/20140822061353/http://cens.ioc.ee/projects/f2py2e
    """
)

# subparsers = parser.add_subparsers(help="Functional subsets")
build_helpers = parser.add_argument_group("build helpers, only with -c")
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
    nargs="*",  # As many as present, 0 OK
    default="ALL",
    help="""Names of fortran routines for which Python C/API
                   functions will be generated. Default is all that are found
                   in <fortran files>.""",
)

parser.add_argument(
    "Fortran Files",
    metavar="<fortran files>",
    action="extend",  # List storage
    nargs="*",
    type=check_fortran,  # Returns a pathlib.Path
    help="""Paths to fortran/signature files that will be scanned for
                   <fortran functions> in order to determine their signatures.""",
)

parser.add_argument(
    "Skip functions",
    metavar="skip:",
    action="extend",
    type=str,
    nargs="*",
    help="Ignore fortran functions that follow until `:'.",
)

parser.add_argument(
    "Keep functions",
    metavar="only:",
    action="extend",
    type=str,
    nargs="*",
    help="Use only fortran functions that follow until `:'.",
)

# TODO: Remove?
parser.add_argument(
    "Fortran Files Again",
    metavar=":",
    action="extend",
    type=check_fortran,
    nargs="*",
    help="Get back to <fortran files> mode.",
)

parser.add_argument(
    "-m",
    "--module",
    metavar="<modulename>",
    type=str,
    nargs=1,
    default="untitled",
    help="""Name of the module; f2py generates a Python/C API
                   file <modulename>module.c or extension module <modulename>.
                   Default is 'untitled'.""",
)

parser.add_argument(
    "--lower",
    "--no-lower",
    metavar="--[no-]lower",
    action=BoolAction,
    default=True,
    type=bool,
    help="""Do [not] lower the cases in <fortran files>.
    By default, --lower is assumed with -h key, and --no-lower without -h
    key.""",
)

parser.add_argument(
    "-b",
    "--build-dir",
    metavar="<dirname>",
    type=check_dir,
    nargs=1,
    default="tempfile.mkdtemp()",
    help="""All f2py generated files are created in <dirname>.
                   Default is tempfile.mkdtemp().""",
)

parser.add_argument(
    "-o",
    "--overwrite-signature",
    action="store_true",
    help="Overwrite existing signature file.",
)

parser.add_argument(
    "--latex-doc",
    "--no-latex-doc",
    metavar="--[no-]latex-doc",
    action=BoolAction,
    type=bool,
    default=False,
    nargs=1,
    help="""Create (or not) <modulename>module.tex.
                   Default is --no-latex-doc.""",
)

parser.add_argument(
    "--short-latex",
    action="store_true",
    help="""Create 'incomplete' LaTeX document (without commands
                   \\documentclass, \\tableofcontents, and \\begin{{document}},
                   \\end{{document}}).""",
)

parser.add_argument(
    "--rest-doc",
    "--no-rest-doc",
    metavar="--[no-]rest-doc",
    action=BoolAction,
    type=bool,
    default=False,
    nargs=1,
    help="""Create (or not) <modulename>module.rst.
                   Default is --no-rest-doc.""",
)

parser.add_argument(
    "--debug-capi",
    action="store_true",
    help="""Create C/API code that reports the state of the wrappers
                   during runtime. Useful for debugging.""",
)

parser.add_argument(
    "--wrap-functions",
    "--no-wrap-functions",
    metavar="--[no-]wrap-functions",
    action=BoolAction,
    type=bool,
    default=False,
    nargs=1,
    help="""Create (or not) <modulename>module.rst.
                   Default is --no-rest-doc.""",
)

parser.add_argument(
    "--include-paths",
    metavar="<path1>:<path2>",
    action="extend",
    nargs="*",
    type=pathlib.Path,
    help="Search include files from the given directories.",
)

parser.add_argument(
    "--help-link",
    metavar="..",
    action="extend",
    nargs="*",
    choices=npd_link,
    type=str,
    help="""List system resources found by system_info.py. See also
            --link-<resource> switch below. [..] is optional list
            of resources names. E.g. try 'f2py --help-link lapack_opt'."""
)

parser.add_argument(
    "--f2cmap",
    metavar="<filename>",
    action="extend",
    nargs="*",
    type=pathlib.Path,
    default=".f2py_f2cmap",
    help="""Load Fortran-to-Python KIND specification from the given
                   file. Default: .f2py_f2cmap in current directory.""",
)

parser.add_argument(
    "--quiet",
    action="store_true",
    help="Run quietly.",
)

parser.add_argument(
    "--verbose",
    action="store_true",
    help="Run with extra verbosity.",
)

parser.add_argument(
    "-v",
    action="store_true",
    help="Print f2py version ID and exit.",
)

# Wrappers/Signatures #
#######################

parser.add_argument(
    # TODO: Seriously consider scrapping this naming convention
    "-h",
    "--hint-signature",
    metavar="<filename>",
    type=pathlib.Path,
    nargs=1,
    help="""
    Write signatures of the fortran routines to file <filename> and exit. You
    can then edit <filename> and use it instead of <fortran files>. If
    <filename>==stdout then the signatures are printed to stdout.
    """,
)

# NumPy Distutils #
###################

# TODO: Generalize to allow -c to take other build systems with numpy.distutils
# as a default
build_helpers.add_argument(
    "-c",
    default="numpy.distutils",
    action="store_true",
    help="Compilation (via NumPy distutils)"
)

build_helpers.add_argument(
    "--fcompiler",
    nargs=1,
    type=check_npfcomp,
    help="Specify Fortran compiler type by vendor"
)

build_helpers.add_argument(
    "--compiler",
    nargs=1,
    type=check_dccomp,
    help="Specify distutils C compiler type"
)

build_helpers.add_argument(
    "--help-fcompiler",
    action="store_true",
    help="List available Fortran compilers and exit"
)

build_helpers.add_argument(
    "--f77exec",
    nargs=1,
    type=pathlib.Path,
    help="Specify the path to a F77 compiler"
)

build_helpers.add_argument(
    "--f90exec",
    nargs=1,
    type=pathlib.Path,
    help="Specify the path to a F90 compiler"
)

build_helpers.add_argument(
    "--f77flags",
    nargs="*",
    type=pathlib.Path,
    action="extend",
    help="Specify F77 compiler flags"
)

build_helpers.add_argument(
    "--f90flags",
    nargs="*",
    type=pathlib.Path,
    action="extend",
    help="Specify F90 compiler flags"
)

build_helpers.add_argument(
    "--opt",
    "--optimization_flags",
    nargs="*",
    type=str,
    action="extend",
    help="Specify optimization flags"
)

build_helpers.add_argument(
    "--arch",
    "--architecture_optimizations",
    nargs="*",
    type=str,
    action="extend",
    help="Specify architecture specific optimization flags"
)

build_helpers.add_argument(
    "--noopt",
    action="store_true",
    help="Compile without optimization"
)

build_helpers.add_argument(
    "--noarch",
    action="store_true",
    help="Compile without arch-dependent optimization"
)

build_helpers.add_argument(
    "--debug",
    action="store_true",
    help="Compile with debugging information"
)

build_helpers.add_argument(
    "-L",
    "--library-path",
    type=pathlib.Path,
    metavar="/path/to/lib/",
    nargs="*",
    action="extend",
    help="Path to library"
)

build_helpers.add_argument(
    "-U",
    "--undef-macros",
    type=str,
    nargs="*",
    action="extend",
    help="Undefined macros"
)

build_helpers.add_argument(
    "-D",
    "--def-macros",
    type=str,
    nargs="*",
    action="extend",
    help="Defined macros"
)

build_helpers.add_argument(
    "-l",
    "--library_name",
    type=pathlib.Path,
    metavar="<libname>",
    nargs="*",
    action="extend",
    help="Library name"
)

build_helpers.add_argument(
    "-I",
    "--include_dirs",
    type=pathlib.Path,
    metavar="/path/to/include",
    nargs="*",
    action="extend",
    help="Include directories"
)

# TODO: Kill this ASAP
# Also collect in to REMAINDER and extract from there
build_helpers.add_argument(
    '--link-atlas', '--link-atlas_threads', '--link-atlas_blas',
    '--link-atlas_blas_threads', '--link-lapack_atlas',
    '--link-lapack_atlas_threads', '--link-atlas_3_10',
    '--link-atlas_3_10_threads', '--link-atlas_3_10_blas',
    '--link-atlas_3_10_blas_threadslapack_atlas_3_10',
    '--link-lapack_atlas_3_10_threads', '--link-flame', '--link-mkl',
    '--link-openblas', '--link-openblas_lapack', '--link-openblas_clapack',
    '--link-blis', '--link-lapack_mkl', '--link-blas_mkl', '--link-accelerate',
    '--link-openblas64_', '--link-openblas64__lapack', '--link-openblas_ilp64',
    '--link-openblas_ilp64_lapackx11', '--link-fft_opt', '--link-fftw',
    '--link-fftw2', '--link-fftw3', '--link-dfftw', '--link-sfftw',
    '--link-fftw_threads', '--link-dfftw_threads', '--link-sfftw_threads',
    '--link-djbfft', '--link-blas', '--link-lapack', '--link-lapack_src',
    '--link-blas_src', '--link-numpy', '--link-f2py', '--link-Numeric',
    '--link-numeric', '--link-numarray', '--link-numerix', '--link-lapack_opt',
    '--link-lapack_ilp64_opt', '--link-lapack_ilp64_plain_opt',
    '--link-lapack64__opt', '--link-blas_opt', '--link-blas_ilp64_opt',
    '--link-blas_ilp64_plain_opt', '--link-blas64__opt', '--link-boost_python',
    '--link-agg2', '--link-wx', '--link-gdk_pixbuf_xlib_2',
    '--link-gdk-pixbuf-xlib-2.0', '--link-gdk_pixbuf_2', '--link-gdk-pixbuf-2.0',
    '--link-gdk', '--link-gdk_2', '--link-gdk-2.0', '--link-gdk_x11_2',
    '--link-gdk-x11-2.0', '--link-gtkp_x11_2', '--link-gtk+-x11-2.0',
    '--link-gtkp_2', '--link-gtk+-2.0', '--link-xft', '--link-freetype2',
    '--link-umfpack', '--link-amd',
    metavar="--link-<resource>",
    dest="link_resource",
    nargs="*",
    action=NPDLinkHelper,
    help="The link helpers for numpy distutils"
)


# The rest, only works for files, since we expect:
#   <filename>.o <filename>.so <filename>.a
parser.add_argument('otherfiles',
                    type=pathlib.Path,
                    nargs=argparse.REMAINDER)

##################
# Parser Actions #
##################


def callcrackfortran(files, options):
    rules.options = options
    crackfortran.debug = options['debug']
    crackfortran.verbose = options['verbose']
    if 'module' in options:
        crackfortran.f77modulename = options['module']
    if 'skipfuncs' in options:
        crackfortran.skipfuncs = options['skipfuncs']
    if 'onlyfuncs' in options:
        crackfortran.onlyfuncs = options['onlyfuncs']
    crackfortran.include_paths[:] = options['include_paths']
    crackfortran.dolowercase = options['do-lower']
    postlist = crackfortran.crackfortran(files)
    if 'signsfile' in options:
        outmess('Saving signatures to file "%s"\n' % (options['signsfile']))
        pyf = crackfortran.crack2fortran(postlist)
        if options['signsfile'][-6:] == 'stdout':
            sys.stdout.write(pyf)
        else:
            with open(options['signsfile'], 'w') as f:
                f.write(pyf)
    if options["coutput"] is None:
        for mod in postlist:
            mod["coutput"] = "%smodule.c" % mod["name"]
    else:
        for mod in postlist:
            mod["coutput"] = options["coutput"]
    if options["f2py_wrapper_output"] is None:
        for mod in postlist:
            mod["f2py_wrapper_output"] = "%s-f2pywrappers.f" % mod["name"]
    else:
        for mod in postlist:
            mod["f2py_wrapper_output"] = options["f2py_wrapper_output"]
    return postlist

################
# Main Process #
################


def process_args(args):
    if args.help:
        parser.print_help()
    elif getattr(args, "Fortran Files"):
        print("BOOM")
        if args.c:
            if args.fcompiler:
                print(f"got {args.fcompiler}")
            elif args.compiler:
                print(args.compiler)
            else:
                print("Compilation requested without options, using defaults")
    else:
        parser.print_usage()


def main():
    logger = logging.getLogger("f2py_cli")
    logger.setLevel(logging.WARNING)
    args = parser.parse_args()
    process_args(args)


if __name__ == "__main__":
    main()
