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

import os
import argparse
import logging
import pathlib
import tempfile

from numpy.version import version as __version__

from .service import check_dccomp, check_npfcomp, check_dir, generate_files, segregate_files, get_f2py_modulename, wrapper_settings, compile_dist
from .utils import open_build_dir

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

debug_api = ['capi']


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

    def __call__(self, parser, namespace, values, option_string: str=None):
        """The logical negation action

        Essentially this returns the semantically valid operation implied by
        --no
        """
        setattr(namespace, self.dest, "no" not in option_string)


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
            items.append(outvar)
            setattr(namespace, self.dest, items)
        else:
            raise RuntimeError(f"{outvar} is not in {npd_link}")

class DebugLinkHelper(argparse.Action):
    """A custom action to work with f2py's --debug-blah"""

    def __call__(self, parser, namespace, values, option_string=None):
        """The storage action

        Essentially, split the value on -, store in dest

        """
        items = getattr(namespace, self.dest) or []
        outvar = option_string.split("--debug-")[1]
        if outvar in debug_api:
            items.append(outvar)
            setattr(namespace, self.dest, items)
        else:
            raise RuntimeError(f"{outvar} is not in {debug_api}")

class ProcessMacros(argparse.Action):
    """Process macros in the form of --Dmacro=value and -Dmacro"""

    def __call__(self, parser, namespace, values, option_string=None):
        """The storage action

        Essentially, split the value on -D, store in dest

        """
        items = getattr(namespace, self.dest) or []
        outvar = option_string.split("-D")[1]
        if('=' in outvar):
            items.append((outvar.split("=")[0], outvar.split("=")[1]))
        else:
            items.append((outvar, None))
        setattr(namespace, self.dest, items)


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
    "Fortran Files",
    metavar="<fortran files>",
    action="extend",  # List storage
    nargs="*",
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

parser.add_argument(
    "-m",
    "--module",
    metavar="<modulename>",
    type=str,
    nargs=1,
    help="""Name of the module; f2py generates a Python/C API
                   file <modulename>module.c or extension module <modulename>.
                   Default is 'untitled'.""",
)

parser.add_argument(
    "--lower",
    "--no-lower",
    metavar="--[no-]lower",
    action=BoolAction,
    default=False,
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
    dest="debug_api",
    default=[],
    nargs="*",
    action=DebugLinkHelper,
    help="""Create C/API code that reports the state of the wrappers
                   during runtime. Useful for debugging.""",
)

parser.add_argument(
    "--wrap-functions",
    "--no-wrap-functions",
    metavar="--[no-]wrap-functions",
    action=BoolAction,
    type=bool,
    default=True,
    nargs=1,
    help="""Create (or not) Fortran subroutine wrappers to Fortran 77
                   functions. Default is --wrap-functions because it
                   ensures maximum portability/compiler independence""",
)

parser.add_argument(
    "--include-paths",
    metavar="<path1>:<path2>",
    action="extend",
    default=[],
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
    default=True,
    help="Run with extra verbosity.",
)

parser.add_argument(
    "-v",
    action="store_true",
    help="Print f2py version ID and exit.",
)

# Wrappers/Signatures #
#######################

generate_wrappers.add_argument(
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
    default=False,
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
    action="extend",
    help="Specify F77 compiler flags"
)

build_helpers.add_argument(
    "--f90flags",
    nargs="*",
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
    type=str,
    nargs="*",
    action="extend",
    dest='undef_macros',
    help="Undefined macros"
)

build_helpers.add_argument(
    "-D",
    type=str,
    nargs="*",
    action=ProcessMacros,
    dest='define_macros',
    help="Define macros"
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
# Flag not working. To be debugged.
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
    default=[],
    nargs="*",
    action=NPDLinkHelper,
    help="The link helpers for numpy distutils"
)


# The rest, only works for files, since we expect:
#   <filename>.o <filename>.so <filename>.a
parser.add_argument('otherfiles',
                    type=pathlib.Path,
                    nargs=argparse.REMAINDER)


################
# Main Process #
################

def get_additional_headers(rem):
    return [val[8:] for val in rem if val[:8] == '-include']

def get_f2pyflags_dist(args, skip_funcs, only_funcs, rem):
    # Distutils requires 'f2py_options' which will be a subset of
    # sys.argv array received. This function reconstructs the array
    # from received args.
    f2py_flags = []
    if(args.wrap_functions):
        f2py_flags.append('--wrap-functions')
    else:
        f2py_flags.append('--no-wrap-functions')
    if(args.lower):
        f2py_flags.append('--lower')
    else:
        f2py_flags.append('--no-lower')
    if(args.debug_api):
        f2py_flags.append('--debug-capi')
    if(args.quiet):
        f2py_flags.append('--quiet')
    f2py_flags.append("--skip-empty-wrappers")
    if(skip_funcs):
        f2py_flags.extend(['skip:']+skip_funcs + [':'])
    if(only_funcs):
        f2py_flags.extend(['only:']+only_funcs + [':'])
    if(args.include_paths):
        f2py_flags.extend(['--include-paths']+args.include_paths)
    return f2py_flags

def get_fortran_library_flags(args):
    flib_flags = []
    if args.fcompiler:
        flib_flags.append(f'--fcompiler={args.fcompiler[0]}')
    if args.compiler:
        flib_flags.append(f'--compiler={args.compiler[0]}')
    return flib_flags

def get_fortran_compiler_flags(args):
    fc_flags = []
    if(args.help_fcompiler):
        fc_flags.append('--help-fcompiler')
    if(args.f77exec):
        fc_flags.append(f'--f77exec={str(args.f77exec[0])}')
    if(args.f90exec):
        fc_flags.append(f'--f90exec={str(args.f90exec[0])}')
    if(args.f77flags):
        fc_flags.append(f'--f77flags={" ".join(args.f77flags)}')
    if(args.f90flags):
        fc_flags.append(f'--f90flags={" ".join(args.f90flags)}')
    if(args.arch):
        fc_flags.append(f'--arch={" ".join(args.arch)}')
    if(args.opt):
        fc_flags.append(f'--opt={" ".join(args.opt)}')
    if(args.noopt):
        fc_flags.append('--noopt')
    if(args.noarch):
        fc_flags.append('--noarch')
    if(args.debug):
        fc_flags.append('--debug')

def get_build_dir(args):
    if(args.build_dir is not None):
        return args.build_dir[0]
    if(args.c):
        return tempfile.mkdtemp()
    return pathlib.Path.cwd()

def get_module_name(args, pyf_files):
    if(args.module is not None):
        return args.module[0]
    if args.c:
        for file in pyf_files:
            if name := get_f2py_modulename(file):
                return name
        return "unititled"
    return None

def get_signature_file(args, build_dir):
    sign_file = None
    if(args.hint_signature):
        sign_file = build_dir /  args.hint_signature[0]
        if sign_file and os.path.isfile(sign_file) and not args.overwrite_signature:
            print(f'Signature file "{sign_file}" exists!!! Use --overwrite-signature to overwrite.')
            parser.exit()
    return sign_file

def segregate_posn_args(args):
    # Currently, argparse does not recognise 'skip:' and 'only:' as optional args
    # and clubs them all in "Fortran Files" attr. This function segregates them.
    funcs = {"skip:": [], "only:": []}
    mode = "file"
    files = []
    for arg in getattr(args, "Fortran Files"):
        if arg in funcs:
            mode = arg
        elif arg == ':' and mode in funcs:
            mode = "file"
        elif mode == "file":
            files.append(arg)
        else:
            funcs[mode].append(arg)
    return files, funcs['skip:'], funcs['only:']

def process_args(args, rem):
    if args.help:
        parser.print_help()
        parser.exit()
    files, skip_funcs, only_funcs = segregate_posn_args(args)
    f77_files, f90_files, pyf_files, obj_files, other_files = segregate_files(files)

    with open_build_dir(args.build_dir, args.c) as build_dir:
        module_name = get_module_name(args, pyf_files)
        sign_file = get_signature_file(args, build_dir)
        headers = get_additional_headers(rem)
        # TODO: Refine rules settings. Read codebase and remove unused ones
        rules_setts = {
            'module': module_name,
            'buildpath': build_dir,
            'dorestdoc': args.rest_doc,
            'dolatexdoc': args.latex_doc,
            'shortlatex': args.short_latex,
            'verbose': args.verbose,
            'do-lower': args.lower,
            'f2cmap_file': args.f2cmap,
            'include_paths': args.include_paths,
            'coutput': None,
            'f2py_wrapper_output': None,
            'emptygen': True,
        }
        crackfortran_setts = {
            'module': module_name,
            'skipfuncs': skip_funcs,
            'onlyfuncs': only_funcs,
            'verbose': args.verbose,
            'include_paths': args.include_paths,
            'do-lower': args.lower,
            'debug': args.debug_api,
            'wrapfuncs': args.wrap_functions,
        }
        capi_maps_setts = {
            'f2cmap': args.f2cmap,
            'headers': headers,
        }
        auxfuncs_setts = {
            'verbose': args.verbose,
            'debug': args.debug_api,
            'wrapfuncs': args.wrap_functions,
        }

        wrapper_settings(rules_setts, crackfortran_setts, capi_maps_setts, auxfuncs_setts)

        # Disutils receives all the options and builds the extension.
        if(args.c):
            link_resource = args.link_resource
            f2py_flags = get_f2pyflags_dist(args, skip_funcs, only_funcs, rem)
            fc_flags = get_fortran_compiler_flags(args)
            flib_flags = get_fortran_library_flags(args)
            
            ext_args = {
                'name': module_name,
                'sources': pyf_files + f77_files + f90_files,
                'include_dirs': args.include_dirs,
                'library_dirs': args.library_path,
                'libraries': args.library_name,
                'define_macros': args.define_macros,
                'undef_macros': args.undef_macros,
                'extra_objects': obj_files,
                'f2py_options': f2py_flags,
            }
            compile_dist(ext_args, link_resource, build_dir, fc_flags, flib_flags, args.quiet)
        else:
            generate_files(f77_files + f90_files, module_name, sign_file)

def main():
    logger = logging.getLogger("f2py_cli")
    logger.setLevel(logging.WARNING)
    args, rem = parser.parse_known_args()
    # since argparse can't handle '-include<header>'
    # we filter it out into rem and parse it manually.
    process_args(args, rem)

if __name__ == "__main__":
    main()
