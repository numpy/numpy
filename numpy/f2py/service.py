import sys
import os
import logging
import tempfile

from pathlib import Path
from typing import Any, Dict, List, Tuple
from argparse import Namespace

from . import crackfortran

logger = logging.getLogger("f2py_cli")
logger.setLevel(logging.WARNING)

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
    fpname = Path(fname)
    if fpname.suffix.lower() not in [".f90", ".f", ".f77"]:
        logger.warning(
            """Does not look like a standard fortran file ending in *.f90, *.f or
            *.f77, continuing against better judgement"""
        )
    return fpname


def check_dir(dname: str):
    """Function which checks the build directory

    This is meant to ensure no odd directories are passed, it will fail if a
    file is passed. Creates directory if not present.

    Parameters
    ----------
    dname : str
        The name of the directory, by default it will be a temporary one

    Returns
    -------
    pathlib.Path
        This is the string as a path
    """
    if dname:
        dpname = Path(dname)
        dpname.mkdir(parents=True, exist_ok=True)
        return dpname


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



def segregate_files(files: List[Path]) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
	"""
	Segregate files into three groups:
	* .f files
	* .o files
	* others
	"""
	f77_ext = ('.f', '.for', '.ftn', '.f77')
	f90_ext = ('.f90', '.f95', '.f03', '.f08')
	out_ext = ('.o', '.out', '.so', '.a')

	f77_files = []
	f90_files = []
	out_files = []
	other_files = []

	for f in files:
		ext = os.path.splitext(f)[1]
		if ext in f77_ext:
			f77_files.append(f)
		elif ext in f90_ext:
			f90_files.append(f)
		elif ext in out_ext:
			out_files.append(f)
		else:
			other_files.append(f)

	return f77_files, f90_files, out_files, other_files