from core.numpyenv import GetNumpyEnvironment, GetNumpyOptions
from core.libinfo_scons import NumpyCheckLib
from core.libinfo import get_paths as scons_get_paths
from core.extension import get_python_inc, get_pythonlib_dir
from core.utils import isstring, rsplit

from checkers import CheckCBLAS, CheckLAPACK
from fortran_scons import CheckF77Verbose, CheckF77Clib, CheckF77Mangling

def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)
