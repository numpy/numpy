from numpyenv import GetNumpyEnvironment, GetNumpyOptions
from libinfo_scons import NumpyCheckLib
from libinfo import get_paths as scons_get_paths
from custom_checkers import CheckMKL, CheckATLAS, CheckCBLAS, \
        CheckAccelerate, CheckMKL, CheckSunperf, CheckLAPACK, \
        CheckNetlibBLAS, CheckNetlibLAPACK
from extension import get_python_inc, get_pythonlib_dir
from utils import isstring
from fortran_scons import CheckF77Verbose, CheckF77Clib, CheckF77Mangling

def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)
