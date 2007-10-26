from numpyenv import GetNumpyEnvironment, GetNumpyOptions
from libinfo_scons import NumpyCheckLib
from libinfo import get_paths as scons_get_paths
from custom_checkers import CheckMKL, CheckATLAS, CheckCBLAS, \
        CheckAccelerate, CheckMKL, CheckSunperf
from extension import get_python_inc, get_pythonlib_dir
from utils import isstring
import fortran

def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)
