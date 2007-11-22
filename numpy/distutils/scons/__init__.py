from core.numpyenv import GetNumpyEnvironment, GetNumpyOptions
from core.libinfo_scons import NumpyCheckLibAndHeader
from core.libinfo import get_paths as scons_get_paths
from core.extension import get_python_inc, get_pythonlib_dir
from core.utils import isstring, rsplit

from checkers import CheckCBLAS, CheckCLAPACK, CheckF77BLAS, CheckF77LAPACK, \
                     IsMKL, IsATLAS, IsVeclib, IsAccelerate, IsSunperf
from checkers.perflib import GetATLASVersion, GetMKLVersion

from fortran_scons import CheckF77Verbose, CheckF77Clib, CheckF77Mangling

# XXX: this is ugly, better find the mathlibs with a checker 
# XXX: this had nothing to do here, too...
def scons_get_mathlib(env):
    from numpy.distutils.misc_util import get_mathlibs
    path_list = scons_get_paths(env['include_bootstrap']) + [None]
    for i in path_list:
        try:
            mlib =  get_mathlibs(i)
            return mlib
        except IOError:
            pass
    raise RuntimeError("FIXME: no mlib found ?")

def test(level=1, verbosity=1):
    from numpy.testing import NumpyTest
    return NumpyTest().test(level, verbosity)
