from __future__ import division, absolute_import, print_function

from .info import __doc__
from numpy.version import version as __version__

# disables OpenBLAS affinity setting of the main thread that limits
# python threads or processes to one core
import os
env_added = []
for envkey in ['OPENBLAS_MAIN_FREE', 'GOTOBLAS_MAIN_FREE']:
    if envkey not in os.environ:
        os.environ[envkey] = '1'
        env_added.append(envkey)

try:
    from . import multiarray
except ImportError as exc:
    import sys
    msg = """

IMPORTANT: PLEASE READ THIS FOR ADVICE ON HOW TO SOLVE THIS ISSUE!

Importing the multiarray numpy extension module failed.  Most
likely you are trying to import a failed build of numpy.
Here is how to proceed:
- If you're working with a numpy git repository, try `git clean -xdf`
  (removes all files not under version control) and rebuild numpy.
- If you are simply trying to use the numpy version that you have installed:
  your installation is broken - please reinstall numpy.
- If you have already reinstalled and that did not fix the problem, then:
  1. Check that you are using the Python you expect (you're using %s),
     and that you have no directories in your PATH or PYTHONPATH that can
     interfere with the Python and numpy versions you're trying to use.
  2. If (1) looks fine, you can open a new issue at
     https://github.com/numpy/numpy/issues.  Please include details on:
     - how you installed Python
     - how you installed numpy
     - your operating system
     - whether or not you have multiple versions of Python installed
     - if you built from source, your compiler versions and ideally a build log

     Note: this error has many possible causes, so please don't comment on
     an existing issue about this - open a new one instead.

Original error was: %s
""" % (sys.executable, exc)
    raise ImportError(msg)
finally:
    for envkey in env_added:
        del os.environ[envkey]
del envkey
del env_added
del os

############### HACK for broken installations #########################

# Test that multiarray is a pure python module wrapping _multiarray_umath,
# and not the old c-extension module.
msg = ("Something is wrong with the NumPy installation. "
       "While importing 'multiarray' we detected an older version of "
       "numpy. One method of fixing this is to repeatedly 'pip uninstall' "
       "numpy until none is found, then reinstall this version.")
if not getattr(multiarray, '_multiarray_umath', None):
    # Old multiarray. Can we override it?
    import sys, os.path as osp
    def get_params():
        for name in ('multiarray', 'umath'):
            modname = "numpy.core.{}".format(name)
            fname = osp.join(osp.dirname(__file__), '{}.py'.format(name))
            # try py, pyc, pyo
            for ext in ('', 'c', 'o'):
                if osp.exists(fname + ext):
                    fname += ext
                    break
            yield modname, fname, name

    if sys.version_info >= (3,5):
        import importlib.util
        try:
            for modname, fname, name in get_params():
                spec = importlib.util.spec_from_file_location(modname, fname)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                sys.modules[modname] = mod
                locals()[name] = mod
        except Exception as e:
            print(e)
            raise ImportError(msg)
    elif sys.version_info[:2] == (2, 7):
        import imp
        try:
            for modname, fname, name in get_params():
                mod = imp.load_source(modname, fname)
                sys.modules[modname] = mod
                locals()[name] = mod
        except Exception as e:
            print(e)
            raise ImportError(msg)
    else:
        raise ImportError(msg)
    del sys, osp
    import warnings
    warnings.warn(msg, ImportWarning, stacklevel=1)

# when this HACK is removed, keep this line
from . import umath

if not getattr(umath, '_multiarray_umath', None):
    # The games in the previous block failed. Give up.
    # The warning above may already have been emitted.
    raise ImportError(msg)

del msg
############### end of HACK for broken installations #######################

from . import _internal  # for freeze programs
from . import numerictypes as nt
multiarray.set_typeDict(nt.sctypeDict)
from . import numeric
from .numeric import *
from . import fromnumeric
from .fromnumeric import *
from . import defchararray as char
from . import records as rec
from .records import *
from .memmap import *
from .defchararray import chararray
from . import function_base
from .function_base import *
from . import machar
from .machar import *
from . import getlimits
from .getlimits import *
from . import shape_base
from .shape_base import *
from . import einsumfunc
from .einsumfunc import *
del nt

from .fromnumeric import amax as max, amin as min, round_ as round
from .numeric import absolute as abs

# do this after everything else, to minimize the chance of this misleadingly
# appearing in an import-time traceback
from . import _add_newdocs

__all__ = ['char', 'rec', 'memmap']
__all__ += numeric.__all__
__all__ += fromnumeric.__all__
__all__ += rec.__all__
__all__ += ['chararray']
__all__ += function_base.__all__
__all__ += machar.__all__
__all__ += getlimits.__all__
__all__ += shape_base.__all__
__all__ += einsumfunc.__all__

# Make it possible so that ufuncs can be pickled
#  Here are the loading and unloading functions
# The name numpy.core._ufunc_reconstruct must be
#   available for unpickling to work.
def _ufunc_reconstruct(module, name):
    # The `fromlist` kwarg is required to ensure that `mod` points to the
    # inner-most module rather than the parent package when module name is
    # nested. This makes it possible to pickle non-toplevel ufuncs such as
    # scipy.special.expit for instance.
    mod = __import__(module, fromlist=[name])
    return getattr(mod, name)

def _ufunc_reduce(func):
    from pickle import whichmodule
    name = func.__name__
    return _ufunc_reconstruct, (whichmodule(func, name), name)


import sys
if sys.version_info[0] >= 3:
    import copyreg
else:
    import copy_reg as copyreg

copyreg.pickle(ufunc, _ufunc_reduce, _ufunc_reconstruct)
# Unclutter namespace (must keep _ufunc_reconstruct for unpickling)
del copyreg
del sys
del _ufunc_reduce

from numpy._pytesttester import PytestTester
test = PytestTester(__name__)
del PytestTester
