"""Fortran to Python Interface Generator."""

__all__ = ['run_main', 'compile', 'get_include']

import sys

from numpy.f2py import frontend
from numpy.f2py import __version__
# Helpers
from numpy.f2py.utils.pathhelper import get_include
# Build tool
from numpy.f2py.utils.npdist import compile
from numpy.f2py.frontend.f2py2e import main

run_main = frontend.f2py2e.run_main

if __name__ == "__main__":
    sys.exit(main())

def __getattr__(attr):

    # Avoid importing things that aren't needed for building
    # which might import the main numpy module
    if attr == "test":
        from numpy.f2py import f2py_testing
        from numpy._pytesttester import PytestTester
        test = PytestTester(__name__)
        return test

    else:
        raise AttributeError("module {!r} has no attribute "
                              "{!r}".format(__name__, attr))


def __dir__():
    return list(globals().keys() | {"test"})

# Deprecated modules
# These modules have been moved, and will throw warnings when used
# They will be removed by the deadline
# The order is from dir(numpy.f2py)
from numpy.f2py.utils.deprecation_helpers import deprecated_submodule
deprecated_submodule(new_module_name = "numpy.f2py.stds.auxfuncs",
                     old_parent = "numpy.f2py",
                     old_child = "auxfuncs",
                     deadline = "1.25")
deprecated_submodule(new_module_name = "numpy.f2py.stds.pyf.capi_maps",
                     old_parent = "numpy.f2py",
                     old_child = "capi_maps",
                     deadline = "1.25")
deprecated_submodule(new_module_name = "numpy.f2py.stds.pyf.cb_rules",
                     old_parent = "numpy.f2py",
                     old_child = "cb_rules",
                     deadline = "1.25")
deprecated_submodule(new_module_name = "numpy.f2py.codegen.cfuncs",
                     old_parent = "numpy.f2py",
                     old_child = "cfuncs",
                     deadline = "1.25")
deprecated_submodule(new_module_name = "numpy.f2py.stds.f77.common_rules",
                     old_parent = "numpy.f2py",
                     old_child = "common_rules",
                     deadline = "1.25")
deprecated_submodule(new_module_name = "numpy.f2py.frontend.crackfortran",
                     old_parent = "numpy.f2py",
                     old_child = "crackfortran",
                     deadline = "1.25")
deprecated_submodule(new_module_name = "numpy.f2py.utils.diagnose",
                     old_parent = "numpy.f2py",
                     old_child = "diagnose",
                     deadline = "1.25")
deprecated_submodule(new_module_name = "numpy.f2py.frontend.f2py2e",
                     old_parent = "numpy.f2py",
                     old_child = "f2py2e",
                     deadline = "1.25")
deprecated_submodule(new_module_name = "numpy.f2py.stds.f90.f90mod_rules",
                     old_parent = "numpy.f2py",
                     old_child = "f90mod_rules",
                     deadline = "1.25")
deprecated_submodule(new_module_name = "numpy.f2py.codegen.func2subr",
                     old_parent = "numpy.f2py",
                     old_child = "func2subr",
                     deadline = "1.25")
deprecated_submodule(new_module_name = "numpy.f2py.codegen.rules",
                     old_parent = "numpy.f2py",
                     old_child = "rules",
                     deadline = "1.25")
deprecated_submodule(new_module_name = "numpy.f2py.stds.f90.use_rules",
                     old_parent = "numpy.f2py",
                     old_child = "use_rules",
                     deadline = "1.25")
