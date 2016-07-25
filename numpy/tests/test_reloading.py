from __future__ import division, absolute_import, print_function

import sys

import numpy as np
from numpy.testing import assert_raises, assert_, run_module_suite

if sys.version_info[:2] >= (3, 4):
    from importlib import reload
else:
    from imp import reload

def test_reloading_exception():
    # gh-7844. Also check that relevant globals retain their identity.
    _NoValue = np._NoValue
    VisibleDeprecationWarning = np.VisibleDeprecationWarning
    ModuleDeprecationWarning = np.ModuleDeprecationWarning

    assert_raises(RuntimeError, reload, np)
    assert_(_NoValue is np._NoValue)
    assert_(ModuleDeprecationWarning is np.ModuleDeprecationWarning)
    assert_(VisibleDeprecationWarning is np.VisibleDeprecationWarning)


if __name__ == "__main__":
    run_module_suite()
