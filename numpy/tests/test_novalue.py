from __future__ import division, absolute_import, print_function

import sys

from numpy.testing import (assert_raises, assert_, run_module_suite,
    assert_equal)

if sys.version_info[:2] >= (3, 4):
    from importlib import reload
else:
    from imp import reload

def test_numpy_reloading():
    # gh-7844. Also check that relevant globals retain their identity.
    import numpy as np
    import numpy._globals

    _NoValue = np._NoValue
    VisibleDeprecationWarning = np.VisibleDeprecationWarning
    ModuleDeprecationWarning = np.ModuleDeprecationWarning

    reload(np)
    assert_(_NoValue is np._NoValue)
    assert_(ModuleDeprecationWarning is np.ModuleDeprecationWarning)
    assert_(VisibleDeprecationWarning is np.VisibleDeprecationWarning)

    assert_raises(RuntimeError, reload, numpy._globals)
    reload(np)
    assert_(_NoValue is np._NoValue)
    assert_(ModuleDeprecationWarning is np.ModuleDeprecationWarning)
    assert_(VisibleDeprecationWarning is np.VisibleDeprecationWarning)

def test_novalue_repr():
    # test repr of np._NoValue, see #8991
    import numpy as np
    assert_equal(repr(np._NoValue), 'np._NoValue')


if __name__ == "__main__":
    run_module_suite()
