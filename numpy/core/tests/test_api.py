import sys

import numpy as np
from numpy.testing import *
from numpy.testing.utils import WarningManager
import warnings

def test_fastCopyAndTranspose():
    # 0D array
    a = np.array(2)
    b = np.fastCopyAndTranspose(a)
    assert_equal(b, a.T)
    assert_(b.flags.owndata)

    # 1D array
    a = np.array([3,2,7,0])
    b = np.fastCopyAndTranspose(a)
    assert_equal(b, a.T)
    assert_(b.flags.owndata)

    # 2D array
    a = np.arange(6).reshape(2,3)
    b = np.fastCopyAndTranspose(a)
    assert_equal(b, a.T)
    assert_(b.flags.owndata)

if __name__ == "__main__":
    run_module_suite()
