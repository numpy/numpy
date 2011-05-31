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

def test_array_astype():
    a = np.arange(6, dtype='f4').reshape(2,3)
    # Default behavior: allows unsafe casts, keeps memory layout,
    #                   always copies.
    b = a.astype('i4')
    assert_equal(a, b)
    assert_equal(b.dtype, np.dtype('i4'))
    assert_equal(a.strides, b.strides)
    b = a.T.astype('i4')
    assert_equal(a.T, b)
    assert_equal(b.dtype, np.dtype('i4'))
    assert_equal(a.T.strides, b.strides)
    b = a.astype('f4')
    assert_equal(a, b)
    assert_(not (a is b))

    # forcecopy=False parameter can sometimes skip a copy
    b = a.astype('f4', forcecopy=False)
    assert_(a is b)

    # order parameter allows overriding of the memory layout,
    # forcing a copy if the layout is wrong
    b = a.astype('f4', order='F', forcecopy=False)
    assert_equal(a, b)
    assert_(not (a is b))
    assert_(b.flags.f_contiguous)

    b = a.astype('f4', order='C', forcecopy=False)
    assert_equal(a, b)
    assert_(a is b)
    assert_(b.flags.c_contiguous)

    # casting parameter allows catching bad casts
    b = a.astype('c8', casting='safe')
    assert_equal(a, b)
    assert_equal(b.dtype, np.dtype('c8'))

    assert_raises(TypeError, a.astype, 'i4', casting='safe')

    # subok=False passes through a non-subclassed array
    b = a.astype('f4', subok=0, forcecopy=False)
    assert_(a is b)

    a = np.matrix([[0,1,2],[3,4,5]], dtype='f4')

    # subok=True passes through a matrix
    b = a.astype('f4', subok=True, forcecopy=False)
    assert_(a is b)

    # subok=True is default, and creates a subtype on a cast
    b = a.astype('i4', forcecopy=False)
    assert_equal(a, b)
    assert_equal(type(b), np.matrix)

    # subok=False never returns a matrix
    b = a.astype('f4', subok=False, forcecopy=False)
    assert_equal(a, b)
    assert_(not (a is b))
    assert_(type(b) != np.matrix)

if __name__ == "__main__":
    run_module_suite()
