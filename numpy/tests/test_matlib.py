import numpy as np
import numpy.matlib
from numpy.testing import assert_array_equal, assert_
import pytest

def test_empty():
    x = numpy.matlib.empty((2,))
    assert_(isinstance(x, np.matrix))
    assert_(x.shape, (1, 2))

def test_ones():
    # valid calls
    assert_array_equal(numpy.matlib.ones((2, 3)),
                       np.matrix([[ 1.,  1.,  1.],
                                 [ 1.,  1.,  1.]]))

    assert_array_equal(numpy.matlib.ones(2), np.matrix([[ 1.,  1.]]))

    # special cases: 0
    assert_array_equal(numpy.matlib.ones((2, 0)),
                       np.matrix([[],
                                 []]))

    assert_array_equal(numpy.matlib.ones(0), np.matrix([]))

    # invalid calls: negative values
    with pytest.raises(ValueError) as exc_info:
        numpy.matlib.ones((2, -1))

    with pytest.raises(ValueError) as exc_info:
        numpy.matlib.ones(-1)

    # invalid calls: wrong types
    with pytest.raises(TypeError) as exc_info:
        numpy.matlib.ones((2, 3.5))

    with pytest.raises(TypeError) as exc_info:
        numpy.matlib.ones(2.5)

def test_zeros():
    # valid calls
    assert_array_equal(numpy.matlib.zeros((2, 3)),
                       np.matrix([[ 0.,  0.,  0.],
                                 [ 0.,  0.,  0.]]))

    assert_array_equal(numpy.matlib.zeros(2), np.matrix([[ 0.,  0.]]))

    # special cases: 0
    assert_array_equal(numpy.matlib.zeros((2, 0)),
                       np.matrix([[],
                                 []]))

    assert_array_equal(numpy.matlib.zeros(0), np.matrix([]))

    # invalid calls: negative values
    with pytest.raises(ValueError) as exc_info:
        numpy.matlib.zeros((2, -1))

    with pytest.raises(ValueError) as exc_info:
        numpy.matlib.zeros(-1)

    # invalid calls: wrong types
    with pytest.raises(TypeError) as exc_info:
        numpy.matlib.zeros((2, 3.5))

    with pytest.raises(TypeError) as exc_info:
        numpy.matlib.zeros(2.5)

def test_identity():
    x = numpy.matlib.identity(2, dtype=int)
    assert_array_equal(x, np.matrix([[1, 0], [0, 1]]))

def test_eye():
    xc = numpy.matlib.eye(3, k=1, dtype=int)
    assert_array_equal(xc, np.matrix([[ 0,  1,  0],
                                      [ 0,  0,  1],
                                      [ 0,  0,  0]]))
    assert xc.flags.c_contiguous
    assert not xc.flags.f_contiguous

    xf = numpy.matlib.eye(3, 4, dtype=int, order='F')
    assert_array_equal(xf, np.matrix([[ 1,  0,  0,  0],
                                      [ 0,  1,  0,  0],
                                      [ 0,  0,  1,  0]]))
    assert not xf.flags.c_contiguous
    assert xf.flags.f_contiguous

def test_rand():
    x = numpy.matlib.rand(3)
    # check matrix type, array would have shape (3,)
    assert_(x.ndim == 2)

def test_randn():
    x = np.matlib.randn(3)
    # check matrix type, array would have shape (3,)
    assert_(x.ndim == 2)

def test_repmat():
    a1 = np.arange(4)
    x = numpy.matlib.repmat(a1, 2, 2)
    y = np.array([[0, 1, 2, 3, 0, 1, 2, 3],
                  [0, 1, 2, 3, 0, 1, 2, 3]])
    assert_array_equal(x, y)
