import numpy as np
from numpy.testing import assert_array_equal, assert_

import pytest

@pytest.mark.filterwarnings("ignore:the matrix subclass is not")
@pytest.mark.filterwarnings("ignore:Importing from numpy.matlib")
def test_empty():
    import numpy.matlib
    x = numpy.matlib.empty((2,))
    assert_(isinstance(x, np.matrix))
    assert_(x.shape, (1, 2))

@pytest.mark.filterwarnings("ignore:the matrix subclass is not")
def test_ones():
    import numpy.matlib
    assert_array_equal(numpy.matlib.ones((2, 3)),
                       np.matrix([[ 1.,  1.,  1.],
                                 [ 1.,  1.,  1.]]))

    assert_array_equal(numpy.matlib.ones(2), np.matrix([[ 1.,  1.]]))

@pytest.mark.filterwarnings("ignore:the matrix subclass is not")
def test_zeros():
    import numpy.matlib
    assert_array_equal(numpy.matlib.zeros((2, 3)),
                       np.matrix([[ 0.,  0.,  0.],
                                 [ 0.,  0.,  0.]]))

    assert_array_equal(numpy.matlib.zeros(2), np.matrix([[ 0.,  0.]]))

@pytest.mark.filterwarnings("ignore:the matrix subclass is not")
def test_identity():
    import numpy.matlib
    x = numpy.matlib.identity(2, dtype=int)
    assert_array_equal(x, np.matrix([[1, 0], [0, 1]]))

@pytest.mark.filterwarnings("ignore:the matrix subclass is not")
def test_eye():
    import numpy.matlib
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

@pytest.mark.filterwarnings("ignore:the matrix subclass is not")
def test_rand():
    import numpy.matlib
    x = numpy.matlib.rand(3)
    # check matrix type, array would have shape (3,)
    assert_(x.ndim == 2)

@pytest.mark.filterwarnings("ignore:the matrix subclass is not")
def test_randn():
    import numpy.matlib
    x = np.matlib.randn(3)
    # check matrix type, array would have shape (3,)
    assert_(x.ndim == 2)

def test_repmat():
    import numpy.matlib
    a1 = np.arange(4)
    x = numpy.matlib.repmat(a1, 2, 2)
    y = np.array([[0, 1, 2, 3, 0, 1, 2, 3],
                  [0, 1, 2, 3, 0, 1, 2, 3]])
    assert_array_equal(x, y)
