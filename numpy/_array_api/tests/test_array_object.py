from numpy.testing import assert_raises
import numpy as np

from .. import ones, asarray

def test_validate_index():
    # The indexing tests in the official array API test suite test that the
    # array object correctly handles the subset of indices that are required
    # by the spec. But the NumPy array API implementation specifically
    # disallows any index not required by the spec, via Array._validate_index.
    # This test focuses on testing that non-valid indices are correctly
    # rejected. See
    # https://data-apis.org/array-api/latest/API_specification/indexing.html
    # and the docstring of Array._validate_index for the exact indexing
    # behavior that should be allowed. This does not test indices that are
    # already invalid in NumPy itself because Array will generally just pass
    # such indices directly to the underlying np.ndarray.

    a = ones((3, 4))

    # Out of bounds slices are not allowed
    assert_raises(IndexError, lambda: a[:4])
    assert_raises(IndexError, lambda: a[:-4])
    assert_raises(IndexError, lambda: a[:3:-1])
    assert_raises(IndexError, lambda: a[:-5:-1])
    assert_raises(IndexError, lambda: a[3:])
    assert_raises(IndexError, lambda: a[-4:])
    assert_raises(IndexError, lambda: a[3::-1])
    assert_raises(IndexError, lambda: a[-4::-1])

    assert_raises(IndexError, lambda: a[...,:5])
    assert_raises(IndexError, lambda: a[...,:-5])
    assert_raises(IndexError, lambda: a[...,:4:-1])
    assert_raises(IndexError, lambda: a[...,:-6:-1])
    assert_raises(IndexError, lambda: a[...,4:])
    assert_raises(IndexError, lambda: a[...,-5:])
    assert_raises(IndexError, lambda: a[...,4::-1])
    assert_raises(IndexError, lambda: a[...,-5::-1])

    # Boolean indices cannot be part of a larger tuple index
    assert_raises(IndexError, lambda: a[a[:,0]==1,0])
    assert_raises(IndexError, lambda: a[a[:,0]==1,...])
    assert_raises(IndexError, lambda: a[..., a[0]==1])
    assert_raises(IndexError, lambda: a[[True, True, True]])
    assert_raises(IndexError, lambda: a[(True, True, True),])

    # Integer array indices are not allowed (except for 0-D)
    idx = asarray([[0, 1]])
    assert_raises(IndexError, lambda: a[idx])
    assert_raises(IndexError, lambda: a[idx,])
    assert_raises(IndexError, lambda: a[[0, 1]])
    assert_raises(IndexError, lambda: a[(0, 1), (0, 1)])
    assert_raises(IndexError, lambda: a[[0, 1]])
    assert_raises(IndexError, lambda: a[np.array([[0, 1]])])

    # np.newaxis is not allowed
    assert_raises(IndexError, lambda: a[None])
    assert_raises(IndexError, lambda: a[None, ...])
    assert_raises(IndexError, lambda: a[..., None])
