import numpy as np

from numpy.testing import assert_array_equal, assert_equal, assert_raises


def test_packbits():
    # Copied from the docstring.
    a = [[[1, 0, 1], [0, 1, 0]],
         [[1, 1, 0], [0, 0, 1]]]
    for dtype in [np.bool, np.uint8, np.int]:
        arr = np.array(a, dtype=dtype)
        b = np.packbits(arr, axis=-1)
        assert_equal(b.dtype, np.uint8)
        assert_array_equal(b, np.array([[[160], [64]], [[192], [32]]]))

    assert_raises(TypeError, np.packbits, np.array(a, dtype=float))


def test_unpackbits():
    # Copied from the docstring.
    a = np.array([[2], [7], [23]], dtype=np.uint8)
    b = np.unpackbits(a, axis=1)
    assert_equal(b.dtype, np.uint8)
    assert_array_equal(b, np.array([[0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1, 1, 1],
                                    [0, 0, 0, 1, 0, 1, 1, 1]]))
