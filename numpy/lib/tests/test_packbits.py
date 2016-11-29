from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_raises


def test_packbits():
    # Copied from the docstring.
    a = [[[1, 0, 1], [0, 1, 0]],
         [[1, 1, 0], [0, 0, 1]]]
    for dt in '?bBhHiIlLqQ':
        arr = np.array(a, dtype=dt)
        b = np.packbits(arr, axis=-1)
        assert_equal(b.dtype, np.uint8)
        assert_array_equal(b, np.array([[[160], [64]], [[192], [32]]]))

    assert_raises(TypeError, np.packbits, np.array(a, dtype=float))


def test_packbits_empty():
    shapes = [
        (0,), (10, 20, 0), (10, 0, 20), (0, 10, 20), (20, 0, 0), (0, 20, 0),
        (0, 0, 20), (0, 0, 0),
    ]
    for dt in '?bBhHiIlLqQ':
        for shape in shapes:
            a = np.empty(shape, dtype=dt)
            b = np.packbits(a)
            assert_equal(b.dtype, np.uint8)
            assert_equal(b.shape, (0,))


def test_packbits_empty_with_axis():
    # Original shapes and lists of packed shapes for different axes.
    shapes = [
        ((0,), [(0,)]),
        ((10, 20, 0), [(2, 20, 0), (10, 3, 0), (10, 20, 0)]),
        ((10, 0, 20), [(2, 0, 20), (10, 0, 20), (10, 0, 3)]),
        ((0, 10, 20), [(0, 10, 20), (0, 2, 20), (0, 10, 3)]),
        ((20, 0, 0), [(3, 0, 0), (20, 0, 0), (20, 0, 0)]),
        ((0, 20, 0), [(0, 20, 0), (0, 3, 0), (0, 20, 0)]),
        ((0, 0, 20), [(0, 0, 20), (0, 0, 20), (0, 0, 3)]),
        ((0, 0, 0), [(0, 0, 0), (0, 0, 0), (0, 0, 0)]),
    ]
    for dt in '?bBhHiIlLqQ':
        for in_shape, out_shapes in shapes:
            for ax, out_shape in enumerate(out_shapes):
                a = np.empty(in_shape, dtype=dt)
                b = np.packbits(a, axis=ax)
                assert_equal(b.dtype, np.uint8)
                assert_equal(b.shape, out_shape)


def test_unpackbits():
    # Copied from the docstring.
    a = np.array([[2], [7], [23]], dtype=np.uint8)
    b = np.unpackbits(a, axis=1)
    assert_equal(b.dtype, np.uint8)
    assert_array_equal(b, np.array([[0, 0, 0, 0, 0, 0, 1, 0],
                                    [0, 0, 0, 0, 0, 1, 1, 1],
                                    [0, 0, 0, 1, 0, 1, 1, 1]]))


def test_unpackbits_empty():
    a = np.empty((0,), dtype=np.uint8)
    b = np.unpackbits(a)
    assert_equal(b.dtype, np.uint8)
    assert_array_equal(b, np.empty((0,)))


def test_unpackbits_empty_with_axis():
    # Lists of packed shapes for different axes and unpacked shapes.
    shapes = [
        ([(0,)], (0,)),
        ([(2, 24, 0), (16, 3, 0), (16, 24, 0)], (16, 24, 0)),
        ([(2, 0, 24), (16, 0, 24), (16, 0, 3)], (16, 0, 24)),
        ([(0, 16, 24), (0, 2, 24), (0, 16, 3)], (0, 16, 24)),
        ([(3, 0, 0), (24, 0, 0), (24, 0, 0)], (24, 0, 0)),
        ([(0, 24, 0), (0, 3, 0), (0, 24, 0)], (0, 24, 0)),
        ([(0, 0, 24), (0, 0, 24), (0, 0, 3)], (0, 0, 24)),
        ([(0, 0, 0), (0, 0, 0), (0, 0, 0)], (0, 0, 0)),
    ]
    for in_shapes, out_shape in shapes:
        for ax, in_shape in enumerate(in_shapes):
            a = np.empty(in_shape, dtype=np.uint8)
            b = np.unpackbits(a, axis=ax)
            assert_equal(b.dtype, np.uint8)
            assert_equal(b.shape, out_shape)
