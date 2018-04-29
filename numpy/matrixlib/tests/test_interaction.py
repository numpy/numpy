"""Tests of interaction of matrix with other parts of numpy.

Note that tests with MaskedArray and linalg are done in separate files.
"""
from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
                           assert_raises_regex)


def test_fancy_indexing():
    # The matrix class messes with the shape. While this is always
    # weird (getitem is not used, it does not have setitem nor knows
    # about fancy indexing), this tests gh-3110
    # 2018-04-29: moved here from core.tests.test_index.
    m = np.matrix([[1, 2], [3, 4]])

    assert_(isinstance(m[[0, 1, 0], :], np.matrix))

    # gh-3110. Note the transpose currently because matrices do *not*
    # support dimension fixing for fancy indexing correctly.
    x = np.asmatrix(np.arange(50).reshape(5, 10))
    assert_equal(x[:2, np.array(-1)], x[:2, -1].T)


def test_polynomial_mapdomain():
    # test that polynomial preserved matrix subtype.
    # 2018-04-29: moved here from polynomial.tests.polyutils.
    dom1 = [0, 4]
    dom2 = [1, 3]
    x = np.matrix([dom1, dom1])
    res = np.polynomial.mapdomain(x, dom1, dom2)
    assert_(isinstance(res, np.matrix))


def test_sort_matrix_none():
    # 2018-04-29: moved here from core.tests.test_multiarray
    a = np.matrix([[2, 1, 0]])
    actual = np.sort(a, axis=None)
    expected = np.matrix([[0, 1, 2]])
    assert_equal(actual, expected)
    assert_(type(expected) is np.matrix)


def test_partition_matrix_none():
    # gh-4301
    # 2018-04-29: moved here from core.tests.test_multiarray
    a = np.matrix([[2, 1, 0]])
    actual = np.partition(a, 1, axis=None)
    expected = np.matrix([[0, 1, 2]])
    assert_equal(actual, expected)
    assert_(type(expected) is np.matrix)


def test_dot_scalar_and_matrix_of_objects():
    # Ticket #2469
    # 2018-04-29: moved here from core.tests.test_multiarray
    arr = np.matrix([1, 2], dtype=object)
    desired = np.matrix([[3, 6]], dtype=object)
    assert_equal(np.dot(arr, 3), desired)
    assert_equal(np.dot(3, arr), desired)


def test_inner_scalar_and_matrix():
    # 2018-04-29: moved here from core.tests.test_multiarray
    for dt in np.typecodes['AllInteger'] + np.typecodes['AllFloat'] + '?':
        sca = np.array(3, dtype=dt)[()]
        arr = np.matrix([[1, 2], [3, 4]], dtype=dt)
        desired = np.matrix([[3, 6], [9, 12]], dtype=dt)
        assert_equal(np.inner(arr, sca), desired)
        assert_equal(np.inner(sca, arr), desired)


def test_inner_scalar_and_matrix_of_objects(self):
    # Ticket #4482
    # 2018-04-29: moved here from core.tests.test_multiarray
    arr = np.matrix([1, 2], dtype=object)
    desired = np.matrix([[3, 6]], dtype=object)
    assert_equal(np.inner(arr, 3), desired)
    assert_equal(np.inner(3, arr), desired)


def test_iter_allocate_output_subtype():
    # Make sure that the subtype with priority wins
    # 2018-04-29: moved here from core.tests.test_nditer, given the
    # matrix specific shape test.

    # matrix vs ndarray
    a = np.matrix([[1, 2], [3, 4]])
    b = np.arange(4).reshape(2, 2).T
    i = np.nditer([a, b, None], [],
                  [['readonly'], ['readonly'], ['writeonly', 'allocate']])
    assert_(type(i.operands[2]) is np.matrix)
    assert_(type(i.operands[2]) is not np.ndarray)
    assert_equal(i.operands[2].shape, (2, 2))

    # matrix always wants things to be 2D
    b = np.arange(4).reshape(1, 2, 2)
    assert_raises(RuntimeError, np.nditer, [a, b, None], [],
                  [['readonly'], ['readonly'], ['writeonly', 'allocate']])
    # but if subtypes are disabled, the result can still work
    i = np.nditer([a, b, None], [],
                  [['readonly'], ['readonly'],
                   ['writeonly', 'allocate', 'no_subtype']])
    assert_(type(i.operands[2]) is np.ndarray)
    assert_(type(i.operands[2]) is not np.matrix)
    assert_equal(i.operands[2].shape, (1, 2, 2))


def like_function():
    # 2018-04-29: moved here from core.tests.test_numeric
    a = np.matrix([[1, 2], [3, 4]])
    for like_function in np.zeros_like, np.ones_like, np.empty_like:
        b = like_function(a)
        assert_(type(b) is np.matrix)

        c = like_function(a, subok=False)
        assert_(type(c) is not np.matrix)


def test_array_astype():
    # 2018-04-29: copied here from core.tests.test_api
    # subok=True passes through a matrix
    a = np.matrix([[0, 1, 2], [3, 4, 5]], dtype='f4')
    b = a.astype('f4', subok=True, copy=False)
    assert_(a is b)

    # subok=True is default, and creates a subtype on a cast
    b = a.astype('i4', copy=False)
    assert_equal(a, b)
    assert_equal(type(b), np.matrix)

    # subok=False never returns a matrix
    b = a.astype('f4', subok=False, copy=False)
    assert_equal(a, b)
    assert_(not (a is b))
    assert_(type(b) is not np.matrix)


def test_stack():
    # 2018-04-29: copied here from core.tests.test_shape_base
    # check np.matrix cannot be stacked
    m = np.matrix([[1, 2], [3, 4]])
    assert_raises_regex(ValueError, 'shape too large to be a matrix',
                        np.stack, [m, m])


def test_object_scalar_multiply():
    # Tickets #2469 and #4482
    # 2018-04-29: moved here from core.tests.test_ufunc
    arr = np.matrix([1, 2], dtype=object)
    desired = np.matrix([[3, 6]], dtype=object)
    assert_equal(np.multiply(arr, 3), desired)
    assert_equal(np.multiply(3, arr), desired)
