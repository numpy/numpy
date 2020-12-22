import itertools

import numpy as np
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_almost_equal,
    assert_raises, suppress_warnings, assert_raises_regex, assert_allclose
    )


def test_einsum_sums(dtype='object', do_opt=False):
    """
        Different from test_einsum.py since object dtype cannot perform
        np.sum(a, axis=-1).astype(dtype) when len(a) == 1
    """

    # sum(a, axis=-1)
    for n in range(1, 17):
        a = np.arange(n, dtype=dtype)
        assert_equal(np.einsum("i->", a, optimize=do_opt),
                     np.sum(a, axis=-1))
        assert_equal(np.einsum(a, [0], [], optimize=do_opt),
                     np.sum(a, axis=-1))


    for n in range(1, 17):
        a = np.arange(2*3*n, dtype=dtype).reshape(2, 3, n)
        assert_equal(np.einsum("...i->...", a, optimize=do_opt),
                     np.sum(a, axis=-1))
        assert_equal(np.einsum(a, [Ellipsis, 0], [Ellipsis], optimize=do_opt),
                     np.sum(a, axis=-1))

        
    # sum(a, axis=0)
    for n in range(1, 17):
        a = np.arange(2*n, dtype=dtype).reshape(2, n)
        assert_equal(np.einsum("i...->...", a, optimize=do_opt),
                     np.sum(a, axis=0))
        assert_equal(np.einsum(a, [0, Ellipsis], [Ellipsis], optimize=do_opt),
                     np.sum(a, axis=0))

    for n in range(1, 17):
        a = np.arange(2*3*n, dtype=dtype).reshape(2, 3, n)
        assert_equal(np.einsum("i...->...", a, optimize=do_opt),
                     np.sum(a, axis=0))
        assert_equal(np.einsum(a, [0, Ellipsis], [Ellipsis], optimize=do_opt),
                     np.sum(a, axis=0))

    # trace(a)
    for n in range(1, 17):
        a = np.arange(n*n, dtype=dtype).reshape(n, n)
        assert_equal(np.einsum("ii", a, optimize=do_opt),
                     np.trace(a))
        assert_equal(np.einsum(a, [0, 0], optimize=do_opt),
                     np.trace(a))

        # gh-15961: should accept numpy int64 type in subscript list
        np_array = np.asarray([0, 0])
        assert_equal(np.einsum(a, np_array, optimize=do_opt),
                     np.trace(a))
        assert_equal(np.einsum(a, list(np_array), optimize=do_opt),
                     np.trace(a))

    # multiply(a, b)
    assert_equal(np.einsum("..., ...", 3, 4), 12)  # scalar case
    for n in range(1, 17):
        a = np.arange(3 * n, dtype=dtype).reshape(3, n)
        b = np.arange(2 * 3 * n, dtype=dtype).reshape(2, 3, n)
        assert_equal(np.einsum("..., ...", a, b, optimize=do_opt),
                     np.multiply(a, b))
        assert_equal(np.einsum(a, [Ellipsis], b, [Ellipsis], optimize=do_opt),
                     np.multiply(a, b))

    # inner(a,b)
    for n in range(1, 17):
        a = np.arange(2 * 3 * n, dtype=dtype).reshape(2, 3, n)
        b = np.arange(n, dtype=dtype)
        assert_equal(np.einsum("...i, ...i", a, b, optimize=do_opt), np.inner(a, b))
        assert_equal(np.einsum(a, [Ellipsis, 0], b, [Ellipsis, 0], optimize=do_opt),
                     np.inner(a, b))

    for n in range(1, 11):
        a = np.arange(n * 3 * 2, dtype=dtype).reshape(n, 3, 2)
        b = np.arange(n, dtype=dtype)
        assert_equal(np.einsum("i..., i...", a, b, optimize=do_opt),
                     np.inner(a.T, b.T).T)
        assert_equal(np.einsum(a, [0, Ellipsis], b, [0, Ellipsis], optimize=do_opt),
                     np.inner(a.T, b.T).T)

    # outer(a,b)
    for n in range(1, 17):
        a = np.arange(3, dtype=dtype)+1
        b = np.arange(n, dtype=dtype)+1
        assert_equal(np.einsum("i,j", a, b, optimize=do_opt),
                     np.outer(a, b))
        assert_equal(np.einsum(a, [0], b, [1], optimize=do_opt),
                     np.outer(a, b))

    # Suppress the complex warnings for the 'as f8' tests
    with suppress_warnings() as sup:
        sup.filter(np.ComplexWarning)

        # matvec(a,b) / a.dot(b) where a is matrix, b is vector
        for n in range(1, 17):
            a = np.arange(4*n, dtype=dtype).reshape(4, n)
            b = np.arange(n, dtype=dtype)
            assert_equal(np.einsum("ij, j", a, b, optimize=do_opt),
                         np.dot(a, b))
            assert_equal(np.einsum(a, [0, 1], b, [1], optimize=do_opt),
                         np.dot(a, b))

            c = np.arange(4, dtype=dtype)
            np.einsum("ij,j", a, b, out=c,
                      dtype='f8', casting='unsafe', optimize=do_opt)
            assert_equal(c,
                         np.dot(a.astype('f8'),
                                b.astype('f8')).astype(dtype))
            c[...] = 0
            np.einsum(a, [0, 1], b, [1], out=c,
                      dtype='f8', casting='unsafe', optimize=do_opt)
            assert_equal(c,
                         np.dot(a.astype('f8'),
                                b.astype('f8')).astype(dtype))

        for n in range(1, 17):
            a = np.arange(4*n, dtype=dtype).reshape(4, n)
            b = np.arange(n, dtype=dtype)
            assert_equal(np.einsum("ji,j", a.T, b.T, optimize=do_opt),
                         np.dot(b.T, a.T))
            assert_equal(np.einsum(a.T, [1, 0], b.T, [1], optimize=do_opt),
                         np.dot(b.T, a.T))

            c = np.arange(4, dtype=dtype)
            np.einsum("ji,j", a.T, b.T, out=c,
                      dtype='f8', casting='unsafe', optimize=do_opt)
            assert_equal(c,
                         np.dot(b.T.astype('f8'),
                                a.T.astype('f8')).astype(dtype))
            c[...] = 0
            np.einsum(a.T, [1, 0], b.T, [1], out=c,
                      dtype='f8', casting='unsafe', optimize=do_opt)
            assert_equal(c,
                         np.dot(b.T.astype('f8'),
                                a.T.astype('f8')).astype(dtype))

        # matmat(a,b) / a.dot(b) where a is matrix, b is matrix
        for n in range(1, 17):
            if n < 8 or dtype != 'f2':
                a = np.arange(4*n, dtype=dtype).reshape(4, n)
                b = np.arange(n*6, dtype=dtype).reshape(n, 6)
                assert_equal(np.einsum("ij,jk", a, b, optimize=do_opt),
                             np.dot(a, b))
                assert_equal(np.einsum(a, [0, 1], b, [1, 2], optimize=do_opt),
                             np.dot(a, b))

        for n in range(1, 17):
            a = np.arange(4*n, dtype=dtype).reshape(4, n)
            b = np.arange(n*6, dtype=dtype).reshape(n, 6)
            c = np.arange(24, dtype=dtype).reshape(4, 6)
            np.einsum("ij,jk", a, b, out=c, dtype='f8', casting='unsafe',
                      optimize=do_opt)
            assert_equal(c,
                         np.dot(a.astype('f8'),
                                b.astype('f8')).astype(dtype))
            c[...] = 0
            np.einsum(a, [0, 1], b, [1, 2], out=c,
                      dtype='f8', casting='unsafe', optimize=do_opt)
            assert_equal(c,
                         np.dot(a.astype('f8'),
                                b.astype('f8')).astype(dtype))

        # matrix triple product (note this is not currently an efficient
        # way to multiply 3 matrices)
        a = np.arange(12, dtype=dtype).reshape(3, 4)
        b = np.arange(20, dtype=dtype).reshape(4, 5)
        c = np.arange(30, dtype=dtype).reshape(5, 6)
        if dtype != 'f2':
            assert_equal(np.einsum("ij,jk,kl", a, b, c, optimize=do_opt),
                         a.dot(b).dot(c))
            assert_equal(np.einsum(a, [0, 1], b, [1, 2], c, [2, 3],
                                   optimize=do_opt), a.dot(b).dot(c))

        d = np.arange(18, dtype=dtype).reshape(3, 6)
        np.einsum("ij,jk,kl", a, b, c, out=d,
                  dtype='f8', casting='unsafe', optimize=do_opt)
        tgt = a.astype('f8').dot(b.astype('f8'))
        tgt = tgt.dot(c.astype('f8')).astype(dtype)
        assert_equal(d, tgt)

        d[...] = 0
        np.einsum(a, [0, 1], b, [1, 2], c, [2, 3], out=d,
                  dtype='f8', casting='unsafe', optimize=do_opt)
        tgt = a.astype('f8').dot(b.astype('f8'))
        tgt = tgt.dot(c.astype('f8')).astype(dtype)
        assert_equal(d, tgt)

        # tensordot(a, b)
        if np.dtype(dtype) != np.dtype('f2'):
            a = np.arange(60, dtype=dtype).reshape(3, 4, 5)
            b = np.arange(24, dtype=dtype).reshape(4, 3, 2)
            assert_equal(np.einsum("ijk, jil -> kl", a, b),
                         np.tensordot(a, b, axes=([1, 0], [0, 1])))
            assert_equal(np.einsum(a, [0, 1, 2], b, [1, 0, 3], [2, 3]),
                         np.tensordot(a, b, axes=([1, 0], [0, 1])))

            c = np.arange(10, dtype=dtype).reshape(5, 2)
            np.einsum("ijk,jil->kl", a, b, out=c,
                      dtype='f8', casting='unsafe', optimize=do_opt)
            assert_equal(c, np.tensordot(a.astype('f8'), b.astype('f8'),
                         axes=([1, 0], [0, 1])).astype(dtype))
            c[...] = 0
            np.einsum(a, [0, 1, 2], b, [1, 0, 3], [2, 3], out=c,
                      dtype='f8', casting='unsafe', optimize=do_opt)
            assert_equal(c, np.tensordot(a.astype('f8'), b.astype('f8'),
                         axes=([1, 0], [0, 1])).astype(dtype))

    # logical_and(logical_and(a!=0, b!=0), c!=0)
    a = np.array([1,   3,   -2,   0,   12,  13,   0,   1], dtype=dtype)
    b = np.array([0,   3.5, 0.,   -2,  0,   1,    3,   12], dtype=dtype)
    c = np.array([True, True, False, True, True, False, True, True])
    assert_equal(np.einsum("i,i,i->i", a, b, c,
                 dtype='?', casting='unsafe', optimize=do_opt),
                 np.logical_and(np.logical_and(a != 0, b != 0), c != 0))
    assert_equal(np.einsum(a, [0], b, [0], c, [0], [0],
                 dtype='?', casting='unsafe'),
                 np.logical_and(np.logical_and(a != 0, b != 0), c != 0))

    a = np.arange(9, dtype=dtype)
    assert_equal(np.einsum(",i->", 3, a), 3*np.sum(a))
    assert_equal(np.einsum(3, [], a, [0], []), 3*np.sum(a))
    assert_equal(np.einsum("i,->", a, 3), 3*np.sum(a))
    assert_equal(np.einsum(a, [0], 3, [], []), 3*np.sum(a))

    # Various stride0, contiguous, and SSE aligned variants
    for n in range(1, 25):
        a = np.arange(n, dtype=dtype)
        if np.dtype(dtype).itemsize > 1:
            assert_equal(np.einsum("...,...", a, a, optimize=do_opt),
                         np.multiply(a, a))
            assert_equal(np.einsum("i,i", a, a, optimize=do_opt), np.dot(a, a))
            assert_equal(np.einsum("i,->i", a, 2, optimize=do_opt), 2*a)
            assert_equal(np.einsum(",i->i", 2, a, optimize=do_opt), 2*a)
            assert_equal(np.einsum("i,->", a, 2, optimize=do_opt), 2*np.sum(a))
            assert_equal(np.einsum(",i->", 2, a, optimize=do_opt), 2*np.sum(a))

            assert_equal(np.einsum("...,...", a[1:], a[:-1], optimize=do_opt),
                         np.multiply(a[1:], a[:-1]))
            assert_equal(np.einsum("i,i", a[1:], a[:-1], optimize=do_opt),
                         np.dot(a[1:], a[:-1]))
            assert_equal(np.einsum("i,->i", a[1:], 2, optimize=do_opt), 2*a[1:])
            assert_equal(np.einsum(",i->i", 2, a[1:], optimize=do_opt), 2*a[1:])
            assert_equal(np.einsum("i,->", a[1:], 2, optimize=do_opt),
                         2*np.sum(a[1:]))
            assert_equal(np.einsum(",i->", 2, a[1:], optimize=do_opt),
                         2*np.sum(a[1:]))

    # An object array, summed as the data type
    a = np.arange(9, dtype=object)

    b = np.einsum("i->", a, dtype=dtype, casting='unsafe')
    assert_equal(b, np.sum(a))
    assert_equal(b.dtype, np.dtype(dtype))

    b = np.einsum(a, [0], [], dtype=dtype, casting='unsafe')
    assert_equal(b, np.sum(a))
    assert_equal(b.dtype, np.dtype(dtype))

    # A case which was failing (ticket #1885)
    p = np.arange(2) + 1
    q = np.arange(4).reshape(2, 2) + 3
    r = np.arange(4).reshape(2, 2) + 7
    assert_equal(np.einsum('z,mz,zm->', p, q, r), 253)

    # singleton dimensions broadcast (gh-10343)
    p = np.ones((10,2))
    q = np.ones((1,2))
    assert_array_equal(np.einsum('ij,ij->j', p, q, optimize=True),
                       np.einsum('ij,ij->j', p, q, optimize=False))
    assert_array_equal(np.einsum('ij,ij->j', p, q, optimize=True),
                       [10.] * 2)

    # a blas-compatible contraction broadcasting case which was failing
    # for optimize=True (ticket #10930)
    x = np.array([2., 3.])
    y = np.array([4.])
    assert_array_equal(np.einsum("i, i", x, y, optimize=False), 20.)
    assert_array_equal(np.einsum("i, i", x, y, optimize=True), 20.)

    # all-ones array was bypassing bug (ticket #10930)
    p = np.ones((1, 5)) / 2
    q = np.ones((5, 5)) / 2
    for optimize in (True, False):
        assert_array_equal(np.einsum("...ij,...jk->...ik", p, p,
                                     optimize=optimize),
                           np.einsum("...ij,...jk->...ik", p, q,
                                     optimize=optimize))
        assert_array_equal(np.einsum("...ij,...jk->...ik", p, q,
                                     optimize=optimize),
                           np.full((1, 5), 1.25))

    # Cases which were failing (gh-10899)
    x = np.eye(2, dtype=dtype)
    y = np.ones(2, dtype=dtype)
    assert_array_equal(np.einsum("ji,i->", x, y, optimize=optimize),
                       [2.])  # contig_contig_outstride0_two
    assert_array_equal(np.einsum("i,ij->", y, x, optimize=optimize),
                       [2.])  # stride0_contig_outstride0_two
    assert_array_equal(np.einsum("ij,i->", x, y, optimize=optimize),
                       [2.])  # contig_stride0_outstride0_two

