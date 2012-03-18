import numpy as np
from numpy.compat import asbytes
from numpy.testing import *
import sys, warnings
from numpy.testing.utils import WarningManager


def combinations(iterable, r):
    # copied from 2.7 documentation in order to support
    # Python < 2.6
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)


def test_array_maskna_flags():
    a = np.arange(3)
    assert_(not a.flags.maskna)
    assert_(not a.flags.ownmaskna)
    assert_(not a.flags['MASKNA'])
    assert_(not a.flags['OWNMASKNA'])
    # Add a mask by setting the flag
    a.flags.maskna = True
    assert_(a.flags.maskna)
    assert_(a.flags.ownmaskna)
    assert_(a.flags['MASKNA'])
    assert_(a.flags['OWNMASKNA'])
    # Can't remove the mask once it's created
    def setmaskna(x, v):
        x.maskna = v
    assert_raises(ValueError, setmaskna, a.flags, False)
    def setownmaskna(x, v):
        x.ownmaskna = v
    assert_raises(ValueError, setownmaskna, a.flags, False)


def test_array_maskna_construction():
    # Construction with NA inputs
    a = np.array([1.0, 2.0, np.NA, 7.0], maskna=True)
    assert_equal(a.dtype, np.dtype('f8'))
    assert_(a.flags.maskna)
    assert_equal(type(a[2]), np.NAType)
    # Without the 'maskna=True', still produces an NA mask if NA is there
    a = np.array([1.0, 2.0, np.NA, 7.0])
    assert_equal(a.dtype, np.dtype('f8'))
    assert_(a.flags.maskna)
    assert_equal(type(a[2]), np.NAType)
    # Without any NAs, does not produce an NA mask
    a = np.array([1.0, 2.0, 4.0, 7.0])
    assert_equal(a.dtype, np.dtype('f8'))
    assert_(not a.flags.maskna)

    # From np.NA as a straight scalar
    a = np.array(np.NA, maskna=True)
    assert_equal(type(a), np.ndarray)
    assert_(np.isna(a))

    # As a special case, converting np.NA to an array produces
    # a zero-dimensional masked array
    a = np.array(np.NA)
    assert_equal(type(a), np.ndarray)
    assert_(np.isna(a))

    # The data type defaults to the same as an empty array if all is NA
    a = np.array([np.NA], maskna=True)
    b = np.array([])
    assert_equal(a.dtype, b.dtype)
    assert_(np.isna(a))

    a = np.zeros((3,))
    assert_(not a.flags.maskna)
    a = np.zeros((3,), maskna=True)
    assert_(a.flags.maskna)
    assert_equal(np.isna(a), False)

    a = np.ones((3,))
    assert_(not a.flags.maskna)
    a = np.ones((3,), maskna=True)
    assert_(a.flags.maskna)
    assert_equal(np.isna(a), False)

    # np.empty returns all NAs if maskna is set to True
    a = np.empty((3,))
    assert_(not a.flags.maskna)
    a = np.empty((3,), maskna=True)
    assert_(a.flags.maskna)
    assert_equal(np.isna(a), True)

    # np.empty_like returns all NAs if maskna is set to True
    tmp = np.arange(3)
    a = np.empty_like(tmp)
    assert_(not a.flags.maskna)
    a = np.empty_like(tmp, maskna=True)
    assert_(a.flags.maskna)
    assert_equal(np.isna(a), True)


def test_array_maskna_asarray():
    a = np.arange(6).reshape(2,3)

    # Should not add an NA mask by default
    res = np.asarray(a)
    assert_(res is a)
    assert_(not res.flags.maskna)

    # Should add an NA mask if requested
    res = np.asarray(a, maskna=True)
    assert_(res.flags.maskna)
    assert_(res.flags.ownmaskna)
    res = np.asarray(a, ownmaskna=True)
    assert_(res.flags.maskna)
    assert_(res.flags.ownmaskna)

    a.flags.maskna = True

    # Should view or create a copy of the NA mask
    res = np.asarray(a)
    assert_(res is a)
    res = np.asarray(a, maskna=True)
    assert_(res is a)
    res = np.asarray(a, ownmaskna=True)
    assert_(res is a)

    b = a.view()
    assert_(not b.flags.ownmaskna)

    res = np.asarray(b)
    assert_(res is b)
    res = np.asarray(b, maskna=True)
    assert_(res is b)
    res = np.asarray(b, ownmaskna=True)
    assert_(not (res is b))
    assert_(res.flags.ownmaskna)


def test_array_maskna_copy():
    a = np.array([1,2,3])
    b = np.array([2,3,4], maskna=True)
    c = np.array([3,4,np.NA], maskna=True)

    # Make a copy, adding a mask
    res = a.copy(maskna=True)
    assert_equal(res, a)
    assert_(res.flags.maskna)
    assert_(res.flags.ownmaskna)

    res = np.copy(a, maskna=True)
    assert_equal(res, a)
    assert_(res.flags.maskna)
    assert_(res.flags.ownmaskna)

    # Make a copy, removing a mask
    res = b.copy(maskna=False)
    assert_equal(res, b)
    assert_(not res.flags.maskna)
    assert_(not res.flags.ownmaskna)

    res = np.copy(b, maskna=False)
    assert_equal(res, b)
    assert_(not res.flags.maskna)
    assert_(not res.flags.ownmaskna)

    # Copy with removing a mask doesn't work if there are NAs
    assert_raises(ValueError, c.copy, maskna=False)
    assert_raises(ValueError, np.copy, c, maskna=False)

    # Make a copy, preserving non-masked
    res = a.copy()
    assert_equal(res, a)
    assert_(not res.flags.maskna)
    assert_(not res.flags.ownmaskna)

    res = np.copy(a)
    assert_equal(res, a)
    assert_(not res.flags.maskna)
    assert_(not res.flags.ownmaskna)

    # Make a copy, preserving masked
    res = b.copy()
    assert_equal(res, b)
    assert_(res.flags.maskna)
    assert_(res.flags.ownmaskna)

    res = np.copy(b)
    assert_equal(res, b)
    assert_(res.flags.maskna)
    assert_(res.flags.ownmaskna)

    # Make a copy, preserving masked with an NA
    res = c.copy()
    assert_array_equal(res, c)
    assert_(res.flags.maskna)
    assert_(res.flags.ownmaskna)

    res = np.copy(c)
    assert_array_equal(res, c)
    assert_(res.flags.maskna)
    assert_(res.flags.ownmaskna)


def test_array_maskna_astype():
    dtsrc = [np.dtype(d) for d in '?bhilqpBHILQPefdgFDGSUO']
    #dtsrc.append(np.dtype([('b', np.int, (1,))]))
    dtsrc.append(np.dtype('datetime64[D]'))
    dtsrc.append(np.dtype('timedelta64[s]'))

    dtdst = [np.dtype(d) for d in '?bhilqpBHILQPefdgFDGSUO']
    #dtdst.append(np.dtype([('b', np.int, (1,))]))
    dtdst.append(np.dtype('datetime64[D]'))
    dtdst.append(np.dtype('timedelta64[s]'))

    warn_ctx = WarningManager()
    warn_ctx.__enter__()
    try:
        warnings.simplefilter("ignore", np.ComplexWarning)
        for dt1 in dtsrc:
            a = np.ones(2, dt1, maskna=1)
            a[1] = np.NA
            for dt2 in dtdst:
                msg = 'type %s to %s conversion' % (dt1, dt2)
                b = a.astype(dt2)
                assert_(b.flags.maskna, msg)
                assert_(b.flags.ownmaskna, msg)
                assert_(np.isna(b[1]), msg)
    finally:
        warn_ctx.__exit__()


def test_array_maskna_repr():
    # Test some simple reprs with NA in them
    a = np.array(np.NA, maskna=True)
    assert_equal(repr(a), 'array(NA, dtype=float64)')
    a = np.array(3, maskna=True)
    assert_equal(repr(a), 'array(3, maskna=True)')
    a = np.array([np.NA, 3], maskna=True)
    assert_equal(repr(a), 'array([NA, 3])')
    a = np.array([np.NA, np.NA])
    assert_equal(repr(a), 'array([ NA,  NA], dtype=float64)')
    a = np.array([3.5, np.NA], maskna=True)
    assert_equal(repr(a), 'array([ 3.5,   NA])')
    a = np.array([3.75, 6.25], maskna=True)
    assert_equal(repr(a), 'array([ 3.75,  6.25], maskna=True)')
    a = np.array([3.75, 6.25], maskna=True, dtype='f4')
    assert_equal(repr(a), 'array([ 3.75,  6.25], maskna=True, dtype=float32)')


def test_isna():
    # Objects which are not np.NA or ndarray all return False
    assert_equal(np.isna(True), False)
    assert_equal(np.isna("abc"), False)
    assert_equal(np.isna([1,2,3]), False)
    assert_equal(np.isna({3:5}), False)
    # Various NA values return True
    assert_equal(np.isna(np.NA), True)
    assert_equal(np.isna(np.NA()), True)
    assert_equal(np.isna(np.NA(5)), True)
    assert_equal(np.isna(np.NA(dtype='f4')), True)
    assert_equal(np.isna(np.NA(12,dtype='f4')), True)


def test_array_maskna_item():
    # With a zero-dimensional array
    a = np.array(np.NA, maskna=True)

    # Should return NA as the item
    assert_equal(type(a.item()), np.NAType)

    # Should be able to set the item
    a.itemset(1.5)
    assert_(not np.isna(a))
    assert_equal(a, 1.5)
    a.itemset(np.NA)
    assert_(np.isna(a))

    # With a one-dimensional array
    a = np.array([1, np.NA, 2, np.NA], maskna=True)

    # Should return the scalar or NA as the item
    assert_(not np.isna(a.item(0)))
    assert_equal(type(a.item(1)), np.NAType)

    # Should be able to set the items
    a.itemset(0, np.NA)
    assert_(np.isna(a[0]))
    a.itemset(1, 12)
    assert_(not np.isna(a[1]))
    assert_equal(a[1], 12)

    # With a two-dimensional array
    a = np.arange(6, maskna=True).reshape(2,3)
    a[0,1] = np.NA
    # Should return the scalar or NA as the item
    assert_(not np.isna(a.item((0,0))))
    assert_equal(type(a.item((0,1))), np.NAType)

    # Should be able to set the items
    a.itemset((0,1), 8)
    assert_(not np.isna(a[0,1]))
    assert_equal(a[0,1], 8)
    a.itemset((1,1), np.NA)
    assert_(np.isna(a[1,1]))


def test_array_maskna_payload():
    # Single numbered index
    a = np.zeros((2,), maskna=True)
    a[0] = np.NA
    assert_equal(a[0].payload, None)

    # Tuple index
    a = np.zeros((2,3), maskna=True)
    a[1,1] = np.NA
    assert_equal(a[1,1].payload, None)


def test_array_maskna_isna_1D():
    a = np.arange(10)

    # With no mask, it returns all False
    assert_equal(np.isna(a), False)
    assert_equal(np.isna(a).shape, (10,))

    # With a mask but no NAs, it still returns all False
    a.flags.maskna = True
    assert_equal(np.isna(a), False)
    assert_equal(np.isna(a).shape, (10,))

    # Checking isna of a single value
    assert_equal(np.isna(a[4]), False)
    # Assigning NA to a single value
    a[3] = np.NA
    assert_equal(np.isna(a), [0,0,0,1,0,0,0,0,0,0])
    # Checking isna of a single value
    assert_equal(np.isna(a[3]), True)

    # Checking isna of a slice
    assert_equal(np.isna(a[1:6]), [0,0,1,0,0])
    # Assigning NA to a slice
    a[5:7] = np.NA
    assert_equal(np.isna(a), [0,0,0,1,0,1,1,0,0,0])

    # Checking isna of a strided slice
    assert_equal(np.isna(a[1:8:2]), [0,1,1,0])
    # Assigning NA to a strided slice
    a[2:10:3] = np.NA
    assert_equal(np.isna(a), [0,0,1,1,0,1,1,0,1,0])

    # Checking isna of a boolean mask index
    mask = np.array([1,1,0,0,0,1,0,1,1,0], dtype='?')
    assert_equal(np.isna(a[mask]), [0,0,1,0,1])
    # Assigning NA to a boolean masked index
    a[mask] = np.NA
    assert_equal(np.isna(a), [1,1,1,1,0,1,1,1,1,0])

    # TODO: fancy indexing is next...


def test_array_maskna_isna_2D():
    a = np.zeros((3,4))

    # With no mask, it returns all False
    assert_equal(np.isna(a), False)
    assert_equal(np.isna(a).shape, (3,4))

    # With a mask but no NAs, it still returns all False
    a.flags.maskna = True
    assert_equal(np.isna(a), False)
    assert_equal(np.isna(a).shape, (3,4))

    # Checking isna of a single value
    assert_equal(np.isna(a[1,2]), False)
    # Assigning NA to a single value
    a[1,2] = np.NA
    assert_equal(np.isna(a), [[0,0,0,0],[0,0,1,0],[0,0,0,0]])
    # Checking isna of a single value
    assert_equal(np.isna(a[1,2]), True)

    # Checking isna of a slice
    assert_equal(np.isna(a[1:4,1:3]), [[0,1],[0,0]])
    # Assigning NA to a slice
    a[1:3,0:2] = np.NA
    assert_equal(np.isna(a), [[0,0,0,0],[1,1,1,0],[1,1,0,0]])

    # Checking isna of a strided slice
    assert_equal(np.isna(a[1:,1:5:2]), [[1,0],[1,0]])
    # Assigning NA to a strided slice
    a[::2,::2] = np.NA
    assert_equal(np.isna(a), [[1,0,1,0],[1,1,1,0],[1,1,1,0]])

    # Checking isna of a boolean mask index
    mask = np.array([[1,1,0,0],[0,1,0,1],[0,0,1,0]], dtype='?')
    assert_equal(np.isna(a[mask]), [1,0,1,0,1])
    # Assigning NA to a boolean masked index
    a[mask] = np.NA
    assert_equal(np.isna(a), [[1,1,1,0],[1,1,1,1],[1,1,1,0]])

    # TODO: fancy indexing is next...


def test_array_maskna_to_nomask():
    # Assignment from an array with NAs to a non-masked array,
    # excluding the NAs with a mask
    a = np.array([[2,np.NA,5],[1,6,np.NA]], maskna=True)
    mask = np.array([[1,0,0],[1,1,0]], dtype='?')
    badmask = np.array([[1,0,0],[0,1,1]], dtype='?')
    expected = np.array([[2,1,2],[1,6,5]])

    # With masked indexing
    b = np.arange(6).reshape(2,3)
    b[mask] = a[mask]
    assert_array_equal(b, expected)

    # With copyto
    b = np.arange(6).reshape(2,3)
    np.copyto(b, a, where=mask)
    assert_array_equal(b, expected)

    # With masked indexing
    b = np.arange(6).reshape(2,3)
    def asn():
        b[badmask] = a[badmask]
    assert_raises(ValueError, asn)

    # With copyto
    b = np.arange(6).reshape(2,3)
    assert_raises(ValueError, np.copyto, b, a, where=badmask)


def test_array_maskna_view_function():
    a = np.arange(10)

    # Taking a view of a non-masked array, making sure there's a mask
    b = a.view(maskna=True)
    assert_(not a.flags.maskna)
    assert_(b.flags.maskna)
    assert_(b.flags.ownmaskna)

    # Taking a view of a non-masked array, making sure there's an owned mask
    b = a.view(ownmaskna=True)
    assert_(not a.flags.maskna)
    assert_(b.flags.maskna)
    assert_(b.flags.ownmaskna)

    # Taking a view of a masked array
    c = b.view()
    assert_(b.flags.maskna)
    assert_(b.flags.ownmaskna)
    assert_(c.flags.maskna)
    assert_(not c.flags.ownmaskna)

    # Taking a view of a masked array with maskna=False is invalid
    assert_raises(ValueError, b.view, maskna=False)

    # Taking a view of a masked array, making sure there's a mask
    c = b.view(maskna = True)
    assert_(b.flags.maskna)
    assert_(b.flags.ownmaskna)
    assert_(c.flags.maskna)
    assert_(not c.flags.ownmaskna)

    # Taking a view of a masked array, making sure there's an owned mask
    c = b.view(ownmaskna = True)
    assert_(b.flags.maskna)
    assert_(b.flags.ownmaskna)
    assert_(c.flags.maskna)
    assert_(c.flags.ownmaskna)

def test_array_maskna_view_dtype():
    tcs = np.typecodes['AllFloat'] + \
          np.typecodes['AllInteger'] + \
          np.typecodes['Complex']

    same_size = []
    diff_size = []
    for x in combinations(tcs, 2):
        if np.dtype(x[0]).itemsize == np.dtype(x[1]).itemsize:
            same_size.append(x)
        else: diff_size.append(x)

    for (from_type, to_type) in diff_size:
        a = np.arange(10, dtype=from_type, maskna=True)

        # Ensure that a view of a masked array cannot change to
        # different sized dtype
        assert_raises(TypeError, a.view, to_type)

    for (from_type, to_type) in same_size:
        a = np.arange(10, dtype=from_type, maskna=True)

        # Ensure that a view of a masked array can change to
        # same sized dtype
        b = a.view(dtype=to_type)

def test_array_maskna_array_function_1D():
    a = np.arange(10)
    a_ref = a.copy()
    b = a.view(maskna=True)
    b[3:10:2] = np.NA
    b_view = b.view()

    # Ensure the setup is correct
    assert_(not a.flags.maskna)
    assert_(b.flags.maskna)
    assert_(b.flags.ownmaskna)
    assert_(b_view.flags.maskna)
    assert_(not b_view.flags.ownmaskna)

    # Should be able to add a mask with 'maskna='
    c = np.array(a, maskna=True)
    assert_(c.flags.maskna)
    assert_(c.flags.ownmaskna)
    assert_(not (c is b))

    # Should be able to add a mask with 'ownmaskna='
    c = np.array(a, ownmaskna=True)
    assert_(c.flags.maskna)
    assert_(c.flags.ownmaskna)
    assert_(not (c is b))

    # Should propagate mask
    c = np.array(b)
    assert_(c.flags.maskna)
    assert_(c.flags.ownmaskna)
    assert_equal(np.isna(b), np.isna(c))
    assert_(not (c is b))

    # Should propagate mask with 'maskna=True'
    c = np.array(b, maskna=True)
    assert_(c.flags.maskna)
    assert_(c.flags.ownmaskna)
    assert_equal(np.isna(b), np.isna(c))
    assert_(not (c is b))

    # Should propagate mask with 'ownmaskna=True'
    c = np.array(b, ownmaskna=True)
    assert_(c.flags.maskna)
    assert_(c.flags.ownmaskna)
    assert_equal(np.isna(b), np.isna(c))
    assert_(not (c is b))

    # Should be able to pass it through
    c = np.array(b, copy=False)
    assert_(c is b)

    # Should be able to pass it through with 'maskna=True'
    c = np.array(b, copy=False, maskna=True)
    assert_(c is b)

    # Should be able to pass it through with 'maskna=True'
    c = np.array(b_view, copy=False, maskna=True)
    assert_(c is b_view)

    # Should be able to pass an owned mask through with 'ownmaskna=True'
    c = np.array(b, copy=False, ownmaskna=True)
    assert_(c is b)

    # Should produce a view with an owned mask with 'ownmaskna=True'
    c = np.array(b_view, copy=False, ownmaskna=True)
    assert_(c.base is b_view.base)
    assert_(c.flags.ownmaskna)
    assert_(not (c is b_view))

    # Should produce a view whose base is 'c', because 'c' owns
    # the data for its mask
    d = c.view()
    assert_(d.base is c)
    assert_(d.flags.maskna)
    assert_(not d.flags.ownmaskna)


def test_array_maskna_setasflat():
    # Copy from a C to a F array with some NAs
    a_orig = np.empty((2,3), order='C')
    b_orig = np.empty((3,2), order='F')
    a = a_orig.view(maskna=True)
    b = b_orig.view(maskna=True)
    a[...] = 1
    a[0,1] = np.NA
    a[1,2] = np.NA
    b[...] = 2
    b.setasflat(a)
    assert_equal(np.isna(a), [[0,1,0],[0,0,1]])
    assert_equal(b_orig, [[1,2],[1,1],[1,2]])
    assert_equal(np.isna(b), [[0,1],[0,0],[0,1]])


def test_array_maskna_ravel():
    # From a C array
    a = np.zeros((2,3), maskna=True, order='C')
    a[0,1] = np.NA
    a[1,2] = np.NA

    # Ravel in C order returns a view
    b = np.ravel(a)
    assert_(b.base is a)
    assert_equal(b.shape, (6,))
    assert_(b.flags.maskna)
    assert_(not b.flags.ownmaskna)
    assert_equal(np.isna(b), [0,1,0,0,0,1])

    # Ravel in F order returns a copy
    b = np.ravel(a, order='F')
    assert_(b.base is None)
    assert_equal(b.shape, (6,))
    assert_(b.flags.maskna)
    assert_(b.flags.ownmaskna)
    assert_equal(np.isna(b), [0,0,1,0,0,1])

    a = np.arange(12, maskna=True).reshape(2,3,2).swapaxes(1,2)
    assert_equal(a.ravel(order='K'), np.arange(12))


def test_array_maskna_reshape():
    # Simple reshape 1D -> 2D
    a = np.arange(6, maskna=True)
    a[1] = np.NA
    a[5] = np.NA

    # Reshape from 1D to C order
    b = a.reshape(2,3)
    assert_(b.base is a)
    assert_equal(b.shape, (2,3))
    assert_(b.flags.maskna)
    assert_(not b.flags.ownmaskna)
    assert_equal(np.isna(b), [[0,1,0],[0,0,1]])

    # Reshape from 1D to F order
    b = a.reshape(2,3,order='F')
    assert_(b.base is a)
    assert_equal(b.shape, (2,3))
    assert_(b.flags.maskna)
    assert_(not b.flags.ownmaskna)
    assert_equal(np.isna(b), [[0,0,0],[1,0,1]])

    # Add a new axis using 'newaxis'
    a = np.array(np.NA, maskna=True)
    assert_equal(np.isna(a[np.newaxis]), [True])


def test_array_maskna_view_NA_assignment_1D():
    a = np.arange(10)
    a_ref = a.copy()

    # Make sure that assigning NA doesn't affect the original data
    b = a.view(maskna=True)
    b[...] = np.NA
    assert_equal(np.isna(b), True)
    assert_equal(a, a_ref)

    b = a.view(maskna=True)
    b[:] = np.NA
    assert_equal(np.isna(b), True)
    assert_equal(a, a_ref)

    b = a.view(maskna=True)
    b[3:5] = np.NA
    assert_equal(np.isna(b), [0,0,0,1,1,0,0,0,0,0])
    assert_equal(a, a_ref)

    b = a.view(maskna=True)
    b[3:10:3] = np.NA
    assert_equal(np.isna(b), [0,0,0,1,0,0,1,0,0,1])
    assert_equal(a, a_ref)

    b = a.view(maskna=True)
    b[3] = np.NA
    assert_equal(np.isna(b), [0,0,0,1,0,0,0,0,0,0])
    assert_equal(a, a_ref)

    b = a.view(maskna=True)
    mask = np.array([0,1,0,1,1,0,0,0,1,1], dtype='?')
    b[mask] = np.NA
    assert_equal(np.isna(b), mask)
    assert_equal(a, a_ref)

    # TODO: fancy indexing is next...


def test_array_maskna_view_NA_assignment_2D():
    a = np.arange(6).reshape(2,3)
    a_ref = a.copy()

    # Make sure that assigning NA doesn't affect the original data
    b = a.view(maskna=True)
    b[...] = np.NA
    assert_equal(np.isna(b), True)
    assert_equal(a, a_ref)

    b = a.view(maskna=True)
    b[:] = np.NA
    assert_equal(np.isna(b), True)
    assert_equal(a, a_ref)

    b = a.view(maskna=True)
    b[0,:] = np.NA
    assert_equal(np.isna(b[0]), True)
    assert_equal(np.isna(b[1]), False)
    assert_equal(a, a_ref)

    b = a.view(maskna=True)
    b[1:,1:3] = np.NA
    assert_equal(np.isna(b), [[0,0,0],[0,1,1]])
    assert_equal(a, a_ref)

    b = a.view(maskna=True)
    b[1,::2] = np.NA
    assert_equal(np.isna(b), [[0,0,0],[1,0,1]])
    assert_equal(a, a_ref)

    b = a.view(maskna=True)
    b[0,2] = np.NA
    assert_equal(np.isna(b), [[0,0,1],[0,0,0]])
    assert_equal(a, a_ref)

    b = a.view(maskna=True)
    mask = np.array([[1,0,1],[1,1,0]], dtype='?')
    b[mask] = np.NA
    assert_equal(np.isna(b), mask)
    assert_equal(a, a_ref)

    # TODO: fancy indexing is next...


def test_array_maskna_view_array_assignment_1D():
    a = np.arange(5)
    b = a.view(maskna=True)

    # Assigning a constant scalar should unmask the values
    b[...] = np.NA
    b[...] = 3
    assert_equal(a, 3)
    assert_equal(np.isna(b), False)

    # Assigning from a list should unmask the values
    b[...] = np.NA
    b[...] = [2]
    assert_equal(a, [2,2,2,2,2])
    assert_equal(np.isna(b), False)

    # Assigning from a list should unmask the values
    b[...] = np.NA
    b[...] = [2,3,4,5,6]
    assert_equal(a, [2,3,4,5,6])
    assert_equal(np.isna(b), False)

    # Assigning from a list with NAs should unmask the non-NA values
    b[...] = np.NA
    b[...] = [7,np.NA,2,0,np.NA]
    assert_equal(a, [7,3,2,0,6])
    assert_equal(np.isna(b), [0,1,0,0,1])

    # Assigning from an unmasked array should unmask the values
    b[...] = np.NA
    b[...] = np.arange(5)
    assert_equal(a, np.arange(5))
    assert_equal(np.isna(b), False)

    # Assigning from a masked array with no NAs should unmask the values
    b[...] = np.NA
    tmp = np.arange(5) + 1
    tmp.flags.maskna = True
    b[...] = tmp
    assert_equal(a, np.arange(5) + 1)
    assert_equal(np.isna(b), False)

    # Assigning from a masked array with some NAs should unmask most
    # of the values, and leave the value behind the NAs untouched
    b[...] = np.NA
    tmp = np.arange(5) + 5
    tmp.flags.maskna = True
    tmp[2] = np.NA
    b[...] = tmp
    assert_equal(a, [5,6,3,8,9])
    assert_equal(np.isna(b), [0,0,1,0,0])

    # Assigning to a single element should unmask the value
    b[...] = np.NA
    b[2] = 10
    assert_equal(a, [5,6,10,8,9])
    assert_equal(np.isna(b), [1,1,0,1,1])

    # Assigning to a simple slice should unmask the values
    b[...] = np.NA
    b[2:] = 4
    assert_equal(a, [5,6,4,4,4])
    assert_equal(np.isna(b), [1,1,0,0,0])

    # Assigning to a strided slice should unmask the values
    b[...] = np.NA
    b[3::-2] = 12
    assert_equal(a, [5,12,4,12,4])
    assert_equal(np.isna(b), [1,0,1,0,1])

    # Assigning to a boolean index should unmask the values
    b[...] = np.NA
    mask = np.array([0,1,1,0,1], dtype='?')
    b[mask] = 7
    assert_equal(a, [5,7,7,12,7])
    assert_equal(np.isna(b), [1,0,0,1,0])

    # Assigning a list to a boolean index should unmask the values
    b[...] = np.NA
    mask = np.array([1,0,0,0,1], dtype='?')
    b[mask] = [8,1]
    assert_equal(a, [8,7,7,12,1])
    assert_equal(np.isna(b), [0,1,1,1,0])

    # Assigning a list with NA to a boolean index should unmask non-NA values
    b[...] = np.NA
    mask = np.array([0,1,1,0,0], dtype='?')
    b[mask] = np.array([8,np.NA], maskna=True)
    assert_equal(a, [8,8,7,12,1])
    assert_equal(np.isna(b), [1,0,1,1,1])

    # TODO: fancy indexing is next...


def test_maskna_nonzero_1D():
    a = np.zeros((5,), maskna=True)

    # The nonzeros without any NAs
    assert_equal(np.count_nonzero(a), 0)
    assert_equal(np.nonzero(a)[0], [])
    a[2] = 3
    assert_equal(np.count_nonzero(a), 1)
    assert_equal(np.nonzero(a)[0], [2])
    a[3:] = 2
    assert_equal(np.count_nonzero(a), 3)
    assert_equal(np.nonzero(a)[0], [2,3,4])

    # The nonzeros with an NA
    a[2] = np.NA
    assert_(np.isna(np.count_nonzero(a)))
    assert_raises(ValueError, np.nonzero, a)


def test_maskna_take_1D():
    a = np.arange(5, maskna=True)
    b = np.arange(3)
    c = b.view(maskna=True)

    # Take without any NAs
    assert_equal(a.take([0,2,4]), [0,2,4])

    # Take without any NAs, into non-NA output parameter
    a.take([0,2,4], out=b)
    assert_equal(b, [0,2,4])

    # Take without any NAs, into NA output parameter
    b[...] = 1
    c[...] = np.NA
    a.take([0,2,4], out=c)
    assert_equal(c, [0,2,4])

    # Take with some NAs
    a[2] = np.NA
    a[3] = np.NA
    ret = a.take([0,2,4])
    assert_equal([ret[0], ret[2]], [0,4])
    assert_equal(np.isna(ret), [0,1,0])

    # Take with some NAs, into NA output parameter
    b[...] = 1
    c[...] = np.NA
    a.take([0,2,4], out=c)
    assert_equal(b, [0,1,4])
    assert_equal([c[0], c[2]], [0,4])
    assert_equal(np.isna(c), [0,1,0])

    c[...] = 1
    a.take([0,2,4], out=c)
    assert_equal(b, [0,1,4])
    assert_equal([c[0], c[2]], [0,4])
    assert_equal(np.isna(c), [0,1,0])

    # Take with an NA just at the start
    a = np.arange(5, maskna=True)
    a[0] = np.NA
    res = a.take([1,2,3,4])
    assert_equal(res, [1,2,3,4])


def test_maskna_ufunc_1D():
    a_orig = np.arange(3)
    a = a_orig.view(maskna=True)
    b_orig = np.array([5,4,3])
    b = b_orig.view(maskna=True)
    c_orig = np.array([0,0,0])
    c = c_orig.view(maskna=True)

    # An NA mask is produced if an operand has one
    res = a + b_orig
    assert_(res.flags.maskna)
    assert_equal(res, [5,5,5])

    res = b_orig + a
    assert_(res.flags.maskna)
    assert_equal(res, [5,5,5])

    # Can still output to a non-NA array if there are no NAs
    np.add(a, b, out=c_orig)
    assert_equal(c_orig, [5,5,5])

    # Should unmask everything if the output has NA support but
    # the inputs don't
    c_orig[...] = 0
    c[...] = np.NA
    np.add(a_orig, b_orig, out=c)
    assert_equal(c, [5,5,5])

    # If the input has NA support but an output parameter doesn't,
    # should work as long as the inputs contain no NAs
    c_orig[...] = 0
    np.add(a, b, out=c_orig)
    assert_equal(c_orig, [5,5,5])

    # An NA is produced if either operand has one
    a[0] = np.NA
    b[1] = np.NA
    res = a + b
    assert_equal(np.isna(res), [1,1,0])
    assert_equal(res[2], 5)

    # If the output contains NA, can't have out= parameter without
    # NA support
    assert_raises(ValueError, np.add, a, b, out=c_orig)

    # Divide in-place with NA
    a_orig = np.array([[3], [12.]])
    a = a_orig.view(maskna=True)
    a[0,0] = np.NA
    a /= 3
    # Shouldn't have touched the masked element
    assert_array_equal(a_orig, [[3], [4.]])
    assert_array_equal(a, [[np.NA], [4.]])
    # double-check assertions
    assert_equal(np.isna(a), [[1], [0]])
    assert_equal(a[~np.isna(a)], [4.])


def test_maskna_ufunc_sum_1D():
    check_maskna_ufunc_sum_1D(np.sum)


def test_maskna_ufunc_add_reduce_1D():
    check_maskna_ufunc_sum_1D(np.add.reduce)


def check_maskna_ufunc_sum_1D(sum_func):
    a = np.arange(3.0, maskna=True)
    b = np.array(0.5)
    c_orig = np.array(0.5)
    c = c_orig.view(maskna=True)

    # Since 'a' has no NA values, this should work
    sum_func(a, out=b)
    assert_equal(b, 3.0)
    b[...] = 7
    sum_func(a, skipna=True, out=b)
    assert_equal(b, 3.0)

    ret = sum_func(a)
    assert_equal(ret, 3.0)
    ret = sum_func(a, skipna=True)
    assert_equal(ret, 3.0)

    # With an NA value, the reduce should throw with the non-NA output param
    a[1] = np.NA
    assert_raises(ValueError, sum_func, a, out=b)

    # With an NA value, the output parameter can still be an NA-array
    c_orig[...] = 0.5
    sum_func(a, out=c)
    assert_equal(c_orig, 0.5)
    assert_(np.isna(c))

    # Should not touch the out= element when assigning NA
    b[...] = 1.0
    d = b.view(maskna=True)
    sum_func(a, out=d)
    assert_(np.isna(d))
    assert_equal(b, 1.0)

    # Without an output parameter, return NA
    ret = sum_func(a)
    assert_(np.isna(ret))

    # With 'skipna=True'
    ret = sum_func(a, skipna=True)
    assert_equal(ret, 2.0)

    # With 'skipna=True', and out= parameter
    b[...] = 0.5
    sum_func(a, skipna=True, out=b)
    assert_equal(b, 2.0)

    # With 'skipna=True', and out= parameter with a mask
    c[...] = 0.5
    c[...] = np.NA
    sum_func(a, skipna=True, out=c)
    assert_(not np.isna(c))
    assert_equal(c, 2.0)


def test_ufunc_max_1D():
    check_ufunc_max_1D(np.max)


def test_ufunc_maximum_reduce_1D():
    check_ufunc_max_1D(np.maximum.reduce)


def check_ufunc_max_1D(max_func):
    a_orig = np.array([0, 3, 2, 10, -1, 5, 7, -2])
    a = a_orig.view(maskna=True)

    # Straightforward reduce with no NAs
    b = max_func(a)
    assert_equal(b, 10)

    # Set the biggest value to NA
    a[3] = np.NA
    b = max_func(a)
    assert_(np.isna(b))

    # Skip the NA
    b = max_func(a, skipna=True)
    assert_(not b.flags.maskna)
    assert_(not np.isna(b))
    assert_equal(b, 7)

    # Set the first value to NA
    a[0] = np.NA
    b = max_func(a, skipna=True)
    assert_(not b.flags.maskna)
    assert_(not np.isna(b))
    assert_equal(b, 7)

    # Set all the values to NA - should raise the same error as
    # for an empty array
    a[...] = np.NA
    assert_raises(ValueError, max_func, a, skipna=True)


def test_ufunc_skipna_max_3D():
    check_ufunc_skipna_max_3D(np.max)


def test_ufunc_skipna_maximum_reduce_3D():
    check_ufunc_skipna_max_3D(np.maximum.reduce)


def check_ufunc_skipna_max_3D(max_func):
    a_orig = np.array([[[29,  6, 24, 11, 24],
                    [17, 26, 10, 29, 21],
                    [ 4,  4,  7,  9, 30],
                    [ 9, 20,  5, 12, 23]],
                   [[ 8,  9, 10, 31, 22],
                    [ 5, 20,  2, 29, 27],
                    [21, 22, 13, 30, 20],
                    [24, 27,  9, 20, 31]],
                   [[14,  0, 13, 11, 22],
                    [ 0, 16, 16, 14,  2],
                    [ 0,  2,  1, 29, 12],
                    [24, 25, 12, 11,  9]]])
    a = a_orig.view(maskna=True)
    b = a_orig.copy()

    def check_all_axis_combos(x, y, badaxes=()):
        if 0 not in badaxes:
            res = max_func(x, axis=0, skipna=True)
            assert_array_equal(res, max_func(y, axis=0, skipna=True))
        if 1 not in badaxes:
            res = max_func(x, axis=1, skipna=True)
            assert_array_equal(res, max_func(y, axis=1, skipna=True))
        if 2 not in badaxes:
            res = max_func(x, axis=2, skipna=True)
            assert_array_equal(res, max_func(y, axis=2, skipna=True))
        res = max_func(x, axis=(0,1), skipna=True)
        assert_array_equal(res, max_func(y, axis=(0,1), skipna=True))
        res = max_func(x, axis=(0,2), skipna=True)
        assert_array_equal(res, max_func(y, axis=(0,2), skipna=True))
        res = max_func(x, axis=(1,2), skipna=True)
        assert_array_equal(res, max_func(y, axis=(1,2), skipna=True))
        res = max_func(x, axis=(0,1,2), skipna=True)
        assert_array_equal(res, max_func(y, axis=(0,1,2), skipna=True))

    # Straightforward reduce with no NAs
    check_all_axis_combos(a, b)

    # Set a few values in 'a' to NA, and set the corresponding
    # values in 'b' to -1 to definitely eliminate them from the maximum
    for coord in [(0,1,2), (1,2,2), (0,2,4), (2,1,0)]:
        a[coord] = np.NA
        b[coord] = -1
    check_all_axis_combos(a, b)

    # Set a few more values in 'a' to NA
    for coord in [(2,1,1), (2,1,2), (2,1,3), (0,0,4), (0,3,4)]:
        a[coord] = np.NA
        b[coord] = -1
    check_all_axis_combos(a, b)

    # Set it so that there's a full set of NAs along the third dimension
    for coord in [(2,1,4)]:
        a[coord] = np.NA
        b[coord] = -1
    check_all_axis_combos(a, b, badaxes=(2,))
    assert_raises(ValueError, max_func, a, axis=2, skipna=True)

    # Set it so that there's a full set of NAs along the second dimension
    for coord in [(0,1,4)]:
        a[coord] = np.NA
        b[coord] = -1
    check_all_axis_combos(a, b, badaxes=(1,2))
    assert_raises(ValueError, max_func, a, axis=1, skipna=True)
    assert_raises(ValueError, max_func, a, axis=2, skipna=True)


def test_ufunc_ndarray_any():
    a = np.array([0,0,0,0,0], dtype='?', maskna=True)
    assert_array_equal(a.any(), False)
    a[0] = np.NA
    assert_array_equal(a.any(), np.NA)
    assert_array_equal(a.any(skipna=True), False)
    a[0] = 0
    a[-1] = np.NA
    assert_array_equal(a.any(), np.NA)
    assert_array_equal(a.any(skipna=True), False)
    a[0] = 1
    assert_array_equal(a.any(), True)
    assert_array_equal(a.any(skipna=True), True)
    a[-1] = 1
    a[-2] = np.NA
    assert_array_equal(a.any(), True)
    assert_array_equal(a.any(skipna=True), True)

    a = np.array([[0,0,0],[0,np.NA,0]], dtype='?')
    assert_array_equal(a.any(axis=0), [False, np.NA, False])
    assert_array_equal(a.any(axis=1), [False, np.NA])
    assert_array_equal(a.any(axis=0, skipna=True), [False, False, False])
    assert_array_equal(a.any(axis=1, skipna=True), [False, False])

    a[0,1] = 1
    assert_array_equal(a.any(axis=0), [False, True, False])
    assert_array_equal(a.any(axis=1), [True, np.NA])
    assert_array_equal(a.any(axis=0, skipna=True), [False, True, False])
    assert_array_equal(a.any(axis=1, skipna=True), [True, False])

    a[0,1] = np.NA
    a[1,1] = 0
    a[0,2] = 1
    assert_array_equal(a.any(axis=0), [False, np.NA, True])
    assert_array_equal(a.any(axis=1), [True, False])
    assert_array_equal(a.any(axis=0, skipna=True), [False, False, True])
    assert_array_equal(a.any(axis=1, skipna=True), [True, False])


def test_ufunc_ndarray_all():
    a = np.array([1,1,1,1,1], dtype='?', maskna=True)
    assert_array_equal(a.all(), True)
    a[0] = np.NA
    assert_array_equal(a.all(), np.NA)
    assert_array_equal(a.all(skipna=True), True)
    a[0] = 1
    a[-1] = np.NA
    assert_array_equal(a.all(), np.NA)
    assert_array_equal(a.all(skipna=True), True)
    a[0] = 0
    assert_array_equal(a.all(), False)
    assert_array_equal(a.all(skipna=True), False)
    a[-1] = 0
    a[-2] = np.NA
    assert_array_equal(a.all(), False)
    assert_array_equal(a.all(skipna=True), False)

    a = np.array([[1,1,1],[1,np.NA,1]], dtype='?')
    assert_array_equal(a.all(axis=0), [True, np.NA, True])
    assert_array_equal(a.all(axis=1), [True, np.NA])
    assert_array_equal(a.all(axis=0, skipna=True), [True, True, True])
    assert_array_equal(a.all(axis=1, skipna=True), [True, True])

    a[0,1] = 0
    assert_array_equal(a.all(axis=0), [True, False, True])
    assert_array_equal(a.all(axis=1), [False, np.NA])
    assert_array_equal(a.all(axis=0, skipna=True), [True, False, True])
    assert_array_equal(a.all(axis=1, skipna=True), [False, True])

    a[0,1] = np.NA
    a[1,1] = 1
    a[0,2] = 0
    assert_array_equal(a.all(axis=0), [True, np.NA, False])
    assert_array_equal(a.all(axis=1), [False, True])
    assert_array_equal(a.all(axis=0, skipna=True), [True, True, False])
    assert_array_equal(a.all(axis=1, skipna=True), [False, True])


def test_count_reduce_items():
    # np.count_reduce_items

    # When skipna is False, it should always return the
    # product of the reduction axes as a NumPy intp scalar
    a = np.zeros((2,3,4))

    res = np.count_reduce_items(a)
    assert_equal(res, 24)
    assert_equal(type(res), np.intp)

    res = np.count_reduce_items(a, axis=0)
    assert_equal(res, 2)
    assert_equal(type(res), np.intp)

    res = np.count_reduce_items(a, axis=(1,2))
    assert_equal(res, 12)
    assert_equal(type(res), np.intp)

    # This still holds if 'a' has an NA mask and some NA values
    a = np.zeros((2,3,4), maskna=True)
    a[1,2,2] = np.NA
    a[0,1,2] = np.NA
    a[1,0,3] = np.NA

    res = np.count_reduce_items(a)
    assert_equal(res, 24)
    assert_equal(type(res), np.intp)

    res = np.count_reduce_items(a, axis=0)
    assert_equal(res, 2)
    assert_equal(type(res), np.intp)

    res = np.count_reduce_items(a, axis=(1,2))
    assert_equal(res, 12)
    assert_equal(type(res), np.intp)

    # If skipna is True, but the array has no NA mask, the result
    # should still be the product of the reduction axes
    a = np.zeros((2,3,4))

    res = np.count_reduce_items(a, skipna=True)
    assert_equal(res, 24)
    assert_equal(type(res), np.intp)

    res = np.count_reduce_items(a, axis=0, skipna=True)
    assert_equal(res, 2)
    assert_equal(type(res), np.intp)

    res = np.count_reduce_items(a, axis=(1,2), skipna=True)
    assert_equal(res, 12)
    assert_equal(type(res), np.intp)

    # Finally, when skipna is True AND the array has an NA mask,
    # we get an array of counts
    a = np.zeros((2,3,4), maskna=True)
    a[1,2,2] = np.NA
    a[0,1,2] = np.NA
    a[1,0,3] = np.NA

    # When doing a full reduction, should still get the scalar
    res = np.count_reduce_items(a, skipna=True)
    assert_equal(res, 21)
    assert_equal(res.dtype, np.dtype(np.intp))

    res = np.count_reduce_items(a, axis=0, skipna=True)
    assert_equal(res, [[2,2,2,1], [2,2,1,2], [2,2,1,2]])
    assert_equal(res.dtype, np.dtype(np.intp))

    res = np.count_reduce_items(a, axis=(1,2), skipna=True)
    assert_equal(res, [11,10])
    assert_equal(res.dtype, np.dtype(np.intp))


def test_array_maskna_clip_method():
    # ndarray.clip
    a = np.array([2, np.NA, 10, 4, np.NA, 7], maskna=True)

    b = np.clip(a, 3, None)
    assert_equal(np.isna(b), [0,1,0,0,1,0])
    assert_equal(b[~np.isna(b)], [3, 10, 4, 7])

    res = np.clip(a, None, 6)
    assert_equal(np.isna(res), [0,1,0,0,1,0])
    assert_equal(res[~np.isna(res)], [2, 6, 4, 6])

    res = np.clip(a, 4, 7)
    assert_equal(np.isna(res), [0,1,0,0,1,0])
    assert_equal(res[~np.isna(res)], [4, 7, 4, 7])


def test_array_maskna_max_min_ptp_methods():
    # ndarray.max, ndarray.min, ndarray.ptp
    a = np.array([[2, np.NA, 10],
                  [4, 8, 7],
                  [12, 4, np.NA]], maskna=True)

    res = a.max(axis=0)
    assert_equal(np.isna(res), [0,1,1])
    assert_equal(res[~np.isna(res)], [12])

    res = a.max(axis=-1)
    assert_equal(np.isna(res), [1,0,1])
    assert_equal(res[~np.isna(res)], [8])

    res = a.min(axis=0)
    assert_equal(np.isna(res), [0,1,1])
    assert_equal(res[~np.isna(res)], [2])

    res = a.min(axis=-1)
    assert_equal(np.isna(res), [1,0,1])
    assert_equal(res[~np.isna(res)], [4])

    res = a.ptp(axis=0)
    assert_equal(np.isna(res), [0,1,1])
    assert_equal(res[~np.isna(res)], [10])

    res = a.ptp(axis=-1)
    assert_equal(np.isna(res), [1,0,1])
    assert_equal(res[~np.isna(res)], [4])


def test_array_maskna_sum_prod_methods():
    # ndarray.sum, ndarray.prod
    a = np.array([[2, np.NA, 10],
                  [4, 8, 7],
                  [12, 4, np.NA],
                  [3, 2, 5]], maskna=True)

    res = a.sum(axis=0)
    assert_equal(np.isna(res), [0,1,1])
    assert_equal(res[~np.isna(res)], [21])

    res = a.sum(axis=-1)
    assert_equal(np.isna(res), [1,0,1,0])
    assert_equal(res[~np.isna(res)], [19,10])

    res = a.prod(axis=0)
    assert_equal(np.isna(res), [0,1,1])
    assert_equal(res[~np.isna(res)], [2*4*12*3])

    res = a.prod(axis=-1)
    assert_equal(np.isna(res), [1,0,1,0])
    assert_equal(res[~np.isna(res)], [4*8*7,3*2*5])

    # Check also with Fortran-order
    a = np.array([[2, np.NA, 10],
                  [4, 8, 7],
                  [12, 4, np.NA],
                  [3, 2, 5]], maskna=True, order='F')

    res = a.sum(axis=0)
    assert_equal(np.isna(res), [0,1,1])
    assert_equal(res[~np.isna(res)], [21])

    res = a.sum(axis=-1)
    assert_equal(np.isna(res), [1,0,1,0])
    assert_equal(res[~np.isna(res)], [19,10])


def test_array_maskna_std_mean_methods():
    # ndarray.std, ndarray.mean
    a = np.array([[2, np.NA, 10],
                  [4, 8, 7],
                  [12, 4, np.NA]], maskna=True)

    res = a.mean(axis=0)
    assert_equal(np.isna(res), [0,1,1])
    assert_equal(res[~np.isna(res)], [np.array([2,4,12]).mean()])

    res = a.mean(axis=-1)
    assert_equal(np.isna(res), [1,0,1])
    assert_equal(res[~np.isna(res)], [np.array([4,8,7]).mean()])

    res = a.std(axis=0)
    assert_equal(np.isna(res), [0,1,1])
    assert_equal(res[~np.isna(res)], [np.array([2,4,12]).std()])

    res = a.std(axis=-1)
    assert_equal(np.isna(res), [1,0,1])
    assert_equal(res[~np.isna(res)], [np.array([4,8,7]).std()])


def test_array_maskna_conjugate_method():
    # ndarray.conjugate
    a = np.array([1j, 2+4j, np.NA, 2-1.5j, np.NA], maskna=True)

    b = a.conjugate()
    assert_equal(np.isna(b), [0,0,1,0,1])
    assert_equal(b[~np.isna(b)], [-1j, 2-4j, 2+1.5j])


def test_array_maskna_diagonal():
    # ndarray.diagonal
    a = np.arange(6, maskna=True)
    a.shape = (2,3)
    a[0,1] = np.NA

    # Should produce a view into a
    res = a.diagonal()
    assert_(res.base is a)
    assert_(res.flags.maskna)
    assert_(not res.flags.ownmaskna)
    assert_equal(res, [0, 4])

    res = a.diagonal(-1)
    assert_equal(res, [3])

    res = a.diagonal(-2)
    assert_equal(res, [])

    # This diagonal has the NA
    res = a.diagonal(1)
    assert_equal(np.isna(res), [1,0])
    assert_equal(res[~np.isna(res)], [5])

    res = a.diagonal(2)
    assert_equal(res, [2])

    res = a.diagonal(3)
    assert_equal(res, [])


def test_array_maskna_concatenate():
    # np.concatenate
    a = np.arange(6, maskna=True, dtype='i4').reshape(2,3)
    a[1,0] = np.NA

    b = np.array([[12],[13]], dtype='i4')
    res = np.concatenate([a, b], axis=1)
    assert_equal(np.isna(res), [[0,0,0,0], [1,0,0,0]])
    assert_equal(res[~np.isna(res)], [0,1,2,12,4,5,13])
    assert_equal(res.strides, (16, 4))

    b = np.array([[10, np.NA, 11]], maskna=True, dtype='i4')
    res = np.concatenate([a,b], axis=0)
    assert_equal(np.isna(res), [[0,0,0], [1,0,0], [0,1,0]])
    assert_equal(res[~np.isna(res)], [0,1,2,4,5,10,11])
    assert_equal(res.strides, (12, 4))

    b = np.array([[np.NA, 10]], order='F', maskna=True, dtype='i4')
    res = np.concatenate([a.T, b], axis=0)
    assert_equal(np.isna(res), [[0,1], [0,0], [0,0], [1,0]])
    assert_equal(res[~np.isna(res)], [0,1,4,2,5,10])
    assert_equal(res.strides, (4, 16))


def test_array_maskna_column_stack():
    # np.column_stack
    a = np.array((1,2,3), maskna=True)
    b = np.array((2,3,4), maskna=True)
    b[2] = np.NA
    res = np.column_stack((a,b))
    assert_equal(np.isna(res), [[0,0], [0,0], [0,1]])
    assert_equal(res[~np.isna(res)], [1,2,2,3,3])


def test_array_maskna_compress():
    # ndarray.compress
    a = np.arange(5., maskna=True)
    a[0] = np.NA

    mask = np.array([0,1,1,1,1], dtype='?')
    res = a.compress(mask)
    assert_equal(res, [1,2,3,4])


def test_array_maskna_squeeze():
    # np.squeeze
    a = np.zeros((1,3,1,1,4,2,1), maskna=True)
    a[0,1,0,0,3,0,0] = np.NA

    res = np.squeeze(a)
    assert_equal(res.shape, (3,4,2))
    assert_(np.isna(res[1,3,0]))

    res = np.squeeze(a, axis=(0,2,6))
    assert_equal(res.shape, (3,1,4,2))
    assert_(np.isna(res[1,0,3,0]))


def test_array_maskna_mean():
    # np.mean

    # With an NA mask, but no NA
    a = np.arange(6, maskna=True).reshape(2,3)

    res = np.mean(a)
    assert_equal(res, 2.5)
    res = np.mean(a, axis=0)
    assert_equal(res, [1.5, 2.5, 3.5])

    # With an NA and skipna=False
    a = np.arange(6, maskna=True).reshape(2,3)
    a[0,1] = np.NA

    res = np.mean(a)
    assert_(type(res) is np.NAType)

    res = np.mean(a, axis=0)
    assert_array_equal(res, [1.5, np.NA, 3.5])

    res = np.mean(a, axis=1)
    assert_array_equal(res, [np.NA, 4.0])

    # With an NA and skipna=True
    res = np.mean(a, skipna=True)
    assert_almost_equal(res, 2.8)

    res = np.mean(a, axis=0, skipna=True)
    assert_array_equal(res, [1.5, 4.0, 3.5])

    res = np.mean(a, axis=1, skipna=True)
    assert_array_equal(res, [1.0, 4.0])


def test_array_maskna_var_std():
    # np.var, np.std

    # With an NA and skipna=False
    a = np.arange(6, maskna=True).reshape(2,3)
    a[0,1] = np.NA

    res = np.var(a)
    assert_(type(res) is np.NAType)
    res = np.std(a)
    assert_(type(res) is np.NAType)

    res = np.var(a, axis=0)
    assert_array_equal(res, [2.25, np.NA, 2.25])
    res = np.std(a, axis=0)
    assert_array_equal(res, [1.5, np.NA, 1.5])

    res = np.var(a, axis=1)
    assert_array_almost_equal(res, [np.NA, 0.66666666666666663])
    res = np.std(a, axis=1)
    assert_array_almost_equal(res, [np.NA, 0.81649658092772603])

    # With an NA and skipna=True
    a = np.arange(6, maskna=True).reshape(2,3)
    a[0,1] = np.NA

    res = np.var(a, skipna=True)
    assert_almost_equal(res, 2.96)
    res = np.std(a, skipna=True)
    assert_almost_equal(res, 1.7204650534085253)

    res = np.var(a, axis=0, skipna=True)
    assert_array_equal(res, [2.25, 0, 2.25])
    res = np.std(a, axis=0, skipna=True)
    assert_array_equal(res, [1.5, 0, 1.5])

    res = np.var(a, axis=1, skipna=True)
    assert_array_almost_equal(res, [1.0, 0.66666666666666663])
    res = np.std(a, axis=1, skipna=True)
    assert_array_almost_equal(res, [1.0, 0.81649658092772603])


def test_array_maskna_linspace_logspace():
    # np.linspace, np.logspace

    a = np.linspace(2.0, 3.0, num=5)
    b = np.linspace(2.0, 3.0, num=5, maskna=True)
    assert_equal(a, b)
    assert_(not a.flags.maskna)
    assert_(b.flags.maskna)

    a = np.logspace(2.0, 3.0, num=4)
    b = np.logspace(2.0, 3.0, num=4, maskna=True)
    assert_equal(a, b)
    assert_(not a.flags.maskna)
    assert_(b.flags.maskna)


def test_array_maskna_eye_identity():
    # np.eye

    # By default there should be no NA mask
    a = np.eye(3)
    assert_(not a.flags.maskna)
    a = np.identity(3)
    assert_(not a.flags.maskna)

    a = np.eye(3, maskna=True)
    assert_(a.flags.maskna)
    assert_(a.flags.ownmaskna)
    assert_equal(a, np.eye(3))

    a = np.eye(3, k=2, maskna=True)
    assert_(a.flags.maskna)
    assert_(a.flags.ownmaskna)
    assert_equal(a, np.eye(3, k=2))

    a = np.identity(3, maskna=True)
    assert_(a.flags.maskna)
    assert_(a.flags.ownmaskna)
    assert_equal(a, np.identity(3))

if __name__ == "__main__":
    run_module_suite()
