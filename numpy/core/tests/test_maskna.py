import numpy as np
from numpy.compat import asbytes
from numpy.testing import *
import sys, warnings

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
    # Without the 'maskna=True', produces an object array
    a = np.array([1.0, 2.0, np.NA, 7.0])
    assert_equal(a.dtype, np.dtype('O'))
    assert_equal(type(a[2]), np.NAType)

    # From np.NA as a straight scalar
    a = np.array(np.NA, maskna=True)
    assert_equal(type(a), np.ndarray)
    assert_(np.isna(a))

    # As a special case, converting np.NA to an array produces
    # a zero-dimensional masked array
    a = np.array(np.NA)
    assert_equal(type(a), np.ndarray)
    assert_(np.isna(a))

    a = np.zeros((3,))
    assert_(not a.flags.maskna)
    a = np.zeros((3,), maskna=True)
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
    assert_(c.base is a)
    assert_(c.flags.ownmaskna)
    assert_(not (c is b_view))

def test_array_maskna_reshape():
    # Simple reshape 1D -> 2D
    a = np.arange(6, maskna=True)
    a[1] = np.NA
    a[5] = np.NA

    b = a.reshape(2,3)
    assert_(b.base is a)
    assert_equal(b.shape, (2,3))
    assert_(b.flags.maskna)
    assert_(not b.flags.ownmaskna)
    assert_equal(np.isna(b), [[0,1,0],[0,0,1]])

    b = a.reshape(2,3,order='F')
    assert_equal(b.shape, (2,3))
    assert_(b.flags.maskna)
    assert_(not b.flags.ownmaskna)
    assert_equal(np.isna(b), [[0,0,0],[1,0,1]])

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
    b[mask] = [8,np.NA]
    assert_equal(a, [8,8,7,12,1])
    assert_equal(np.isna(b), [1,0,1,1,1])

    # TODO: fancy indexing is next...

def test_ufunc_1D():
    a = np.arange(3, maskna=True)
    b = np.arange(3)

    # An NA mask is produced if an operand has one
    c = a + b
    assert_(c.flags.maskna)
    #assert_equal(c, [0,2,4])

if __name__ == "__main__":
    run_module_suite()
