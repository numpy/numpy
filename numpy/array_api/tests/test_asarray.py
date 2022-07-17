import numpy as np


def test_dtype_identity():
    """Confirm the intended behavior for ``asarray`` results.

    The result of ``asarray()`` should have the dtype provided through the
    keyword argument, when used. This forces unique array handles to be
    produced for unique np.dtype objects, but (for equivalent dtypes), the
    underlying data (the base object) is shared with the original array object.

    Ref https://github.com/numpy/numpy/issues/1468
    """
    int_array = np.array([1, 2, 3], dtype='i')
    assert np.asarray(int_array) is int_array

    # The character code resolves to the singleton dtype object provided
    # by the numpy package.
    assert np.asarray(int_array, dtype='i') is int_array

    # Derive a dtype from n.dtype('i'), but add a metadata object to force
    # the dtype to be distinct.
    unequal_type = np.dtype('i', metadata={'spam': True})
    annotated_int_array = np.asarray(int_array, dtype=unequal_type)
    assert annotated_int_array is not int_array
    assert annotated_int_array.base is int_array

    # These ``asarray()`` calls may produce a new view or a copy,
    # but never the same object.
    long_int_array = np.asarray(int_array, dtype='l')
    assert long_int_array is not int_array
    assert np.asarray(int_array, dtype='q') is not int_array
    assert np.asarray(long_int_array, dtype='q') is not long_int_array
    assert long_int_array is not np.asarray(int_array, dtype='l')
    assert long_int_array.base is np.asarray(int_array, dtype='l').base

    equivalent_requirement = np.dtype('i', metadata={'spam': True})
    annotated_int_array_alt = np.asarray(annotated_int_array,
                                         dtype=equivalent_requirement)
    # The descriptors are equivalent, but we have created
    # distinct dtype instances.
    assert unequal_type == equivalent_requirement
    assert unequal_type is not equivalent_requirement
    assert annotated_int_array_alt is not annotated_int_array
    assert annotated_int_array_alt.dtype is equivalent_requirement
