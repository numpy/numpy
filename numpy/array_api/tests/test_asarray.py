import itertools

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
    # Create an equivalent descriptor with a new and distinct dtype instance.
    equivalent_requirement = np.dtype('i', metadata={'spam': True})
    annotated_int_array_alt = np.asarray(annotated_int_array,
                                         dtype=equivalent_requirement)
    assert unequal_type == equivalent_requirement
    assert unequal_type is not equivalent_requirement
    assert annotated_int_array_alt is not annotated_int_array
    assert annotated_int_array_alt.dtype is equivalent_requirement

    # Check the same logic for a pair of C types whose equivalence may vary
    # between computing environments.
    # Find an equivalent pair.
    integer_type_codes = ('i', 'l', 'q')
    integer_dtypes = [np.dtype(code) for code in integer_type_codes]
    typeA = None
    typeB = None
    for typeA, typeB in itertools.permutations(integer_dtypes, r=2):
        if typeA == typeB:
            assert typeA is not typeB
            break
    assert isinstance(typeA, np.dtype) and isinstance(typeB, np.dtype)

    # These ``asarray()`` calls may produce a new view or a copy,
    # but never the same object.
    long_int_array = np.asarray(int_array, dtype='l')
    long_long_int_array = np.asarray(int_array, dtype='q')
    assert long_int_array is not int_array
    assert long_long_int_array is not int_array
    assert np.asarray(long_int_array, dtype='q') is not long_int_array
    array_a = np.asarray(int_array, dtype=typeA)
    assert typeA == typeB
    assert typeA is not typeB
    assert array_a.dtype is typeA
    assert array_a is not np.asarray(array_a, dtype=typeB)
    assert np.asarray(array_a, dtype=typeB).dtype is typeB
    assert array_a is np.asarray(array_a, dtype=typeB).base
