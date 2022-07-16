import numpy as np


def test_fast_return():
    """"""
    a = np.array([1, 2, 3], dtype='i')
    assert np.asarray(a) is a
    assert np.asarray(a, dtype='i') is a
    # This may produce a new view or a copy, but is never the same object.
    assert np.asarray(a, dtype='l') is not a

    unequal_type = np.dtype('i', metadata={'spam': True})
    b = np.asarray(a, dtype=unequal_type)
    assert b is not a
    assert b.base is a

    equivalent_requirement = np.dtype('i', metadata={'spam': True})
    c = np.asarray(b, dtype=equivalent_requirement)
    # The descriptors are equivalent, but we have created
    # distinct dtype instances.
    assert unequal_type == equivalent_requirement
    assert unequal_type is not equivalent_requirement
    assert c is not b
    assert c.dtype is equivalent_requirement
