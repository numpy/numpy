import pytest

import numpy as np
from numpy.testing import assert_array_equal


SF = np.core._multiarray_umath._get_sfloat_dtype()


@pytest.mark.parametrize("scaling", [1., -1., 2.])
def test_scaled_float_from_floats(scaling):
    a = np.array([1., 2., 3.], dtype=SF(scaling))

    assert a.dtype.get_scaling() == scaling
    assert_array_equal(scaling * a.view(np.float64), np.array([1., 2., 3.]))


@pytest.mark.parametrize("scaling", [1., -1., 2.])
def test_sfloat_from_float(scaling):
    a = np.array([1., 2., 3.]).astype(dtype=SF(scaling))

    assert a.dtype.get_scaling() == scaling
    assert_array_equal(scaling * a.view(np.float64), np.array([1., 2., 3.]))


def _get_array(scaling, aligned=True):
    if not aligned:
        a = np.empty(3*8 + 1, dtype=np.uint8)[1:]
        a = a.view(np.float64)
        a[:] = [1., 2., 3.]
    else:
        a = np.array([1., 2., 3.])

    a *= 1./scaling  # the casting code also uses the reciprocal.
    return a.view(SF(scaling))


@pytest.mark.parametrize("aligned", [True, False])
def test_sfloat_casts(aligned):
    a = _get_array(1., aligned)

    assert np.can_cast(a, SF(-1.), casting="equiv")
    assert not np.can_cast(a, SF(-1.), casting="no")
    na = a.astype(SF(-1.))
    assert_array_equal(-1 * na.view(np.float64), a.view(np.float64))

    assert np.can_cast(a, SF(2.), casting="same_kind")
    assert not np.can_cast(a, SF(2.), casting="safe")
    a2 = a.astype(SF(2.))
    assert_array_equal(2 * a2.view(np.float64), a.view(np.float64))


@pytest.mark.parametrize("aligned", [True, False])
def test_sfloat_cast_internal_errors(aligned):
    a = _get_array(2e300, aligned)

    with pytest.raises(TypeError,
            match="error raised inside the core-loop: non-finite factor!"):
        a.astype(SF(2e-300))

