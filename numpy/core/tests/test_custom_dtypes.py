import pytest

import numpy as np
from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import (
    _discover_array_parameters as discover_array_params, _get_sfloat_dtype)


SF = _get_sfloat_dtype()


class TestSFloat:
    def _get_array(self, scaling, aligned=True):
        if not aligned:
            a = np.empty(3*8 + 1, dtype=np.uint8)[1:]
            a = a.view(np.float64)
            a[:] = [1., 2., 3.]
        else:
            a = np.array([1., 2., 3.])

        a *= 1./scaling  # the casting code also uses the reciprocal.
        return a.view(SF(scaling))

    def test_sfloat_rescaled(self):
        sf = SF(1.)
        sf2 = sf.scaled_by(2.)
        assert sf2.get_scaling() == 2.
        sf6 = sf2.scaled_by(3.)
        assert sf6.get_scaling() == 6.

    def test_class_discovery(self):
        # This does not test much, since we always discover the scaling as 1.
        # But most of NumPy (when writing) does not understand DType classes
        dt, _ = discover_array_params([1., 2., 3.], dtype=SF)
        assert dt == SF(1.)

    @pytest.mark.parametrize("scaling", [1., -1., 2.])
    def test_scaled_float_from_floats(self, scaling):
        a = np.array([1., 2., 3.], dtype=SF(scaling))

        assert a.dtype.get_scaling() == scaling
        assert_array_equal(scaling * a.view(np.float64), [1., 2., 3.])

    def test_repr(self):
        # Check the repr, mainly to cover the code paths:
        assert repr(SF(scaling=1.)) == "_ScaledFloatTestDType(scaling=1.0)"

    @pytest.mark.parametrize("scaling", [1., -1., 2.])
    def test_sfloat_from_float(self, scaling):
        a = np.array([1., 2., 3.]).astype(dtype=SF(scaling))

        assert a.dtype.get_scaling() == scaling
        assert_array_equal(scaling * a.view(np.float64), [1., 2., 3.])

    @pytest.mark.parametrize("aligned", [True, False])
    @pytest.mark.parametrize("scaling", [1., -1., 2.])
    def test_sfloat_getitem(self, aligned, scaling):
        a = self._get_array(1., aligned)
        assert a.tolist() == [1., 2., 3.]

    @pytest.mark.parametrize("aligned", [True, False])
    def test_sfloat_casts(self, aligned):
        a = self._get_array(1., aligned)

        assert np.can_cast(a, SF(-1.), casting="equiv")
        assert not np.can_cast(a, SF(-1.), casting="no")
        na = a.astype(SF(-1.))
        assert_array_equal(-1 * na.view(np.float64), a.view(np.float64))

        assert np.can_cast(a, SF(2.), casting="same_kind")
        assert not np.can_cast(a, SF(2.), casting="safe")
        a2 = a.astype(SF(2.))
        assert_array_equal(2 * a2.view(np.float64), a.view(np.float64))

    @pytest.mark.parametrize("aligned", [True, False])
    def test_sfloat_cast_internal_errors(self, aligned):
        a = self._get_array(2e300, aligned)

        with pytest.raises(TypeError,
                match="error raised inside the core-loop: non-finite factor!"):
            a.astype(SF(2e-300))

    def test_sfloat_promotion(self):
        assert np.result_type(SF(2.), SF(3.)) == SF(3.)
        assert np.result_type(SF(3.), SF(2.)) == SF(3.)
        # Float64 -> SF(1.) and then promotes normally, so both of this work:
        assert np.result_type(SF(3.), np.float64) == SF(3.)
        assert np.result_type(np.float64, SF(0.5)) == SF(1.)

        # Test an undefined promotion:
        with pytest.raises(TypeError):
            np.result_type(SF(1.), np.int64)

    def test_basic_multiply(self):
        a = self._get_array(2.)
        b = self._get_array(4.)

        res = a * b
        # multiplies dtype scaling and content separately:
        assert res.dtype.get_scaling() == 8.
        expected_view = a.view(np.float64) * b.view(np.float64)
        assert_array_equal(res.view(np.float64), expected_view)

    def test_basic_addition(self):
        a = self._get_array(2.)
        b = self._get_array(4.)

        res = a + b
        # addition uses the type promotion rules for the result:
        assert res.dtype == np.result_type(a.dtype, b.dtype)
        expected_view = (a.astype(res.dtype).view(np.float64) +
                         b.astype(res.dtype).view(np.float64))
        assert_array_equal(res.view(np.float64), expected_view)

    def test_addition_cast_safety(self):
        """The addition method is special for the scaled float, because it
        includes the "cast" between different factors, thus cast-safety
        is influenced by the implementation.
        """
        a = self._get_array(2.)
        b = self._get_array(-2.)
        c = self._get_array(3.)

        # sign change is "equiv":
        np.add(a, b, casting="equiv")
        with pytest.raises(TypeError):
            np.add(a, b, casting="no")

        # Different factor is "same_kind" (default) so check that "safe" fails
        with pytest.raises(TypeError):
            np.add(a, c, casting="safe")

        # Check that casting the output fails also (done by the ufunc here)
        with pytest.raises(TypeError):
            np.add(a, a, out=c, casting="safe")
