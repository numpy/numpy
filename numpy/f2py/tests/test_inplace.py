
import pytest

import numpy as np
from numpy.f2py.tests import util
from numpy.testing import assert_array_equal


@pytest.mark.slow
class TestInplace(util.F2PyTest):
    sources = [util.getpath("tests", "src", "inplace", "foo.f")]

    @pytest.mark.parametrize("func", ["inplace", "inplace_out"])
    @pytest.mark.parametrize("writeable", ["writeable", "readonly"])
    @pytest.mark.parametrize("view", [
        None, (), (slice(None, 2, None), slice(None, None, 2))])
    @pytest.mark.parametrize("dtype", ["f4", "f8"])
    def test_inplace(self, dtype, view, writeable, func):
        # Test inplace modifications of an input array.
        a = np.arange(12.0, dtype=dtype).reshape((3, 4)).copy()
        a.flags.writeable = writeable == "writeable"
        k = a if view is None else a[view]

        ffunc = getattr(self.module, func)
        if not a.flags.writeable:
            with pytest.raises(ValueError, match="WRITEBACKIFCOPY base is read-only"):
                ffunc(k)
            return

        ref_k = k
        exp_copy = k.copy()
        exp_k = k ** 2
        exp_a = a.copy()
        exp_a[view or ()] = exp_k
        if func == "inplace_out":
            kout, copy = ffunc(k)
            assert kout is k
        else:
            copy = ffunc(k)
        assert_array_equal(copy, exp_copy)
        assert k is ref_k
        assert np.allclose(k, exp_k)
        assert np.allclose(a, exp_a)

    @pytest.mark.parametrize("func", ["inplace", "inplace_out"])
    def test_inplace_error(self, func):
        ffunc = getattr(self.module, func)
        with pytest.raises(ValueError, match="input.*not compatible"):
            ffunc(np.array([1 + 1j]))
