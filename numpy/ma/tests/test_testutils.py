from numpy.testing import assert_raises
from numpy.testing import run_module_suite
from numpy.ma.core import MaskError
import numpy as np
import numpy.ma as ma
from numpy.ma.testutils import assert_equal


class TestAssertEqual:
    def test_dictionary(self):
        dict_1 = {'a': 'b', 'c': 'd'}
        dict_2 = {'a': 'b', 'c': 'd', 'e': 'f'}
        dict_3 = {'a': 'a', 'b': 'b', 'c': 'c'}
        tp_1 = ('foo')
        arr_1 = np.array([1, 2])

        assert_equal({}, {})
        assert_equal(dict_1, dict_1)
        assert_equal(dict_2, dict_2)

        assert_raises(AssertionError, assert_equal, dict_1, dict_2)
        assert_raises(AssertionError, assert_equal, dict_2, dict_3)
        assert_raises(AssertionError, assert_equal, dict_1, dict_3)
        assert_raises(AssertionError, assert_equal, dict_1, 1)
        assert_raises(AssertionError, assert_equal, dict_1, {})
        assert_raises(AssertionError, assert_equal, dict_1, [])
        assert_raises(AssertionError, assert_equal, dict_1, tp_1)
        assert_raises(AssertionError, assert_equal, dict_1, arr_1)

    def test_nan(self):
        assert_equal(np.nan, np.nan)

        assert_raises(AssertionError, assert_equal, np.nan, 1)
        assert_raises(AssertionError, assert_equal, np.nan, {})
        assert_raises(AssertionError, assert_equal, np.nan, [1])
        assert_raises(AssertionError, assert_equal, np.nan, 1.0)

    def test_list(self):
        assert_equal([], [])
        assert_equal([1], [1])
        assert_raises(AssertionError, assert_equal, [1], [2])
        assert_raises(AssertionError, assert_equal, 1, [2])
        assert_raises(AssertionError, assert_equal, "a", ["a"])
        assert_raises(AssertionError, assert_equal, "1", [1])
        assert_raises(AssertionError, assert_equal, [1], 2)
        assert_raises(AssertionError, assert_equal, [1], 1)
        assert_raises(AssertionError, assert_equal, 1, [1])

    def test_tuple(self):
        tp_1 = ('a')
        tup_2 = ('a', 'b', 'c')

        assert_equal((), ())
        assert_equal(tp_1, tp_1)
        assert_equal(tup_2, tup_2)

        assert_raises(AssertionError, assert_equal, tp_1, tup_2)
        assert_raises(AssertionError, assert_equal, tp_1, 1)
        assert_raises(AssertionError, assert_equal, tp_1, np.nan)
        assert_raises(AssertionError, assert_equal, tp_1, ['a'])

    def test_masked_array(self):
        masked_1 = ma.masked_all((3, 3))
        masked_2 = ma.masked_all((2, 2))
        assert_equal(masked_1, masked_1)
        assert_equal(masked_2, masked_2)
        assert_raises(MaskError, assert_equal, masked_1, 2)
        assert_raises(MaskError, assert_equal, [1], masked_1)
        assert_raises(ValueError, assert_equal, masked_1, masked_2)

    def test_array(self):
        arr_1 = np.array([1, np.nan])
        array_2 = np.array([1, 2, 3])
        array_3 = np.array([1, 2, 3])

        assert_equal(arr_1, arr_1)
        assert_equal(array_2, array_3)
        assert_equal(arr_1, [1, np.nan])

        assert_raises(AssertionError, assert_equal, arr_1, array_2)
        assert_raises(AssertionError, assert_equal, array_2, arr_1)
        assert_raises(AssertionError, assert_equal, arr_1, np.nan)
        assert_raises(AssertionError, assert_equal, arr_1, 1)

    def test_string(self):
        assert_equal('', '')
        assert_equal('a', 'a')

        assert_raises(AssertionError, assert_equal, 'a', 'b')
        assert_raises(AssertionError, assert_equal, 'a', {})
        assert_raises(AssertionError, assert_equal, 'a', 1)
        assert_raises(AssertionError, assert_equal, 'a', [1])
        assert_raises(AssertionError, assert_equal, 'a', np.nan)


if __name__ == "__main__":
    run_module_suite()
