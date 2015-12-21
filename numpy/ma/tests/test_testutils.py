from numpy.testing import assert_raises
from numpy.testing import run_module_suite
from numpy.ma.core import MaskError
import numpy as np
import numpy.ma as ma
from numpy.ma.testutils import assert_equal


class TestAssertEqual:
    def test_dictionary(self):
        dict_a = {'foo': 'bar', 'foo2': 'bar2'}
        dict_b = {'foo': 'bar', 'foo2': 'bar2', 'foo3': 'bar3'}
        dict_c = {'foo': 'bar', 'foo2': 'bar2', 'foo3': 'baz1'}
        tup_1 = ('foo')
        array_1 = np.array([1, 2])

        assert_equal({}, {})
        assert_equal(dict_a, dict_a)
        assert_equal(dict_b, dict_b)

        assert_raises(AssertionError, assert_equal, dict_a, dict_b)
        assert_raises(AssertionError, assert_equal, dict_b, dict_c)
        assert_raises(AssertionError, assert_equal, dict_a, dict_c)
        assert_raises(AssertionError, assert_equal, dict_a, 1)
        assert_raises(AssertionError, assert_equal, dict_a, {})
        assert_raises(AssertionError, assert_equal, dict_a, [])
        assert_raises(AssertionError, assert_equal, dict_a, tup_1)
        assert_raises(AssertionError, assert_equal, dict_a, array_1)

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
        assert_raises(AssertionError, assert_equal, [1], 2)
        assert_raises(AssertionError, assert_equal, [1], 1)
        assert_raises(AssertionError, assert_equal, 1, [1])

    def test_tuple(self):
        tup_1 = ('foo')
        tup_2 = ('foo', 'bar', 'baz')

        assert_equal((), ())
        assert_equal(tup_1, tup_1)
        assert_equal(tup_2, tup_2)

        assert_raises(AssertionError, assert_equal, tup_1, tup_2)
        assert_raises(AssertionError, assert_equal, tup_1, 1)
        assert_raises(AssertionError, assert_equal, tup_1, np.nan)
        assert_raises(AssertionError, assert_equal, tup_1, ['foo'])

    def test_masked_array(self):
        masked_1 = ma.masked_all((3, 3))
        masked_2 = ma.masked_all((2, 2))
        assert_equal(masked_1, masked_1)
        assert_equal(masked_2, masked_2)
        assert_raises(MaskError, assert_equal, masked_1, 2)
        assert_raises(MaskError, assert_equal, [1], masked_1)
        assert_raises(ValueError, assert_equal, masked_1, masked_2)

    def test_array(self):
        array_1 = np.array([1, np.nan])
        array_2 = np.array([1, 2, 3])
        array_3 = np.array([1, 2, 3])

        assert_equal(array_1, array_1)
        assert_equal(array_2, array_3)
        assert_equal(array_1, [1, np.nan])

        assert_raises(AssertionError, assert_equal, array_1, array_2)
        assert_raises(AssertionError, assert_equal, array_2, array_1)
        assert_raises(AssertionError, assert_equal, array_1, np.nan)
        assert_raises(AssertionError, assert_equal, array_1, 1)

    def test_string(self):
        assert_equal('', '')
        assert_equal('a', 'a')

        assert_raises(AssertionError, assert_equal, 'a', 'b')
        assert_raises(AssertionError, assert_equal, 'a', 1)
        assert_raises(AssertionError, assert_equal, 'a', [1])
        assert_raises(AssertionError, assert_equal, 'a', np.nan)


if __name__ == "__main__":
    run_module_suite()
