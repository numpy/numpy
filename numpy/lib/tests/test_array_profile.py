"""
Tests for array_profile function.
"""

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal
import pytest


class TestArrayProfile:
    def setup_method(self):
        self.a = np.array([[1, 2, 3], [4, 5, 6]])
        self.b = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        self.c = np.array([1, 2, np.nan, 4, 5])
        self.d = np.array(['a', 'b', 'c'])
        self.e = np.array([])

    def test_basic_properties(self):
        """Test that basic array properties are correctly reported."""
        result = np.array_profile(self.a, return_dict=True)
        assert_equal(result['shape'], (2, 3))
        assert_equal(result['dimensions'], 2)
        assert_equal(result['size'], 6)
        assert_equal(result['dtype'], np.int64)
        assert_equal(result['has_nan'], False)

    def test_statistical_properties(self):
        """Test that statistical properties are correctly calculated."""
        result = np.array_profile(self.a, return_dict=True)
        assert_equal(result['max_value'], 6)
        assert_equal(result['min_value'], 1)
        assert_almost_equal(result['mean'], 3.5)
        assert_equal(result['total_sum'], 21)
        assert_equal(result['range'], 5)
        assert_almost_equal(result['std_dev'], 1.7078, decimal=4)
        assert_almost_equal(result['variance'], 2.9167, decimal=4)
        assert_equal(result['unique_values'], 6)

    def test_float_array(self):
        """Test with float array."""
        result = np.array_profile(self.b, return_dict=True)
        assert_equal(result['dtype'], np.float64)
        assert_almost_equal(result['mean'], 3.5)
        assert_almost_equal(result['total_sum'], 17.5)

    def test_nan_handling(self):
        """Test that NaN values are handled correctly."""
        result = np.array_profile(self.c, return_dict=True)
        assert_equal(result['has_nan'], True)
        assert_equal(result['max_value'], 5.0)
        assert_equal(result['min_value'], 1.0)
        assert_almost_equal(result['mean'], 3.0)
        assert_equal(result['unique_values'], 4)  # NaN is not counted as unique

    def test_non_numeric_array(self):
        """Test with non-numeric array."""
        result = np.array_profile(self.d, return_dict=True)
        assert_equal(result['dtype'], np.dtype('<U1'))
        assert_equal(result['max_value'], 'N/A')
        assert_equal(result['mean'], 'N/A')

    def test_empty_array(self):
        """Test with empty array."""
        result = np.array_profile(self.e, return_dict=True)
        assert_equal(result['size'], 0)
        assert_equal(result['max_value'], 'N/A')
        assert_equal(result['mean'], 'N/A')

    def test_print_output(self, capsys):
        """Test that the printed output contains expected information."""
        np.array_profile(self.a)
        
        captured = capsys.readouterr()
        output = captured.out
        
        assert "Array Profile" in output
        assert "Shape        : (2, 3)" in output
        assert "Dimensions   : 2" in output
        assert "Size         : 6" in output
        assert "Max Value    : 6" in output
        assert "Min Value    : 1" in output
        assert "Mean         : 3.5000" in output
