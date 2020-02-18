import numpy
import pytest

from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_almost_equal,
    assert_array_almost_equal, assert_raises, assert_allclose
    )
import unittest.mock

from numpy.f2py.f2py2e import scaninputline
from numpy.f2py.cb_rules import buildcallback


class TestScanInput:

#Test check scan input line exit with invalid input
    def test_input1(self):
        mock = unittest.mock.MagicMock()
        mock.return_value = "h-overwrite"
        with pytest.raises(SystemExit) as pytest_wrapped_e:
            numpy.f2py.f2py2e.scaninputline(mock())
        assert_equal(pytest_wrapped_e.type, SystemExit)

