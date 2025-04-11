from . import util
import numpy as np
import pytest
from numpy.testing import assert_allclose

class TestIOSF(util.F2PyTest):
    sources = [
        util.getpath('tests', 'src', 'isofenv', 'isoftests.f90'),
    ]

    def test_f_real64(self):
        out = self.module.foddity.f_add(1.0, 2.0)
        exp_out = 3
        assert out == exp_out

    def test_f_addf(self):
        out = self.module.foddity.f_addf(1.0, 2.0)
        exp_out = 3
        assert out == exp_out

    def test_wat(self):
        out = self.module.foddity.wat(1, 2)
        exp_out = 8
        assert out == exp_out

    def test_f_add_int64(self):
        out = self.module.foddity.f_add_int64(2**32, 2**32)
        exp_out = 2 ** 33
        assert out == exp_out

    def test_f_add_int16_arr(self):
        args = np.arange(6, dtype=np.int16)
        out = self.module.foddity.f_add_int16_arr(args[:3], args[3:])
        exp_out = args[:3] + args[3:]
        assert_allclose(out, exp_out)
        assert exp_out.dtype == out.dtype

    def test_f_add_int8_arr(self):
        args = np.arange(6, dtype = np.int8)
        out = self.module.foddity.f_add_int8_arr(args[:3], args[3:])
        exp_out = args[:3] + args[3:]
        assert_allclose(out, exp_out)
        assert out.dtype == exp_out.dtype

    def test_f_add_arr(self):
        args = np.arange(6, dtype=np.int64)
        out = self.module.foddity.add_arr(args[:3], args[3:])
        exp_out = args[:3] + args[3:]
        assert_allclose(out, exp_out)
        assert out.dtype == exp_out.dtype
