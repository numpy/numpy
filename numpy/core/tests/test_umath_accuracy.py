import numpy as np
import platform
from os import path
import sys
import pytest
from ctypes import c_longlong, c_double, c_float, c_int, cast, pointer, POINTER
from numpy.testing import assert_array_max_ulp
from numpy.core._multiarray_umath import __cpu_features__

IS_AVX = __cpu_features__.get('AVX512F', False) or \
        (__cpu_features__.get('FMA3', False) and __cpu_features__.get('AVX2', False))
runtest = sys.platform.startswith('linux') and IS_AVX
platform_skip = pytest.mark.skipif(not runtest,
                                   reason="avoid testing inconsistent platform "
                                   "library implementations")


def from_hex(s, dtype):
    barr = bytearray.fromhex(s[2:])
    barr.reverse()
    return np.frombuffer(barr, dtype=dtype)[0]


def to_hex(x):
    barr = bytearray(x)
    barr.reverse()
    return "0x" + barr.hex()


def data_from_file(input_path):
    with open(input_path, 'r') as input_file:
        for line in input_file:
            if not line.strip() or line[0] == '#':
                continue
            if '#' in line:
                line = line[:line.index('#')]

            name, dtype_str, ulperr, arg_hex, exp_hex = line.strip().split(' ')
            dtype = getattr(np, dtype_str[3:])
            ulperr = int(ulperr)
            arg = from_hex(arg_hex, dtype)
            exp = from_hex(exp_hex, dtype)
            yield name, ulperr, arg, exp


files = ['umath-validation-set-exp',
         'umath-validation-set-log',
         'umath-validation-set-sin',
         'umath-validation-set-cos']


class TestAccuracy:
    @platform_skip
    def test_validate_transcendentals(self):
        with np.errstate(all='ignore'):
            for filename in files:
                data_dir = path.join(path.dirname(__file__), 'data')
                input_path = path.join(data_dir, filename)
                func_name = filename.split('-')[-1]
                ufunc = getattr(np, func_name)
                for name, ulperr, arg, exp in data_from_file(input_path):
                    act = ufunc(arg)
                    try:
                        assert_array_max_ulp(act, exp, maxulp=ulperr)
                    except:
                        raise AssertionError(
                            f"\n{name}:   {to_hex(arg)} ({arg})\n"
                            f"Received: {to_hex(act)} ({act})\n"
                            f"Expected: {to_hex(exp)} ({exp})\n"
                            "Received value insufficiently close to expected value."
                        )

    def test_ignore_nan_ulperror(self):
        # Ignore ULP differences between various NAN's
        nan1_f32 = from_hex('0xffffffff', dtype=np.float32)
        nan2_f32 = from_hex('0x7fddbfbf', dtype=np.float32)
        assert_array_max_ulp(nan1_f32, nan2_f32, 0)
