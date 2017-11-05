"""
Test the scalar constructors, which also do type-coercion
"""
from __future__ import division, absolute_import, print_function

import fractions
import numpy as np

from numpy.testing import (
    run_module_suite,
    assert_equal, assert_almost_equal, assert_raises, assert_warns,
    dec
)

float_types = [np.half, np.single, np.double, np.longdouble]

def test_float_as_integer_ratio():
    # derived from the cpython test "test_floatasratio"
    for ftype in float_types:
        for f, ratio in [
                (0.875, (7, 8)),
                (-0.875, (-7, 8)),
                (0.0, (0, 1)),
                (11.5, (23, 2)),
            ]:
            assert_equal(ftype(f).as_integer_ratio(), ratio)

        rstate = np.random.RandomState(0)
        fi = np.finfo(ftype)
        for i in range(1000):
            exp = rstate.randint(fi.minexp, fi.maxexp - 1)
            frac = rstate.rand()
            f = np.ldexp(frac, exp, dtype=ftype)

            n, d = f.as_integer_ratio()

            try:
                dn = np.longdouble(str(n))
                df = np.longdouble(str(d))
            except (OverflowError, RuntimeWarning):
                # the values may not fit in any float type
                continue

            assert_equal(
                dn / df, f,
                "{}/{} (dtype={})".format(n, d, ftype.__name__))

        R = fractions.Fraction
        assert_equal(R(0, 1),
                     R(*ftype(0.0).as_integer_ratio()))
        assert_equal(R(5, 2),
                     R(*ftype(2.5).as_integer_ratio()))
        assert_equal(R(1, 2),
                     R(*ftype(0.5).as_integer_ratio()))
        assert_equal(R(-2100, 1),
                     R(*ftype(-2100.0).as_integer_ratio()))

        assert_raises(OverflowError, ftype('inf').as_integer_ratio)
        assert_raises(OverflowError, ftype('-inf').as_integer_ratio)
        assert_raises(ValueError, ftype('nan').as_integer_ratio)


    assert_equal(R(1075, 512),
                 R(*np.half(2.1).as_integer_ratio()))
    assert_equal(R(-1075, 512),
                 R(*np.half(-2.1).as_integer_ratio()))
    assert_equal(R(4404019, 2097152),
                 R(*np.single(2.1).as_integer_ratio()))
    assert_equal(R(-4404019, 2097152),
                 R(*np.single(-2.1).as_integer_ratio()))
    assert_equal(R(4728779608739021, 2251799813685248),
                 R(*np.double(2.1).as_integer_ratio()))
    assert_equal(R(-4728779608739021, 2251799813685248),
                 R(*np.double(-2.1).as_integer_ratio()))
    # longdouble is platform depedent


if __name__ == "__main__":
    run_module_suite()
