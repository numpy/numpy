# -*- coding: utf-8 -*-
""" Test printing of scalar types.

"""
from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.testing import assert_equal, run_module_suite


class TestRealScalars(object):
    def test_str(self):
        svals = [0.0, -0.0, 1, -1, np.inf, -np.inf, np.nan]
        styps = [np.float16, np.float32, np.float64, np.longdouble]
        wanted = [
             ['0.0',  '0.0',  '0.0',  '0.0' ],
             ['-0.0', '-0.0', '-0.0', '-0.0'],
             ['1.0',  '1.0',  '1.0',  '1.0' ],
             ['-1.0', '-1.0', '-1.0', '-1.0'],
             ['inf',  'inf',  'inf',  'inf' ],
             ['-inf', '-inf', '-inf', '-inf'],
             ['nan',  'nan',  'nan',  'nan']]

        for wants, val in zip(wanted, svals):
            for want, styp in zip(wants, styps):
                msg = 'for str({}({}))'.format(np.dtype(styp).name, repr(val))
                assert_equal(str(styp(val)), want, err_msg=msg)


if __name__ == "__main__":
    run_module_suite()
