import numpy as np
from numpy.testing import *

class TestArrayRepr(object):
    def test_nan_inf(self):
        x = np.array([np.nan, np.inf])
        assert_equal(repr(x), 'array([ nan,  inf])')

class TestComplexArray(TestCase):
    def test_str(self):
        rvals = [0, 1, -1, np.inf, -np.inf, np.nan]
        cvals = [complex(rp, ip) for rp in rvals for ip in rvals]
        dtypes = [np.complex64, np.cdouble, np.clongdouble]
        actual = [str(np.array([c], dt)) for c in cvals for dt in dtypes]
        wanted = [
            '[ 0.+0.j]',    '[ 0.+0.j]',    '[ 0.0+0.0j]', 
            '[ 0.+1.j]',    '[ 0.+1.j]',    '[ 0.0+1.0j]', 
            '[ 0.-1.j]',    '[ 0.-1.j]',    '[ 0.0-1.0j]', 
            '[ 0.+infj]',   '[ 0.+infj]',   '[ 0.0+infj]', 
            '[ 0.-infj]',   '[ 0.-infj]',   '[ 0.0-infj]', 
            '[ 0.+nanj]',   '[ 0.+nanj]',   '[ 0.0+nanj]', 
            '[ 1.+0.j]',    '[ 1.+0.j]',    '[ 1.0+0.0j]', 
            '[ 1.+1.j]',    '[ 1.+1.j]',    '[ 1.0+1.0j]', 
            '[ 1.-1.j]',    '[ 1.-1.j]',    '[ 1.0-1.0j]', 
            '[ 1.+infj]',   '[ 1.+infj]',   '[ 1.0+infj]', 
            '[ 1.-infj]',   '[ 1.-infj]',   '[ 1.0-infj]', 
            '[ 1.+nanj]',   '[ 1.+nanj]',   '[ 1.0+nanj]', 
            '[-1.+0.j]',    '[-1.+0.j]',    '[-1.0+0.0j]', 
            '[-1.+1.j]',    '[-1.+1.j]',    '[-1.0+1.0j]', 
            '[-1.-1.j]',    '[-1.-1.j]',    '[-1.0-1.0j]', 
            '[-1.+infj]',   '[-1.+infj]',   '[-1.0+infj]', 
            '[-1.-infj]',   '[-1.-infj]',   '[-1.0-infj]', 
            '[-1.+nanj]',   '[-1.+nanj]',   '[-1.0+nanj]', 
            '[ inf+0.j]',   '[ inf+0.j]',   '[ inf+0.0j]', 
            '[ inf+1.j]',   '[ inf+1.j]',   '[ inf+1.0j]', 
            '[ inf-1.j]',   '[ inf-1.j]',   '[ inf-1.0j]', 
            '[ inf+infj]',  '[ inf+infj]',  '[ inf+infj]', 
            '[ inf-infj]',  '[ inf-infj]',  '[ inf-infj]', 
            '[ inf+nanj]',  '[ inf+nanj]',  '[ inf+nanj]', 
            '[-inf+0.j]',   '[-inf+0.j]',   '[-inf+0.0j]', 
            '[-inf+1.j]',   '[-inf+1.j]',   '[-inf+1.0j]', 
            '[-inf-1.j]',   '[-inf-1.j]',   '[-inf-1.0j]', 
            '[-inf+infj]',  '[-inf+infj]',  '[-inf+infj]', 
            '[-inf-infj]',  '[-inf-infj]',  '[-inf-infj]', 
            '[-inf+nanj]',  '[-inf+nanj]',  '[-inf+nanj]', 
            '[ nan+0.j]',   '[ nan+0.j]',   '[ nan+0.0j]', 
            '[ nan+1.j]',   '[ nan+1.j]',   '[ nan+1.0j]', 
            '[ nan-1.j]',   '[ nan-1.j]',   '[ nan-1.0j]', 
            '[ nan+infj]',  '[ nan+infj]',  '[ nan+infj]', 
            '[ nan-infj]',  '[ nan-infj]',  '[ nan-infj]', 
            '[ nan+nanj]',  '[ nan+nanj]',  '[ nan+nanj]']

        for res, val in zip(actual, wanted):
            assert_(res == val)
 
if __name__ == "__main__":
    run_module_suite()
