from numpy.testing import *
set_package_path()
import numpy as N
from numpy.ctypeslib import ndpointer
restore_path()

class test_ndpointer(NumpyTestCase):
    def check_dtype(self):
        dt = N.intc
        p = ndpointer(dtype=dt)
        self.assert_(p.from_param(N.array([1], dt)))
        dt = '<i4'
        p = ndpointer(dtype=dt)
        self.assert_(p.from_param(N.array([1], dt)))
        dt = N.dtype('>i4')
        p = ndpointer(dtype=dt)
        p.from_param(N.array([1], dt))
        self.assertRaises(TypeError, p.from_param,
                          N.array([1], dt.newbyteorder('swap')))
        dtnames = ['x', 'y']
        dtformats = [N.intc, N.float64]
        dtdescr = {'names' : dtnames, 'formats' : dtformats}
        dt = N.dtype(dtdescr)
        p = ndpointer(dtype=dt)
        self.assert_(p.from_param(N.zeros((10,), dt)))
        samedt = N.dtype(dtdescr)
        p = ndpointer(dtype=samedt)
        self.assert_(p.from_param(N.zeros((10,), dt)))
        dt2 = N.dtype(dtdescr, align=True)
        if dt.itemsize != dt2.itemsize:
            self.assertRaises(TypeError, p.from_param, N.zeros((10,), dt2))
        else:
            self.assert_(p.from_param(N.zeros((10,), dt2)))

    def check_ndim(self):
        p = ndpointer(ndim=0)
        self.assert_(p.from_param(N.array(1)))
        self.assertRaises(TypeError, p.from_param, N.array([1]))
        p = ndpointer(ndim=1)
        self.assertRaises(TypeError, p.from_param, N.array(1))
        self.assert_(p.from_param(N.array([1])))
        p = ndpointer(ndim=2)
        self.assert_(p.from_param(N.array([[1]])))

    def check_shape(self):
        p = ndpointer(shape=(1,2))
        self.assert_(p.from_param(N.array([[1,2]])))
        self.assertRaises(TypeError, p.from_param, N.array([[1],[2]]))
        p = ndpointer(shape=())
        self.assert_(p.from_param(N.array(1)))

    def check_flags(self):
        x = N.array([[1,2,3]], order='F')
        p = ndpointer(flags='FORTRAN')
        self.assert_(p.from_param(x))
        p = ndpointer(flags='CONTIGUOUS')
        self.assertRaises(TypeError, p.from_param, x)
        p = ndpointer(flags=x.flags.num)
        self.assert_(p.from_param(x))
        self.assertRaises(TypeError, p.from_param, N.array([[1,2,3]]))

if __name__ == "__main__":
    NumpyTest().run()
