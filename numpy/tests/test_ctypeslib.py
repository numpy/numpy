import numpy as np
from numpy.ctypeslib import ndpointer, load_library
from numpy.testing import *

class TestLoadLibrary(NumpyTestCase):
    def check_basic(self):
        cdll = load_library('multiarray',
                            np.core.multiarray.__file__)

class TestNdpointer(NumpyTestCase):
    def check_dtype(self):
        dt = np.intc
        p = ndpointer(dtype=dt)
        self.assert_(p.from_param(np.array([1], dt)))
        dt = '<i4'
        p = ndpointer(dtype=dt)
        self.assert_(p.from_param(np.array([1], dt)))
        dt = np.dtype('>i4')
        p = ndpointer(dtype=dt)
        p.from_param(np.array([1], dt))
        self.assertRaises(TypeError, p.from_param,
                          np.array([1], dt.newbyteorder('swap')))
        dtnames = ['x', 'y']
        dtformats = [np.intc, np.float64]
        dtdescr = {'names' : dtnames, 'formats' : dtformats}
        dt = np.dtype(dtdescr)
        p = ndpointer(dtype=dt)
        self.assert_(p.from_param(np.zeros((10,), dt)))
        samedt = np.dtype(dtdescr)
        p = ndpointer(dtype=samedt)
        self.assert_(p.from_param(np.zeros((10,), dt)))
        dt2 = np.dtype(dtdescr, align=True)
        if dt.itemsize != dt2.itemsize:
            self.assertRaises(TypeError, p.from_param, np.zeros((10,), dt2))
        else:
            self.assert_(p.from_param(np.zeros((10,), dt2)))

    def check_ndim(self):
        p = ndpointer(ndim=0)
        self.assert_(p.from_param(np.array(1)))
        self.assertRaises(TypeError, p.from_param, np.array([1]))
        p = ndpointer(ndim=1)
        self.assertRaises(TypeError, p.from_param, np.array(1))
        self.assert_(p.from_param(np.array([1])))
        p = ndpointer(ndim=2)
        self.assert_(p.from_param(np.array([[1]])))

    def check_shape(self):
        p = ndpointer(shape=(1,2))
        self.assert_(p.from_param(np.array([[1,2]])))
        self.assertRaises(TypeError, p.from_param, np.array([[1],[2]]))
        p = ndpointer(shape=())
        self.assert_(p.from_param(np.array(1)))

    def check_flags(self):
        x = np.array([[1,2,3]], order='F')
        p = ndpointer(flags='FORTRAN')
        self.assert_(p.from_param(x))
        p = ndpointer(flags='CONTIGUOUS')
        self.assertRaises(TypeError, p.from_param, x)
        p = ndpointer(flags=x.flags.num)
        self.assert_(p.from_param(x))
        self.assertRaises(TypeError, p.from_param, np.array([[1,2,3]]))

if __name__ == "__main__":
    NumpyTest().run()
