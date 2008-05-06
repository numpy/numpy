import os
import tempfile

from numpy.testing import *

set_local_path()
import numpy as np
restore_path()

nrepeat = 10
deflevel = 5

# Here are the C functions which use PyDataMem_RENEW:
# - PyArray_Resize (2 calls)
# - PyArray_FromIter (2 calls)
# - array_from_text (2 calls)
# - PyArray_FromFile (1 call)

class TestResize(NumpyTestCase):
    def _test(self, sz, shift):
        y = np.random.randn(sz)

        print '%8.2f' % self.measure('np.resize(y, sz + shift)', nrepeat)

    def test_small_up(self, level = deflevel):
        sz = 1024 * 1024 * 10
        self._test(sz, 10)

    def test_small_down(self, level = deflevel):
        sz = 1024 * 1024 * 10
        self._test(sz, -10)

    def test_half_up(self, level = deflevel):
        sz = 1024 * 1024 * 10
        self._test(sz, sz/2)

    def test_half_down(self, level = deflevel):
        sz = 1024 * 1024 * 10
        self._test(sz, -sz/2)

class TestFromIter(NumpyTestCase):
    def _test(self, sz, rep = nrepeat):
        fid, name = tempfile.mkstemp('ypyp')
        f = os.fdopen(fid, 'w')
        a = '\n'.join(['1', '2', '3', '4', '5'])
        for i in xrange(sz):
            f.writelines(a)

        print '%8.2f' % self.measure('f.seek(0); np.fromiter(f, np.float64)', rep)

    def test1(self, level = deflevel):
        self._test(1000, 100)

    def test2(self, level = deflevel):
        self._test(100000)

class TestLoadText(NumpyTestCase):
    def _test(self, sz):
        fid, name = tempfile.mkstemp('ypyp')
        f = os.fdopen(fid, 'w')
        a = np.random.randn(sz)
        np.savetxt(f, a)

        print '%8.2f' % self.measure('f.seek(0); np.loadtxt(f)', nrepeat)

    def test1(self, level = deflevel):
        self._test(1000)

    def test2(self, level = deflevel):
        self._test(10000)

class TestFromFile(NumpyTestCase):
    def _test(self, sz, nrep = nrepeat):
        fid, name = tempfile.mkstemp('ypyp')
        f = os.fdopen(fid, 'w')
        a = np.random.randn(sz)
        a.tofile(f)

        print '%8.2f' % self.measure('f.seek(0); np.fromfile(f)', nrep)

    def test1(self, level = deflevel):
        self._test(100000, 100)

    def test2(self, level = deflevel):
        self._test(1000000, 100)

if __name__ == "__main__":
    NumpyTest().test(verbosity = 10, level = 5)
