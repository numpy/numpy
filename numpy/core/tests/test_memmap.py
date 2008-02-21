
from tempfile import NamedTemporaryFile

from numpy.core import memmap
from numpy import arange, allclose
from numpy.testing import *

class TestMemmap(NumpyTestCase):
    def setUp(self):
        self.tmpfp = NamedTemporaryFile(prefix='mmap')
        self.shape = (3,4)
        self.dtype = 'float32'
        self.data = arange(12, dtype=self.dtype)
        self.data.resize(self.shape)

    def test_RoundTrip(self):
        fp = memmap(self.tmpfp.name, dtype=self.dtype, mode='w+', 
                    shape=self.shape)
        fp[:] = self.data[:]
        del fp
        newfp = memmap(self.tmpfp.name, dtype=self.dtype, mode='r', 
                       shape=self.shape)
        assert allclose(self.data, newfp)
        assert_array_equal(self.data, newfp)

if __name__ == '__main__':
    NumpyTest().run()
