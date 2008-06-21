from tempfile import NamedTemporaryFile, mktemp
import os
import warnings

from numpy.core import memmap
from numpy import arange, allclose
from numpy.testing import *

class TestMemmap(TestCase):
    def setUp(self):
        self.tmpfp = NamedTemporaryFile(prefix='mmap')
        self.shape = (3,4)
        self.dtype = 'float32'
        self.data = arange(12, dtype=self.dtype)
        self.data.resize(self.shape)

    def test_roundtrip(self):
        # Write data to file
        fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+',
                    shape=self.shape)
        fp[:] = self.data[:]
        del fp # Test __del__ machinery, which handles cleanup

        # Read data back from file
        newfp = memmap(self.tmpfp, dtype=self.dtype, mode='r',
                       shape=self.shape)
        assert allclose(self.data, newfp)
        assert_array_equal(self.data, newfp)

    def test_open_with_filename(self):
        tmpname = mktemp('','mmap')
        fp = memmap(tmpname, dtype=self.dtype, mode='w+',
                       shape=self.shape)
        fp[:] = self.data[:]
        del fp
        os.unlink(tmpname)

    def test_flush(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+',
                    shape=self.shape)
        fp[:] = self.data[:]
        fp.flush()

        warnings.simplefilter('ignore', DeprecationWarning)
        fp.sync()
        warnings.simplefilter('default', DeprecationWarning)


if __name__ == "__main__":
    run_module_suite()

