from tempfile import NamedTemporaryFile, mktemp
import os
import warnings

from numpy import memmap
from numpy import arange, allclose
from numpy.testing import TestCase, assert_, assert_array_equal

class TestMemmap(TestCase):
    def setUp(self):
        self.tmpfp = NamedTemporaryFile(prefix='mmap')
        self.shape = (3,4)
        self.dtype = 'float32'
        self.data = arange(12, dtype=self.dtype)
        self.data.resize(self.shape)

    def tearDown(self):
        self.tmpfp.close()

    def test_roundtrip(self):
        # Write data to file
        fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+',
                    shape=self.shape)
        fp[:] = self.data[:]
        del fp # Test __del__ machinery, which handles cleanup

        # Read data back from file
        newfp = memmap(self.tmpfp, dtype=self.dtype, mode='r',
                       shape=self.shape)
        assert_(allclose(self.data, newfp))
        assert_array_equal(self.data, newfp)

    def test_open_with_filename(self):
        tmpname = mktemp('','mmap')
        fp = memmap(tmpname, dtype=self.dtype, mode='w+',
                       shape=self.shape)
        fp[:] = self.data[:]
        del fp
        os.unlink(tmpname)

    def test_attributes(self):
        offset = 1
        mode = "w+"
        fp = memmap(self.tmpfp, dtype=self.dtype, mode=mode,
                    shape=self.shape, offset=offset)
        self.assertEquals(offset, fp.offset)
        self.assertEquals(mode, fp.mode)
        del fp

    def test_filename(self):
        tmpname = mktemp('','mmap')
        fp = memmap(tmpname, dtype=self.dtype, mode='w+',
                       shape=self.shape)
        abspath = os.path.abspath(tmpname)
        fp[:] = self.data[:]
        self.assertEquals(abspath, fp.filename)
        b = fp[:1]
        self.assertEquals(abspath, b.filename)
        del b
        del fp
        os.unlink(tmpname)

    def test_filename_fileobj(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, mode="w+",
                    shape=self.shape)
        self.assertEquals(fp.filename, self.tmpfp.name)

    def test_flush(self):
        fp = memmap(self.tmpfp, dtype=self.dtype, mode='w+',
                    shape=self.shape)
        fp[:] = self.data[:]
        fp.flush()

    def test_del(self):
        # Make sure a view does not delete the underlying mmap
        fp_base = memmap(self.tmpfp, dtype=self.dtype, mode='w+',
                    shape=self.shape)
        fp_view = fp_base[:]
        class ViewCloseError(Exception):
            pass
        _close = memmap._close
        def replace_close(self):
            raise ViewCloseError('View should not call _close on memmap')
        try:
            memmap._close = replace_close
            del fp_view
        finally:
            memmap._close = _close

if __name__ == "__main__":
    run_module_suite()
