from os.path import join

from numpy.compat import isfileobj, isfilelikeobj
from numpy.testing import assert_
from numpy.testing import tempdir

import io
import gzip

def test_isfileobj():
    with tempdir(prefix="numpy_test_compat_") as folder:
        filename = join(folder, 'a.bin')

        with open(filename, 'wb') as f:
            assert_(isfileobj(f))

        with open(filename, 'ab') as f:
            assert_(isfileobj(f))

        with open(filename, 'rb') as f:
            assert_(isfileobj(f))

def test_isfilelikeobj():
    with tempdir(prefix="numpy_test_compat_") as folder:
        filename = join(folder, 'a.bin')

        with open(filename, 'wb') as f:
            assert_(isfilelikeobj(f))

        with gzip.open(filename, 'rb') as f:
            assert_(isfilelikeobj(f))

        assert_(isfilelikeobj(io.BytesIO()))

        assert_(not isfilelikeobj(io.StringIO()))
