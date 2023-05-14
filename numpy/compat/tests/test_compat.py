from os.path import join
from io import BufferedReader, BytesIO

from numpy.compat.py3k import isfileobj, _isfileobj
from numpy.testing import assert_
from numpy.testing import tempdir


def test_isfileobj():
    with tempdir(prefix="numpy_test_compat_") as folder:
        filename = join(folder, 'a.bin')

        with open(filename, 'wb') as f:
            assert_(isfileobj(f))

        with open(filename, 'ab') as f:
            assert_(isfileobj(f))

        with open(filename, 'rb') as f:
            assert_(isfileobj(f))

def test__isfileobj():
    with tempdir(prefix="numpy_test_compat_") as folder:
        filename = join(folder, 'a.bin')

        with open(filename, 'wb') as f:
            assert_(_isfileobj(f))

        with open(filename, 'ab') as f:
            assert_(_isfileobj(f))

        with open(filename, 'rb') as f:
            assert_(_isfileobj(f))
        
        assert_(_isfileobj(BufferedReader(BytesIO())) is False)
