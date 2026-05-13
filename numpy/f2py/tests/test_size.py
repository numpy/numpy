import pytest

import numpy as np

from . import util


# Note: do not mark as slow! This is one of a small set of f2py compile tests retained
# in the default suite invocation for compile-path coverage.
class TestSizeSumExample(util.F2PyTest):
    sources = [util.getpath("tests", "src", "size", "foo.f90")]

    def test_all(self):
        r = self.module.foo([[]])
        assert r == [0]

        r = self.module.foo([[1, 2]])
        assert r == [3]

        r = self.module.foo([[1, 2], [3, 4]])
        assert np.allclose(r, [3, 7])

        r = self.module.foo([[1, 2], [3, 4], [5, 6]])
        assert np.allclose(r, [3, 7, 11])

    def test_transpose(self):
        r = self.module.trans([[]])
        assert np.allclose(r.T, np.array([[]]))

        r = self.module.trans([[1, 2]])
        assert np.allclose(r, [[1.], [2.]])

        r = self.module.trans([[1, 2, 3], [4, 5, 6]])
        assert np.allclose(r, [[1, 4], [2, 5], [3, 6]])

    def test_flatten(self):
        r = self.module.flatten([[]])
        assert np.allclose(r, [])

        r = self.module.flatten([[1, 2]])
        assert np.allclose(r, [1, 2])

        r = self.module.flatten([[1, 2, 3], [4, 5, 6]])
        assert np.allclose(r, [1, 2, 3, 4, 5, 6])
