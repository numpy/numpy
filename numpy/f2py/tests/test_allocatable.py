import pytest

import numpy as np

from . import util


@pytest.mark.slow
class TestAllocatable(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "allocatable", "alloc.f90")
    ]

    def test_one_d(self):
        v = self.module.mod.probe_one_d(1)
        assert v == -1

        self.module.mod.one_d = np.linspace(1, 10, 5, dtype=np.float32)

        v = self.module.mod.probe_one_d(1)
        assert np.allclose(v, 1.0)
        v = self.module.mod.probe_one_d(5)
        assert np.allclose(v, 10.0)

        assert np.allclose(self.module.mod.one_d[0], 1.0)
        assert np.allclose(self.module.mod.one_d[4], 10.0)

        self.module.mod.one_d = None
        assert self.module.mod.probe_one_d(1) == -1

        #with pytest.raises(ValueError):
        #    self.module.mod.one_d = [[1, 2, 3], [9, 9, 11]]
        with pytest.raises(TypeError):
            self.module.mod.one_d = [1.j]
        with pytest.raises(TypeError):
            self.module.mod.one_d = ...

    def test_two_d(self):
        v = self.module.mod.probe_two_d(1, 1)
        assert v == -1

        self.module.mod.two_d = [[1, 2, 3],
                                 [4, 5, 6]]
        v = self.module.mod.probe_two_d(1, 1)
        assert np.allclose(v, 1)
        v = self.module.mod.probe_two_d(2, 2)
        assert np.allclose(v, 5)
        v = self.module.mod.probe_two_d(2, 3)
        assert np.allclose(v, 6)

    def test_three_d(self):
        v = self.module.mod.probe_three_d(1, 1, 1)
        assert v == -1

        self.module.mod.three_d = (
            np.linspace(0, 27 - 27j, 27, dtype=np.complex64)
        ).reshape(3, 3, 3)

        v = self.module.mod.probe_three_d(1, 1, 1)
        assert np.allclose(v, 0)
        v = self.module.mod.probe_three_d(2, 2, 2)
        assert np.allclose(v, 13.5 - 13.5j)
        v = self.module.mod.probe_three_d(3, 3, 3)
        assert np.allclose(v, 27 - 27.j)

        self.module.mod.three_d[1, 1, 1] = 42j
        v = self.module.mod.probe_three_d(2, 2, 2)
        assert np.allclose(v, 42j)
