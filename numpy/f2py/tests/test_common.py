import pytest

import numpy as np
from numpy.f2py import _testutils


@pytest.mark.slow
class TestCommonBlock(_testutils.F2PyTest):
    sources = [_testutils.getpath("tests", "src", "common", "block.f")]

    def test_common_block(self):
        self.module.initcb()
        assert self.module.block.long_bn == np.array(1.0, dtype=np.float64)
        assert self.module.block.string_bn == np.array("2", dtype="|S1")
        assert self.module.block.ok == np.array(3, dtype=np.int32)


class TestCommonWithUse(_testutils.F2PyTest):
    sources = [_testutils.getpath("tests", "src", "common", "gh19161.f90")]

    def test_common_gh19161(self):
        assert self.module.data.x == 0
