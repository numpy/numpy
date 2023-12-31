import pytest
import numpy as np
from . import util

@pytest.fixture(scope="module")
def common_block_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestCommonBlock",
        sources = [util.getpath("tests", "src", "common", "block.f")]
    )
    return spec

@pytest.mark.parametrize("_mod", ["common_block_spec"], indirect=True)
def test_common_block(_mod):
    _mod.initcb()
    assert _mod.block.long_bn == np.array(1.0, dtype=np.float64)
    assert _mod.block.string_bn == np.array("2", dtype="|S1")
    assert _mod.block.ok == np.array(3, dtype=np.int32)


@pytest.fixture(scope="module")
def common_use_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestCommonWithUse",
        sources = [util.getpath("tests", "src", "common", "gh19161.f90")]
    )
    return spec

@pytest.mark.parametrize("_mod", ["common_use_spec"], indirect=True)
def test_common_gh19161(_mod):
    assert _mod.data.x == 0
