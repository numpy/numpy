import os
import pytest
import numpy as np

from . import util


@pytest.fixture(scope="module")
def size_sum_spec():
    spec = util.F2PyModuleSpec(
        test_class_name="TestSizeSumExample",
        sources=[
            util.getpath("tests", "src", "size", "foo.f90"),
        ],
    )
    return spec


@pytest.mark.parametrize("_mod", ["size_sum_spec"], indirect=True)
def test_all(_mod):
    r = _mod.foo([[]])
    assert r == [0]

    r = _mod.foo([[1, 2]])
    assert r == [3]

    r = _mod.foo([[1, 2], [3, 4]])
    assert np.allclose(r, [3, 7])

    r = _mod.foo([[1, 2], [3, 4], [5, 6]])
    assert np.allclose(r, [3, 7, 11])


@pytest.mark.parametrize("_mod", ["size_sum_spec"], indirect=True)
def test_transpose(_mod):
    r = _mod.trans([[]])
    assert np.allclose(r.T, np.array([[]]))

    r = _mod.trans([[1, 2]])
    assert np.allclose(r, [[1.0], [2.0]])

    r = _mod.trans([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(r, [[1, 4], [2, 5], [3, 6]])


@pytest.mark.parametrize("_mod", ["size_sum_spec"], indirect=True)
def test_flatten(_mod):
    r = _mod.flatten([[]])
    assert np.allclose(r, [])

    r = _mod.flatten([[1, 2]])
    assert np.allclose(r, [1, 2])

    r = _mod.flatten([[1, 2, 3], [4, 5, 6]])
    assert np.allclose(r, [1, 2, 3, 4, 5, 6])
