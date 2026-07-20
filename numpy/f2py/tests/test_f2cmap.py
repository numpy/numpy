import pytest

import numpy as np
from numpy.f2py import capi_maps

from . import util


@pytest.mark.slow
class TestF2Cmap(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "f2cmap", "isoFortranEnvMap.f90"),
        util.getpath("tests", "src", "f2cmap", ".f2py_f2cmap")
    ]

    # gh-15095
    def test_gh15095(self):
        inp = np.ones(3)
        out = self.module.func1(inp)
        exp_out = 3
        assert out == exp_out


class TestF2CmapParsing:
    # An f2cmap file must never be able to execute arbitrary code.
    def test_dict_literal_form(self):
        d = capi_maps._f2cmap_from_str("{'real': {'low': 'float'}}")
        assert d == {'real': {'low': 'float'}}

    def test_dict_constructor_form(self):
        d = capi_maps._f2cmap_from_str("dict(real=dict(rk='double'))")
        assert d == {'real': {'rk': 'double'}}

    @pytest.mark.parametrize("payload", [
        "__import__('os').system('echo pwned')",
        "(open('x', 'wb').write(b'pwned'), {'real': {'4': 'float'}})[1]",
        "dict(real=__import__('os').getcwd())",
    ])
    def test_rejects_code_execution(self, payload):
        with pytest.raises(ValueError):
            capi_maps._f2cmap_from_str(payload)
