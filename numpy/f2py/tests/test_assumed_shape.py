import os
import tempfile

import pytest

from . import util


# Note: do not mark as slow! This is one of a small set of f2py compile tests retained
# in the default suite invocation for compile-path coverage.
class TestAssumedShapeSumExample(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "assumed_shape", "foo_free.f90"),
        util.getpath("tests", "src", "assumed_shape", "foo_use.f90"),
        util.getpath("tests", "src", "assumed_shape", "precision.f90"),
        util.getpath("tests", "src", "assumed_shape", "foo_mod.f90"),
        util.getpath("tests", "src", "assumed_shape", ".f2py_f2cmap"),
    ]

    def test_all(self):
        r = self.module.fsum([1, 2])
        assert r == 3
        r = self.module.sum([1, 2])
        assert r == 3
        r = self.module.sum_with_use([1, 2])
        assert r == 3

        r = self.module.mod.sum([1, 2])
        assert r == 3
        r = self.module.mod.fsum([1, 2])
        assert r == 3


@pytest.mark.slow
class TestF2cmapOption(TestAssumedShapeSumExample):
    @pytest.fixture(scope="function", autouse=True)
    def setup_method_fixture(self, f2py_limited_api):
        # Use a custom file name for .f2py_f2cmap
        self.sources = list(self.sources)
        f2cmap_src = self.sources.pop(-1)

        self.f2cmap_file = tempfile.NamedTemporaryFile(delete=False)
        with open(f2cmap_src, "rb") as f:
            self.f2cmap_file.write(f.read())
        self.f2cmap_file.close()

        self.sources.append(self.f2cmap_file.name)
        self.options = ["--f2cmap", self.f2cmap_file.name]

        yield from super()._setup_method_fixture(f2py_limited_api)

        os.unlink(self.f2cmap_file.name)
