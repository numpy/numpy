import os
import pytest
import tempfile

from . import util


@pytest.fixture(scope="function")
def custom_f2cmap(request):
    test_instance = request.instance
    original_sources = test_instance.sources.copy()

    f2cmap_src = original_sources.pop(-1)
    f2cmap_file = tempfile.NamedTemporaryFile(delete=False)
    with open(f2cmap_src, "rb") as f:
        f2cmap_file.write(f.read())
    f2cmap_file.close()

    test_instance.sources = original_sources + [f2cmap_file.name]
    test_instance.options = ["--f2cmap", f2cmap_file.name]

    def cleanup_f2cmap_file():
        os.unlink(f2cmap_file.name)

    request.addfinalizer(cleanup_f2cmap_file)


@pytest.mark.usefixtures("build_module")
class TestAssumedShapeSumExample(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "assumed_shape", "foo_free.f90"),
        util.getpath("tests", "src", "assumed_shape", "foo_use.f90"),
        util.getpath("tests", "src", "assumed_shape", "precision.f90"),
        util.getpath("tests", "src", "assumed_shape", "foo_mod.f90"),
        util.getpath("tests", "src", "assumed_shape", ".f2py_f2cmap"),
    ]

    @pytest.mark.slow
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


class TestF2cmapOption(TestAssumedShapeSumExample):
    @pytest.mark.usefixtures("custom_f2cmap")
    def test_all(self):
        super().test_all()
