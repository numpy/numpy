import os
import pytest
import tempfile

from . import util


@pytest.fixture(scope="function")
def base_assumed_shape_spec():
    return util.F2PyModuleSpec(
        test_class_name="TestAssumedShapeSumExample",
        sources=[
            util.getpath("tests", "src", "assumed_shape", "foo_free.f90"),
            util.getpath("tests", "src", "assumed_shape", "foo_use.f90"),
            util.getpath("tests", "src", "assumed_shape", "precision.f90"),
            util.getpath("tests", "src", "assumed_shape", "foo_mod.f90"),
            util.getpath("tests", "src", "assumed_shape", ".f2py_f2cmap"),
        ],
    )


@pytest.fixture(scope="function")
def f2cmap_assumed_shape_spec(base_assumed_shape_spec):
    original_sources = base_assumed_shape_spec.sources.copy()
    f2cmap_src = original_sources.pop(-1)

    f2cmap_file = tempfile.NamedTemporaryFile(delete=False)
    with open(f2cmap_src, "rb") as f:
        f2cmap_file.write(f.read())
    f2cmap_file.close()

    modified_spec = util.F2PyModuleSpec(
        test_class_name="TestF2cmapOption",
        sources=original_sources + [f2cmap_file.name],
        options=["--f2cmap", f2cmap_file.name],
    )

    yield modified_spec

    # Cleanup
    os.unlink(f2cmap_file.name)


@pytest.mark.slow
@pytest.fixture(scope="function")
def _mod(module_builder_factory, request):
    spec = request.getfixturevalue(request.param)
    return module_builder_factory(spec)


@pytest.mark.parametrize(
    "_mod",
    ["base_assumed_shape_spec", "f2cmap_assumed_shape_spec"],
    indirect=True,
)
def test_assumed_shape(_mod):
    r = _mod.fsum([1, 2])
    assert r == 3
    r = _mod.sum([1, 2])
    assert r == 3
    r = _mod.sum_with_use([1, 2])
    assert r == 3

    r = _mod.mod.sum([1, 2])
    assert r == 3
    r = _mod.mod.fsum([1, 2])
    assert r == 3
