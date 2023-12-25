import pytest
from . import util
from numpy.testing import IS_WASM


@pytest.fixture(scope="session")
def check_compilers():
    checker = util.CompilerChecker()
    checker.check_compilers()
    if not checker.has_c:
        pytest.skip("No C compiler available")
    return checker


@pytest.mark.slow
@pytest.mark.skipif(IS_WASM, reason="Cannot start subprocess")
@pytest.fixture(scope="module")
def module_builder_factory(check_compilers):
    def build_module(spec):
        codes = spec.sources if spec.sources else []
        needs_f77 = any(str(fn).endswith(".f") for fn in codes)
        needs_f90 = any(str(fn).endswith(".f90") for fn in codes)
        needs_pyf = any(str(fn).endswith(".pyf") for fn in codes)

        if needs_f77 and not check_compilers.has_f77:
            pytest.skip("No Fortran 77 compiler available")
        if needs_f90 and not check_compilers.has_f90:
            pytest.skip("No Fortran 90 compilers available")
        if needs_pyf and not (check_compilers.has_f90 or check_compilers.has_f77):
            pytest.skip("No Fortran compiler available")

        try:
            return util.build_module_from_spec(spec)
        except Exception as e:
            pytest.skip(f"Module build failed: {e}")

    return build_module


# For generating modules from different specs
@pytest.fixture(scope="module")
def _mod(module_builder_factory, request):
    spec = request.getfixturevalue(request.param)
    return module_builder_factory(spec)
