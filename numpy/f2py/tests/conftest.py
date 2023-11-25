import pytest
from . import util


@pytest.fixture(scope="session")
def check_compilers():
    checker = util.CompilerChecker()
    checker.check_compilers()
    if not checker.has_c:
        pytest.skip("No C compiler available")
    return checker

@pytest.fixture(scope="class")
def build_module(request, check_compilers):
    test_instance = request.cls()
    codes = test_instance.sources if test_instance.sources else []
    needs_f77 = any(str(fn).endswith(".f") for fn in codes)
    needs_f90 = any(str(fn).endswith(".f90") for fn in codes)
    needs_pyf = any(str(fn).endswith(".pyf") for fn in codes)

    if needs_f77 and not check_compilers.has_f77:
        pytest.skip("No Fortran 77 compiler available")
    if needs_f90 and not check_compilers.has_f90:
        pytest.skip("No Fortran 90 compilers available")
    if needs_pyf and not (check_compilers.has_f90 or check_compilers.has_f77):
        pytest.skip("No Fortran compiler available")

    test_instance.build_mod()
    request.cls.module = test_instance.module
