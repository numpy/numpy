import pytest

from numpy._utils import _pep440

MIN_VERSION = "1.2"


def importorskip_quaddtype():
    from importlib.metadata import PackageNotFoundError, version
    try:
        installed = version("numpy_quaddtype")
    except PackageNotFoundError:
        pytest.skip("numpy_quaddtype is not installed")
    if _pep440.Version(installed) < _pep440.Version(MIN_VERSION):
        pytest.skip(f"numpy_quaddtype >= {MIN_VERSION} is required")
    return pytest.importorskip("numpy_quaddtype")
