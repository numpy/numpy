import os

import pytest

from numpy.testing import IS_EDITABLE, IS_WASM


if IS_WASM:
    pytest.skip(
        "WASM/Pyodide does not use or support Fortran",
        allow_module_level=True
    )


if IS_EDITABLE:
    pytest.skip(
        "Editable install doesn't support tests with a compile step",
        allow_module_level=True
    )


if os.environ.get("CIBW_ARCHS_MACOS") == "x86_64":
    pytest.skip(
        "f2py compile tests don't work when running on arm64 under Rosetta",
        allow_module_level=True
    )
