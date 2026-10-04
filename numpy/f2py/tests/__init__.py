import pytest

from numpy.testing import IS_EDITABLE, IS_IOS, IS_WASM

if IS_WASM or IS_IOS:
    pytest.skip(
        "Platform does not use or support Fortran",
        allow_module_level=True
    )


if IS_EDITABLE:
    pytest.skip(
        "Editable install doesn't support tests with a compile step",
        allow_module_level=True
    )
