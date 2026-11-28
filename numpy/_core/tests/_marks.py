import sys

import pytest

longdouble_fpe_mark = pytest.mark.xfail(
    any(name in sys.platform for name in ["android", "bsd"]),
    reason="(c)longdouble may not raise FPE errors as expected on this platform, "
    "see gh-24876, gh-23379",
)
