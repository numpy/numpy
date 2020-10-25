import f2py
import sys
import warnings

warnings.warn(
    "f2py is deprecated, use f2py instead", DeprecationWarning
)
sys.modules[__name__] = f2py