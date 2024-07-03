import sys
from contextvars import ContextVar

_default_format_options_dict = {
    "edgeitems": 3,  # repr N leading and trailing items of each dimension
    "threshold": 1000,  # total items > triggers array summarization
    "floatmode": "maxprec",
    "precision": 8,  # precision of floating point representations
    "suppress": False,  # suppress printing small floating values in exp format
    "linewidth": 75,
    "nanstr": "nan",
    "infstr": "inf",
    "sign": "-",
    "formatter": None,
    # Internally stored as an int to simplify comparisons; converted from/to
    # str/False on the way in/out.
    'legacy': sys.maxsize,
    'override_repr': None,
}

_format_options = ContextVar(
    "format_options", default=_default_format_options_dict.copy())
