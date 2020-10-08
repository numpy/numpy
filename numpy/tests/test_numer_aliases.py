from typing import Any, Type

import numpy as np
import numpy.typing as npt

number_aliases = [
    "byte",
    "short",
    "intc",
    "intp",
    "int_",
    "longlong",
    "ubyte",
    "ushort",
    "uintc",
    "uintp",
    "uint",
    "ulonglong",
    "half",
    "single",
    "float_",
    "longfloat",
    "csingle",
    "complex_",
    "clongfloat",
]

iterator = (
    (getattr(npt._number_aliases, k), getattr(np, k)) for k in number_aliases
)
for typ, ref_typ in iterator:  # type: Type[np.number], Type[np.number]
    assert typ in {ref_typ, Any}
