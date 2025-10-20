from typing import ClassVar, Final

import numpy as np

from ._polybase import ABCPolyBase
from ._polytypes import (
    _Array1,
    _Array2,
    _FuncBinOp,
    _FuncCompanion,
    _FuncDer,
    _FuncFit,
    _FuncFromRoots,
    _FuncInteg,
    _FuncLine,
    _FuncPow,
    _FuncRoots,
    _FuncUnOp,
    _FuncVal,
    _FuncVal2D,
    _FuncVal3D,
    _FuncValFromRoots,
    _FuncVander,
    _FuncVander2D,
    _FuncVander3D,
)
from .polyutils import trimcoef as polytrim

__all__ = [
    "polyzero",
    "polyone",
    "polyx",
    "polydomain",
    "polyline",
    "polyadd",
    "polysub",
    "polymulx",
    "polymul",
    "polydiv",
    "polypow",
    "polyval",
    "polyvalfromroots",
    "polyder",
    "polyint",
    "polyfromroots",
    "polyvander",
    "polyfit",
    "polytrim",
    "polyroots",
    "Polynomial",
    "polyval2d",
    "polyval3d",
    "polygrid2d",
    "polygrid3d",
    "polyvander2d",
    "polyvander3d",
    "polycompanion",
]

polydomain: Final[_Array2[np.float64]] = ...
polyzero: Final[_Array1[np.int_]] = ...
polyone: Final[_Array1[np.int_]] = ...
polyx: Final[_Array2[np.int_]] = ...

polyline: Final[_FuncLine] = ...
polyfromroots: Final[_FuncFromRoots] = ...
polyadd: Final[_FuncBinOp] = ...
polysub: Final[_FuncBinOp] = ...
polymulx: Final[_FuncUnOp] = ...
polymul: Final[_FuncBinOp] = ...
polydiv: Final[_FuncBinOp] = ...
polypow: Final[_FuncPow] = ...
polyder: Final[_FuncDer] = ...
polyint: Final[_FuncInteg] = ...
polyval: Final[_FuncVal] = ...
polyval2d: Final[_FuncVal2D] = ...
polyval3d: Final[_FuncVal3D] = ...
polyvalfromroots: Final[_FuncValFromRoots] = ...
polygrid2d: Final[_FuncVal2D] = ...
polygrid3d: Final[_FuncVal3D] = ...
polyvander: Final[_FuncVander] = ...
polyvander2d: Final[_FuncVander2D] = ...
polyvander3d: Final[_FuncVander3D] = ...
polyfit: Final[_FuncFit] = ...
polycompanion: Final[_FuncCompanion] = ...
polyroots: Final[_FuncRoots] = ...

class Polynomial(ABCPolyBase[None]):
    basis_name: ClassVar[None] = None  # pyright: ignore[reportIncompatibleMethodOverride]
    domain: _Array2[np.float64] = ...  # pyright: ignore[reportIncompatibleMethodOverride]
    window: _Array2[np.float64] = ...  # pyright: ignore[reportIncompatibleMethodOverride]
