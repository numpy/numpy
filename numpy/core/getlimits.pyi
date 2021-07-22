from typing import Any, Generic, List, Type, TypeVar

from numpy import (
    finfo as finfo,
    iinfo as iinfo,
    floating,
    signedinteger,
)

from numpy.typing import NBitBase, NDArray

_NBit = TypeVar("_NBit", bound=NBitBase)

__all__: List[str]

class MachArLike(Generic[_NBit]):
    def __init__(
        self,
        ftype: Type[floating[_NBit]],
        *,
        eps: floating[Any],
        epsneg: floating[Any],
        huge: floating[Any],
        tiny: floating[Any],
        ibeta: int,
        smallest_subnormal: None | floating[Any] = ...,
        # Expand `**kwargs` into keyword-only arguments
        machep: int,
        negep: int,
        minexp: int,
        maxexp: int,
        it: int,
        iexp: int,
        irnd: int,
        ngrd: int,
    ) -> None: ...
    @property
    def smallest_subnormal(self) -> NDArray[floating[_NBit]]: ...
    eps: NDArray[floating[_NBit]]
    epsilon: NDArray[floating[_NBit]]
    epsneg: NDArray[floating[_NBit]]
    huge: NDArray[floating[_NBit]]
    ibeta: signedinteger[_NBit]
    iexp: int
    irnd: int
    it: int
    machep: int
    maxexp: int
    minexp: int
    negep: int
    ngrd: int
    precision: int
    resolution: NDArray[floating[_NBit]]
    smallest_normal: NDArray[floating[_NBit]]
    tiny: NDArray[floating[_NBit]]
    title: str
    xmax: NDArray[floating[_NBit]]
    xmin: NDArray[floating[_NBit]]
