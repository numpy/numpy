from typing import Any, Dict, Literal, Union

from numpy import dtype as dtype
from numpy import ndarray as ndarray
from numpy import uint64
from numpy.random.bit_generator import BitGenerator as BitGenerator
from numpy.random.bit_generator import SeedSequence as SeedSequence
from numpy.typing import _ArrayLikeInt_co

class SFC64(BitGenerator):
    def __init__(
        self, seed: Union[None, int, _ArrayLikeInt_co, SeedSequence] = ...
    ) -> None: ...
    def state(
        self,
    ) -> Dict[
        Literal["bit_generator", "state", "has_uint32", "uinteger"],
        Union[str, int, Dict[str, ndarray[Any, dtype[uint64]]]],
    ]: ...
    @state.setter
    def state(
        self,
        value: Dict[
            Literal["bit_generator", "state", "has_uint32", "uinteger"],
            Union[str, int, Dict[str, ndarray[Any, dtype[uint64]]]],
        ],
    ) -> None: ...
