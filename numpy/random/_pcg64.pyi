from typing import Dict, Literal, Union

from numpy.random.bit_generator import BitGenerator as BitGenerator
from numpy.random.bit_generator import SeedSequence as SeedSequence
from numpy.typing import _ArrayLikeInt_co

class PCG64(BitGenerator):
    def __init__(
        self, seed: Union[None, int, _ArrayLikeInt_co, SeedSequence] = ...
    ) -> None: ...
    def jumped(self, jumps=1) -> PCG64: ...
    @property
    def state(
        self,
    ) -> Dict[
        Literal["bit_generator", "state", "has_uint32", "uinteger"],
        Union[str, int, Dict[str, int]],
    ]: ...
    @state.setter
    def state(
        self,
        value: Dict[
            Literal["bit_generator", "state", "has_uint32", "uinteger"],
            Union[str, int, Dict[str, int]],
        ],
    ) -> None: ...
    def advance(self, delta: int) -> PCG64: ...
