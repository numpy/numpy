from typing import Dict, Union

from numpy.random.bit_generator import BitGenerator, SeedSequence
from numpy.typing import _ArrayLikeInt_co

_PCG64State = Dict[
    str,
    Union[str, int, Dict[str, int]],
]

class PCG64(BitGenerator):
    def __init__(self, seed: Union[None, _ArrayLikeInt_co, SeedSequence] = ...) -> None: ...
    def jumped(self, jumps: int = ...) -> PCG64: ...
    @property
    def state(
        self,
    ) -> _PCG64State: ...
    @state.setter
    def state(
        self,
        value: _PCG64State,
    ) -> None: ...
    def advance(self, delta: int) -> PCG64: ...
