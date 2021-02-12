from typing import Any, Dict, Union

from numpy import dtype, ndarray, uint64
from numpy.random.bit_generator import BitGenerator, SeedSequence
from numpy.typing import _ArrayLikeInt_co

_PhiloxState = Dict[
    str,
    Union[
        str,
        int,
        ndarray[Any, dtype[uint64]],
        Dict[str, ndarray[Any, dtype[uint64]]],
    ],
]

class Philox(BitGenerator):
    def __init__(
        self,
        seed: Union[None, _ArrayLikeInt_co, SeedSequence] = ...,
        counter: Union[None, _ArrayLikeInt_co] = ...,
        key: Union[None, _ArrayLikeInt_co] = ...,
    ) -> None: ...
    @property
    def state(
        self,
    ) -> _PhiloxState: ...
    @state.setter
    def state(
        self,
        value: _PhiloxState,
    ) -> None: ...
    def jumped(self, jumps: int = ...) -> Philox: ...
    def advance(self, delta: int) -> Philox: ...
