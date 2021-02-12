from typing import Any, Dict, Union

from numpy import dtype as dtype
from numpy import ndarray as ndarray
from numpy import uint64
from numpy.random.bit_generator import BitGenerator, SeedSequence
from numpy.typing import _ArrayLikeInt_co

_SFC64State = Dict[
    str,
    Union[str, int, Dict[str, ndarray[Any, dtype[uint64]]]],
]

class SFC64(BitGenerator):
    def __init__(self, seed: Union[None, _ArrayLikeInt_co, SeedSequence] = ...) -> None: ...
    @property
    def state(
        self,
    ) -> _SFC64State: ...
    @state.setter
    def state(
        self,
        value: _SFC64State,
    ) -> None: ...
