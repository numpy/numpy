from typing import TypedDict, type_check_only

from numpy import uint32, uint64
from numpy.typing import NDArray
from numpy.random.bit_generator import BitGenerator, SeedSequence
from numpy._typing import _ArrayLikeInt_co


@type_check_only
class _Romu64StateInternal(TypedDict):
    key: NDArray[uint64]

@type_check_only
class _Romu64State(TypedDict):
    bit_generator: str
    state: _Romu64StateInternal

@type_check_only
class _Romu32StateInternal(TypedDict):
    key: NDArray[uint32]

@type_check_only
class _Romu32State(TypedDict):
    bit_generator: str
    state: _Romu64StateInternal

@type_check_only
class Romu64(BitGenerator):
    def __init__(self, seed: None | _ArrayLikeInt_co | SeedSequence = ...) -> None: ...
    @property
    def state(self) -> _Romu64State: ...
    @state.setter
    def state(self, value: _Romu64State) -> None: ...

@type_check_only
class Romu32(BitGenerator):
    def __init__(self, seed: None | _ArrayLikeInt_co | SeedSequence = ...) -> None: ...
    @property
    def state(self) -> _Romu32State: ...
    @state.setter
    def state(self, value: _Romu32State) -> None: ...


class RomuQuad(Romu64):
    """
    RomuQuad
    --------

    More robust than anyone could need, but uses more registers than RomuTrio.
    Est. capacity >= 2^90 bytes. Register pressure = 8 (high). State size = 256 bits.
    """



class RomuTrio(Romu64):
    """
    RomuTrio
    --------

    Great for general purpose work, including huge jobs.
    Est. capacity = 2^75 bytes. Register pressure = 6. State size = 192 bits.

    """
    


class RomuDuo(Romu64):
    """
    RomuDuo
    -------

    Might be faster than RomuTrio due to using fewer registers, but might struggle with massive jobs.
    Est. capacity = 2^61 bytes. Register pressure = 5. State size = 128 bits.
    """
    


class RomuDuoJR(Romu64):
    """

    RomuDuoJR
    ---------

    The fastest generator using 64-bit arith., but not suited for huge jobs.
    Est. capacity = 2^51 bytes. Register pressure = 4. State size = 128 bits.
    """
    


class RomuQuad32(Romu32):
    """
    RomuQuad32
    ----------

    32-bit arithmetic: Good for general purpose use.
    Est. capacity >= 2^62 bytes. Register pressure = 7. State size = 128 bits.
    """


class RomuTrio32(Romu32):
    """
    RomuTrio32
    ----------

    32-bit arithmetic: Good for general purpose use, except for huge jobs.
    Est. capacity >= 2^53 bytes. Register pressure = 5. State size = 96 bits.
    """

    


