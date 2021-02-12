import abc
import sys
from threading import Lock
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from numpy import dtype, ndarray, uint32, uint64, unsignedinteger
from numpy.typing import DTypeLike, _ArrayLikeInt_co, _DTypeLikeUInt, _ShapeLike, _SupportsDType

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

_T = TypeVar("_T")

_UIntType = TypeVar("_UIntType", uint64, uint32)
_DTypeLike = Union[
    Type[_UIntType],
    dtype[_UIntType],
    _SupportsDType[dtype[_UIntType]],
]

class _Interface(NamedTuple):
    state_address: Any
    state: Any
    next_uint64: Any
    next_uint32: Any
    next_double: Any
    bit_generator: Any

class ISeedSequence(abc.ABC):
    @overload
    @abc.abstractmethod
    def generate_state(
        self, n_words: int, dtype: _DTypeLike[_UIntType] = ...
    ) -> ndarray[Any, dtype[_UIntType]]: ...
    @overload
    @abc.abstractmethod
    def generate_state(
        self, n_words: int, dtype: _DTypeLikeUInt = ...
    ) -> ndarray[Any, dtype[unsignedinteger[Any]]]: ...

class ISpawnableSeedSequence(ISeedSequence):
    @abc.abstractmethod
    def spawn(self: _T, n_children: int) -> List[_T]: ...

class SeedlessSeedSequence(ISpawnableSeedSequence):
    @overload
    def generate_state(
        self, n_words: int, dtype: _DTypeLike[_UIntType] = ...
    ) -> ndarray[Any, dtype[_UIntType]]: ...
    @overload
    def generate_state(
        self, n_words: int, dtype: _DTypeLikeUInt = ...
    ) -> ndarray[Any, dtype[unsignedinteger[Any]]]: ...
    def spawn(self: _T, n_children: int) -> List[_T]: ...

class SeedSequence(ISpawnableSeedSequence):
    entropy: Union[None, int, Sequence[int]]
    spawn_key: Sequence[int]
    pool_size: int
    n_children_spawned: int
    pool: ndarray[Any, dtype[uint32]]
    def __init__(
        self,
        entropy: Union[None, int, Sequence[int]] = ...,
        *,
        spawn_key: Tuple[int, ...] = ...,
        pool_size: int = ...,
        n_children_spawned: int = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def state(
        self,
    ) -> Dict[str, Union[None, Sequence[int], int, Tuple[int, ...]]]: ...
    def generate_state(self, n_words: int, dtype: DTypeLike = ...) -> ndarray[Any, Any]: ...
    def spawn(self, n_children: int) -> List[SeedSequence]: ...

class BitGenerator:
    lock: Lock
    def __init__(self, seed: Union[None, _ArrayLikeInt_co, SeedSequence] = ...) -> None: ...
    def __getstate__(self) -> Dict[str, Any]: ...
    def __setstate__(self, state: Dict[str, Any]) -> None: ...
    def __reduce__(
        self,
    ) -> Tuple[Callable[[str], BitGenerator], Tuple[str], Tuple[Dict[str, Any]]]: ...
    @property
    def state(self) -> Mapping[str, Any]: ...
    @state.setter
    def state(self, value: Mapping[str, Any]) -> None: ...
    @overload
    def random_raw(self, size: None = ..., output: Literal[True] = ...) -> int: ...  # type: ignore[misc]
    @overload
    def random_raw(self, size: _ShapeLike = ..., output: Literal[True] = ...) -> ndarray[Any, dtype[uint64]]: ...  # type: ignore[misc]
    @overload
    def random_raw(self, size: Optional[_ShapeLike] = ..., output: Literal[False] = ...) -> None: ...  # type: ignore[misc]
    def _benchmark(self, cnt: int, method: str = ...) -> None: ...
    @property
    def ctypes(self) -> _Interface: ...
    @property
    def cffi(self) -> _Interface: ...
