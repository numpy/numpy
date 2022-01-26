from typing import Any
import numpy as np
import numpy.typing as npt

def func(array_like: npt.ArrayLike) -> None: ...

a: npt.NDArray[Any]
b: npt.NDArray[np.int64]
c: int
d: list[int]
e: int | npt.NDArray[np.int64]
f: list[int | npt.NDArray[np.int64]]
g: tuple[int | npt.NDArray[np.int64], ...]
h: list[list[int] | npt.NDArray[np.int64]]
i: tuple[list[int] | npt.NDArray[np.int64], ...]
j: np.int64
k: list[np.str_ | npt.NDArray[np.int64]]
l: tuple[np.str_ | npt.NDArray[np.int64], ...]

reveal_type(func(a))  # E: None
reveal_type(func(b))  # E: None
reveal_type(func(c))  # E: None
reveal_type(func(d))  # E: None
reveal_type(func(e))  # E: None
reveal_type(func(f))  # E: None
reveal_type(func(g))  # E: None
reveal_type(func(h))  # E: None
reveal_type(func(i))  # E: None
reveal_type(func(j))  # E: None
reveal_type(func(k))  # E: None
reveal_type(func(l))  # E: None
