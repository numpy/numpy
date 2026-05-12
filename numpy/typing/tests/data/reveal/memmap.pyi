from typing import Any, Literal, assert_type

import numpy as np

type _Memmap[ScalarT: np.generic] = np.memmap[tuple[Any, ...], np.dtype[ScalarT]]

memmap_obj: _Memmap[np.str_]

assert_type(np.memmap.__array_priority__, float)
assert_type(memmap_obj.__array_priority__, float)
assert_type(memmap_obj.filename, str | None)
assert_type(memmap_obj.offset, int)
assert_type(memmap_obj.mode, Literal["r", "r+", "w+", "c"])
assert_type(memmap_obj.flush(), None)

assert_type(np.memmap("file.txt", offset=5), _Memmap[np.uint8])
assert_type(np.memmap(b"file.txt", dtype=np.float64, shape=(10, 3)), _Memmap[np.float64])
with open("file.txt", "rb") as f:
    assert_type(np.memmap(f, dtype=float, order="K"), np.memmap)

assert_type(memmap_obj.__array_finalize__(object()), None)
