import numpy as np

np.dtype(dtype=np.int64)
np.dtype(int)
np.dtype("int")
np.dtype(None)

np.dtype((int, 2))
np.dtype((int, (1,)))

np.dtype({"names": ["a", "b"], "formats": [int, float]})
np.dtype({"names": ["a"], "formats": [int], "titles": [object]})
np.dtype({"names": ["a"], "formats": [int], "titles": [object()]})

np.dtype([("name", np.unicode_, 16), ("grades", np.float64, (2,)), ("age", "int32")])

np.dtype(
    {
        "names": ["a", "b"],
        "formats": [int, float],
        "itemsize": 9,
        "aligned": False,
        "titles": ["x", "y"],
        "offsets": [0, 1],
    }
)

np.dtype((np.float_, float))


class Test:
    dtype = float


np.dtype(Test())
