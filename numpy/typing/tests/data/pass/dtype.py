import numpy as np

dtype_obj = np.dtype(np.str_)

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
    dtype = np.dtype(float)


np.dtype(Test())

# Methods and attributes
dtype_obj.base
dtype_obj.subdtype
dtype_obj.newbyteorder()
dtype_obj.type
