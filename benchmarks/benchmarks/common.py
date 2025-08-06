import random
from functools import lru_cache
from pathlib import Path

import numpy as np

# Various pre-crafted datasets/variables for testing
# !!! Must not be changed -- only appended !!!
# while testing numpy we better not rely on numpy to produce random
# sequences
random.seed(1)
# but will seed it nevertheless
np.random.seed(1)

nx, ny = 1000, 1000
# reduced squares based on indexes_rand, primarily for testing more
# time-consuming functions (ufunc, linalg, etc)
nxs, nys = 100, 100

# a list of interesting types to test
TYPES1 = [
    'int16', 'float16',
    'int32', 'float32',
    'int64', 'float64', 'complex64',
    'complex128',
]

DLPACK_TYPES = [
    'int16', 'float16',
    'int32', 'float32',
    'int64', 'float64', 'complex64',
    'complex128', 'bool',
]

# Path for caching
CACHE_ROOT = Path(__file__).resolve().parent.parent / 'env' / 'numpy_benchdata'

# values which will be used to construct our sample data matrices
# replicate 10 times to speed up initial imports of this helper
# and generate some redundancy

@lru_cache(typed=True)
def get_values():
    rnd = np.random.RandomState(1804169117)
    values = np.tile(rnd.uniform(0, 100, size=nx * ny // 10), 10)
    return values


@lru_cache(typed=True)
def get_square(dtype):
    values = get_values()
    arr = values.astype(dtype=dtype).reshape((nx, ny))

    # adjust complex ones to have non-degenerated imagery part -- use
    # original data transposed for that
    if arr.dtype.kind == 'c':
        arr += arr.T * 1j

    return arr

@lru_cache(typed=True)
def get_squares():
    return {t: get_square(t) for t in sorted(TYPES1)}


@lru_cache(typed=True)
def get_square_(dtype):
    arr = get_square(dtype)
    return arr[:nxs, :nys]


@lru_cache(typed=True)
def get_squares_():
    # smaller squares
    return {t: get_square_(t) for t in sorted(TYPES1)}


@lru_cache(typed=True)
def get_indexes():
    indexes = list(range(nx))
    # so we do not have all items
    indexes.pop(5)
    indexes.pop(95)

    indexes = np.array(indexes)
    return indexes


@lru_cache(typed=True)
def get_indexes_rand():
    rnd = random.Random(1)

    indexes_rand = get_indexes().tolist()       # copy
    rnd.shuffle(indexes_rand)         # in-place shuffle
    indexes_rand = np.array(indexes_rand)
    return indexes_rand


@lru_cache(typed=True)
def get_indexes_():
    # smaller versions
    indexes = get_indexes()
    indexes_ = indexes[indexes < nxs]
    return indexes_


@lru_cache(typed=True)
def get_indexes_rand_():
    indexes_rand = get_indexes_rand()
    indexes_rand_ = indexes_rand[indexes_rand < nxs]
    return indexes_rand_


@lru_cache(typed=True)
def get_data(size, dtype, ip_num=0, zeros=False, finite=True, denormal=False):
    """
    Generates a cached random array that covers several scenarios that
    may affect the benchmark for fairness and to stabilize the benchmark.

    Parameters
    ----------
    size: int
        Array length.

    dtype: dtype or dtype specifier

    ip_num: int
        Input number, to avoid memory overload
        and to provide unique data for each operand.

    zeros: bool
        Spreading zeros along with generated data.

    finite: bool
        Avoid spreading fp special cases nan/inf.

    denormal:
        Spreading subnormal numbers along with generated data.
    """
    dtype = np.dtype(dtype)
    dname = dtype.name
    cache_name = f'{dname}_{size}_{ip_num}_{int(zeros)}'
    if dtype.kind in 'fc':
        cache_name += f'{int(finite)}{int(denormal)}'
    cache_name += '.bin'
    cache_path = CACHE_ROOT / cache_name
    if cache_path.exists():
        return np.fromfile(cache_path, dtype)

    array = np.ones(size, dtype)
    rands = []
    if dtype.kind == 'i':
        dinfo = np.iinfo(dtype)
        scale = 8
        if zeros:
            scale += 1
        lsize = size // scale
        for low, high in (
            (-0x80, -1),
            (1, 0x7f),
            (-0x8000, -1),
            (1, 0x7fff),
            (-0x80000000, -1),
            (1, 0x7fffffff),
            (-0x8000000000000000, -1),
            (1, 0x7fffffffffffffff),
        ):
            rands += [np.random.randint(
                max(low, dinfo.min),
                min(high, dinfo.max),
                lsize, dtype
            )]
    elif dtype.kind == 'u':
        dinfo = np.iinfo(dtype)
        scale = 4
        if zeros:
            scale += 1
        lsize = size // scale
        for high in (0xff, 0xffff, 0xffffffff, 0xffffffffffffffff):
            rands += [np.random.randint(1, min(high, dinfo.max), lsize, dtype)]
    elif dtype.kind in 'fc':
        scale = 1
        if zeros:
            scale += 1
        if not finite:
            scale += 2
        if denormal:
            scale += 1
        dinfo = np.finfo(dtype)
        lsize = size // scale
        rands = [np.random.rand(lsize).astype(dtype)]
        if not finite:
            rands += [
                np.empty(lsize, dtype=dtype), np.empty(lsize, dtype=dtype)
            ]
            rands[1].fill(float('nan'))
            rands[2].fill(float('inf'))
        if denormal:
            rands += [np.empty(lsize, dtype=dtype)]
            rands[-1].fill(dinfo.smallest_subnormal)

    if rands:
        if zeros:
            rands += [np.zeros(lsize, dtype)]
        stride = len(rands)
        for start, r in enumerate(rands):
            array[start:len(r) * stride:stride] = r

    if not CACHE_ROOT.exists():
        CACHE_ROOT.mkdir(parents=True)
    array.tofile(cache_path)
    return array

class Benchmark:
    pass
