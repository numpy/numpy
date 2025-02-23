"""
Try calling every method `numpy.ndarray` with deterministic
but nonsensical arguments to see if we can induce a segfault.
"""
import pytest

import warnings
import itertools
import numpy as np

from typing import Tuple, List, Optional
from numpy.typing import NDArray


def _hash(value: tuple) -> int:
    """
    To verify our combinations logic hasn't added a bunch
    of duplicates: de-duplicate our sample data using a hash.
    """
    hashed = -1
    constant = 55508814645456645
    for i, v in enumerate(value):
        if hasattr(v, 'tobytes') and isinstance(v.size, int):
            hashed ^= hash(v.tobytes()) * (i + 1) * constant
        else:
            # try hashing the string representation
            hashed ^= hash(str(v)) * (i + 1) * constant
    return hashed


@pytest.fixture
def source_arrays(max_len: Optional[int] = 3) -> NDArray:
    """
    Generate source arrays constructed through several
    methods and dtypes in an attempt to break error
    handling of numpy array methods. If you are suspicious
    you can expand the number of checks.

    Parameters
    ----------
    max_len
     If None, return many arrays with different
     ordering, dtypes, and constructors or if passed
     as an integer cap the return length to this value.

    Returns
    ---------
    arrays
      Array with different dtype, ordering, etc.
    """
    # construct arrays with different methods
    constructors = [np.empty,
                    np.zeros,
                    np.ones,
                    np.random.random]

    # try creating test arrays of multiple kinds
    constructors_opt = [(0,),
                        (10,),
                        (((10, 3)),),
                        (((10, 3, 3)),)]

    # check a few data types
    dtypes_opt = ['><',  # check endian-ness
                  'if',  # check int and float types
                  [2, 4, 8]]  # check sizes
    dtypes = [np.dtype(endian + kind + str(size))
              for endian, kind, size in
              itertools.product(*dtypes_opt)]
    # add default python types
    dtypes.extend([int, float, bool])

    # try with different ordering preferences
    orders = ['C', 'F', 'A', 'K', None]

    # construct the arrays of various kinds
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        arrays = [con(*opt).astype(dtype=dtype,
                                   order=order)
                  for con, opt, dtype, order
                  in itertools.product(constructors,
                                       constructors_opt,
                                       dtypes,
                                       orders)]

    if max_len is not None and max_len < len(arrays):
        # try to get a distributed group from the created arrays
        return arrays[::len(arrays) // max_len]

    return arrays


@pytest.fixture
def source_args(max_len: int = 2) -> List[Tuple]:
    """
    Generate possible combinations of arguments
    to call numpy methods with.

    Parameters
    -----------
    max_len
      Maximum length of guesses to return.
      This will expand the number of guesses to check
      substantially, values above 3 are probably too large.

    Returns
    ---------
    args
      Different argument options as a tuple.
    """
    dim = ((5, 3))
    flat = [2.3,
            1,
            10,
            [3, -1],
            (4.2,),
            np.iinfo(np.int64).max,
            np.iinfo(np.int64).min,
            (11012.0, 1.1, 1.09),
            {'shape': 10},
            np.int64,
            np.float64,
            True,
            False,
            np.random.random(dim),
            np.random.random(dim[::1]),
            'shape']

    # start with no arguments
    attempts = [tuple()]
    # add a single argument from our guesses
    attempts.extend([(A,) for A in flat])
    # add all possible variants of our guesses
    for length in range(2, 1 + max_len):
        attempts.extend(
            tuple(G) for G in itertools.product(flat, repeat=length))

    # get a string copy of our return args that's de-duplicated
    checks = [str(i) for i in {_hash(a): a for a in attempts}.values()]

    # check a few key values to make sure our generation logic is OK
    assert str(tuple()) in checks
    assert str(tuple([False, False])) in checks
    assert str(tuple([False, True])) in checks
    assert str(tuple([True, True])) in checks
    assert str(tuple([True, False])) in checks
    # checks is de-duplicated so assert we aren't doing the same thing over
    assert len(checks) == len(attempts)

    return attempts


def test_junk_calls(source_arrays, source_args):
    """
    Call every method available on an `numpy.ndarray`
    with bad data to see if we can induce a segfault.
    """
    # a list of all methods on the numpy array
    methods = dir(np.empty(1))

    with warnings.catch_warnings():
        # ignore all warnings inside this context manager
        warnings.filterwarnings("ignore")
        # loop through the named methods
        for method in methods:
            print('checking method: `{}`'.format(method))
            # if you are suspicious you can set `max_len=None` here
            for array in source_arrays:
                # if you are suspicious you can set `max_len=3` here
                for args in source_args:
                    try:
                        # evaluate the methods of an `ndarray` with junk
                        eval('array.{method}(*args)'.format(method=method))
                    except BaseException:
                        # should not be segfaulting
                        pass
