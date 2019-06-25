cimport numpy as np
import numpy as np

from libc.stdint cimport uint32_t, uint64_t

__all__ = ['random_entropy', 'seed_by_array']

np.import_array()

cdef extern from "src/splitmix64/splitmix64.h":
    cdef uint64_t splitmix64_next(uint64_t *state)  nogil

cdef extern from "src/entropy/entropy.h":
    cdef bint entropy_getbytes(void* dest, size_t size)
    cdef bint entropy_fallback_getbytes(void *dest, size_t size)

cdef Py_ssize_t compute_numel(size):
    cdef Py_ssize_t i, n = 1
    if isinstance(size, tuple):
        for i in range(len(size)):
            n *= size[i]
    else:
        n = size
    return n


def seed_by_array(object seed, Py_ssize_t n):
    """
    Transforms a seed array into an initial state

    Parameters
    ----------
    seed: ndarray, 1d, uint64
        Array to use.  If seed is a scalar, promote to array.
    n : int
        Number of 64-bit unsigned integers required

    Notes
    -----
    Uses splitmix64 to perform the transformation
    """
    cdef uint64_t seed_copy = 0
    cdef uint64_t[::1] seed_array
    cdef uint64_t[::1] initial_state
    cdef Py_ssize_t seed_size, iter_bound
    cdef int i, loc = 0

    if hasattr(seed, 'squeeze'):
        seed = seed.squeeze()
    arr = np.asarray(seed)
    if arr.shape == ():
        err_msg = 'Scalar seeds must be integers between 0 and 2**64 - 1'
        if not np.isreal(arr):
            raise TypeError(err_msg)
        int_seed = int(seed)
        if int_seed != seed:
            raise TypeError(err_msg)
        if int_seed < 0 or int_seed > 2**64 - 1:
            raise ValueError(err_msg)
        seed_array = np.array([int_seed], dtype=np.uint64)
    elif issubclass(arr.dtype.type, np.inexact):
        raise TypeError('seed array must be integers')
    else:
        err_msg = "Seed values must be integers between 0 and 2**64 - 1"
        obj = np.asarray(seed).astype(np.object)
        if obj.ndim != 1:
            raise ValueError('Array-valued seeds must be 1-dimensional')
        if not np.isreal(obj).all():
            raise TypeError(err_msg)
        if ((obj > int(2**64 - 1)) | (obj < 0)).any():
            raise ValueError(err_msg)
        try:
            obj_int = obj.astype(np.uint64, casting='unsafe')
        except ValueError:
            raise ValueError(err_msg)
        if not (obj == obj_int).all():
            raise TypeError(err_msg)
        seed_array = obj_int

    seed_size = seed_array.shape[0]
    iter_bound = n if n > seed_size else seed_size

    initial_state = <np.ndarray>np.empty(n, dtype=np.uint64)
    for i in range(iter_bound):
        if i < seed_size:
            seed_copy ^= seed_array[i]
        initial_state[loc] = splitmix64_next(&seed_copy)
        loc += 1
        if loc == n:
            loc = 0

    return np.array(initial_state)


def random_entropy(size=None, source='system'):
    """
    random_entropy(size=None, source='system')

    Read entropy from the system cryptographic provider

    Parameters
    ----------
    size : int or tuple of ints, optional
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    source : str {'system', 'fallback'}
        Source of entropy.  'system' uses system cryptographic pool.
        'fallback' uses a hash of the time and process id.

    Returns
    -------
    entropy : scalar or array
        Entropy bits in 32-bit unsigned integers. A scalar is returned if size
        is `None`.

    Notes
    -----
    On Unix-like machines, reads from ``/dev/urandom``. On Windows machines
    reads from the RSA algorithm provided by the cryptographic service
    provider.

    This function reads from the system entropy pool and so samples are
    not reproducible.  In particular, it does *NOT* make use of a
    BitGenerator, and so ``seed`` and setting ``state`` have no
    effect.

    Raises RuntimeError if the command fails.
    """
    cdef bint success = True
    cdef Py_ssize_t n = 0
    cdef uint32_t random = 0
    cdef uint32_t [:] randoms

    if source not in ('system', 'fallback'):
        raise ValueError('Unknown value in source.')

    if size is None:
        if source == 'system':
            success = entropy_getbytes(<void *>&random, 4)
        else:
            success = entropy_fallback_getbytes(<void *>&random, 4)
    else:
        n = compute_numel(size)
        randoms = np.zeros(n, dtype=np.uint32)
        if source == 'system':
            success = entropy_getbytes(<void *>(&randoms[0]), 4 * n)
        else:
            success = entropy_fallback_getbytes(<void *>(&randoms[0]), 4 * n)
    if not success:
        raise RuntimeError('Unable to read from system cryptographic provider')

    if n == 0:
        return random
    return np.asarray(randoms).reshape(size)
