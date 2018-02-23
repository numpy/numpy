cimport numpy as np
import numpy as np

from libc.stdint cimport uint32_t

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
    entropy : scalar or ndarray
        Entropy bits in 32-bit unsigned integers

    Notes
    -----
    On Unix-like machines, reads from ``/dev/urandom``. On Windows machines
    reads from the RSA algorithm provided by the cryptographic service
    provider.

    This function reads from the system entropy pool and so samples are
    not reproducible.  In particular, it does *NOT* make use of a
    RandomState, and so ``seed``, ``get_state`` and ``set_state`` have no
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