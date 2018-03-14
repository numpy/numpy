#!python
#cython: wraparound=False, nonecheck=False, boundscheck=False, cdivision=True

import numpy as np
cimport numpy as np
from distributions cimport *
np.import_array()




cdef object _rand_uint32_broadcast(np.ndarray low, np.ndarray high, object size, brng_t *state, object lock):
    """Array path for smaller integer types"""
    cdef uint32_t rng, last_rng, off, val, mask, out_val
    cdef uint32_t buf
    cdef uint32_t *out_data
    cdef uint64_t low_v, high_v
    cdef np.ndarray low_arr, high_arr, out_arr
    cdef np.npy_intp i, cnt
    cdef np.broadcast it
    cdef int buf_rem = 0


    # Array path
    low_arr = <np.ndarray>low
    high_arr = <np.ndarray>high
    if np.any(np.less(low_arr, 0)):
        raise ValueError('low is out of bounds for uint32')
    if np.any(np.greater(high_arr, 0X100000000ULL)):
        raise ValueError('high is out of bounds for uint32')
    if np.any(np.greater_equal(low_arr, high_arr)):
        raise ValueError('low >= high')

    low_arr = <np.ndarray>np.PyArray_FROM_OTF(low, np.NPY_UINT64, np.NPY_ALIGNED | np.NPY_FORCECAST)
    high_arr = <np.ndarray>np.PyArray_FROM_OTF(high, np.NPY_UINT64, np.NPY_ALIGNED | np.NPY_FORCECAST)

    if size is not None:
        out_arr = <np.ndarray>np.empty(size, np.uint32)
    else:
        it = np.PyArray_MultiIterNew2(low_arr, high_arr)
        out_arr = <np.ndarray>np.empty(it.shape, np.uint32)

    it = np.PyArray_MultiIterNew3(low_arr, high_arr, out_arr)
    out_data = <uint32_t *>np.PyArray_DATA(out_arr)
    cnt = np.PyArray_SIZE(out_arr)
    mask = last_rng = 0
    with lock, nogil:
        for i in range(cnt):
            low_v = (<uint64_t*>np.PyArray_MultiIter_DATA(it, 0))[0]
            high_v = (<uint64_t*>np.PyArray_MultiIter_DATA(it, 1))[0]
            rng = <uint32_t>((high_v - 1) - low_v)
            off = <uint32_t>(<uint64_t>low_v)

            if rng != last_rng:
                # Smallest bit mask >= max
                mask = <uint32_t>_gen_mask(rng)

            out_data[i] = random_buffered_bounded_uint32(state, off, rng, mask, &buf_rem, &buf)

            np.PyArray_MultiIter_NEXT(it)
    return out_arr


cdef object _rand_uint16_broadcast(np.ndarray low, np.ndarray high, object size, brng_t *state, object lock):
    """Array path for smaller integer types"""
    cdef uint16_t rng, last_rng, off, val, mask, out_val
    cdef uint32_t buf
    cdef uint16_t *out_data
    cdef uint32_t low_v, high_v
    cdef np.ndarray low_arr, high_arr, out_arr
    cdef np.npy_intp i, cnt
    cdef np.broadcast it
    cdef int buf_rem = 0


    # Array path
    low_arr = <np.ndarray>low
    high_arr = <np.ndarray>high
    if np.any(np.less(low_arr, 0)):
        raise ValueError('low is out of bounds for uint16')
    if np.any(np.greater(high_arr, 0X10000UL)):
        raise ValueError('high is out of bounds for uint16')
    if np.any(np.greater_equal(low_arr, high_arr)):
        raise ValueError('low >= high')

    low_arr = <np.ndarray>np.PyArray_FROM_OTF(low, np.NPY_UINT32, np.NPY_ALIGNED | np.NPY_FORCECAST)
    high_arr = <np.ndarray>np.PyArray_FROM_OTF(high, np.NPY_UINT32, np.NPY_ALIGNED | np.NPY_FORCECAST)

    if size is not None:
        out_arr = <np.ndarray>np.empty(size, np.uint16)
    else:
        it = np.PyArray_MultiIterNew2(low_arr, high_arr)
        out_arr = <np.ndarray>np.empty(it.shape, np.uint16)

    it = np.PyArray_MultiIterNew3(low_arr, high_arr, out_arr)
    out_data = <uint16_t *>np.PyArray_DATA(out_arr)
    cnt = np.PyArray_SIZE(out_arr)
    mask = last_rng = 0
    with lock, nogil:
        for i in range(cnt):
            low_v = (<uint32_t*>np.PyArray_MultiIter_DATA(it, 0))[0]
            high_v = (<uint32_t*>np.PyArray_MultiIter_DATA(it, 1))[0]
            rng = <uint16_t>((high_v - 1) - low_v)
            off = <uint16_t>(<uint32_t>low_v)

            if rng != last_rng:
                # Smallest bit mask >= max
                mask = <uint16_t>_gen_mask(rng)

            out_data[i] = random_buffered_bounded_uint16(state, off, rng, mask, &buf_rem, &buf)

            np.PyArray_MultiIter_NEXT(it)
    return out_arr


cdef object _rand_uint8_broadcast(np.ndarray low, np.ndarray high, object size, brng_t *state, object lock):
    """Array path for smaller integer types"""
    cdef uint8_t rng, last_rng, off, val, mask, out_val
    cdef uint32_t buf
    cdef uint8_t *out_data
    cdef uint16_t low_v, high_v
    cdef np.ndarray low_arr, high_arr, out_arr
    cdef np.npy_intp i, cnt
    cdef np.broadcast it
    cdef int buf_rem = 0


    # Array path
    low_arr = <np.ndarray>low
    high_arr = <np.ndarray>high
    if np.any(np.less(low_arr, 0)):
        raise ValueError('low is out of bounds for uint8')
    if np.any(np.greater(high_arr, 0X100UL)):
        raise ValueError('high is out of bounds for uint8')
    if np.any(np.greater_equal(low_arr, high_arr)):
        raise ValueError('low >= high')

    low_arr = <np.ndarray>np.PyArray_FROM_OTF(low, np.NPY_UINT16, np.NPY_ALIGNED | np.NPY_FORCECAST)
    high_arr = <np.ndarray>np.PyArray_FROM_OTF(high, np.NPY_UINT16, np.NPY_ALIGNED | np.NPY_FORCECAST)

    if size is not None:
        out_arr = <np.ndarray>np.empty(size, np.uint8)
    else:
        it = np.PyArray_MultiIterNew2(low_arr, high_arr)
        out_arr = <np.ndarray>np.empty(it.shape, np.uint8)

    it = np.PyArray_MultiIterNew3(low_arr, high_arr, out_arr)
    out_data = <uint8_t *>np.PyArray_DATA(out_arr)
    cnt = np.PyArray_SIZE(out_arr)
    mask = last_rng = 0
    with lock, nogil:
        for i in range(cnt):
            low_v = (<uint16_t*>np.PyArray_MultiIter_DATA(it, 0))[0]
            high_v = (<uint16_t*>np.PyArray_MultiIter_DATA(it, 1))[0]
            rng = <uint8_t>((high_v - 1) - low_v)
            off = <uint8_t>(<uint16_t>low_v)

            if rng != last_rng:
                # Smallest bit mask >= max
                mask = <uint8_t>_gen_mask(rng)

            out_data[i] = random_buffered_bounded_uint8(state, off, rng, mask, &buf_rem, &buf)

            np.PyArray_MultiIter_NEXT(it)
    return out_arr


cdef object _rand_bool_broadcast(np.ndarray low, np.ndarray high, object size, brng_t *state, object lock):
    """Array path for smaller integer types"""
    cdef bool_t rng, last_rng, off, val, mask, out_val
    cdef uint32_t buf
    cdef bool_t *out_data
    cdef uint8_t low_v, high_v
    cdef np.ndarray low_arr, high_arr, out_arr
    cdef np.npy_intp i, cnt
    cdef np.broadcast it
    cdef int buf_rem = 0


    # Array path
    low_arr = <np.ndarray>low
    high_arr = <np.ndarray>high
    if np.any(np.less(low_arr, 0)):
        raise ValueError('low is out of bounds for bool')
    if np.any(np.greater(high_arr, 0x2UL)):
        raise ValueError('high is out of bounds for bool')
    if np.any(np.greater_equal(low_arr, high_arr)):
        raise ValueError('low >= high')

    low_arr = <np.ndarray>np.PyArray_FROM_OTF(low, np.NPY_UINT8, np.NPY_ALIGNED | np.NPY_FORCECAST)
    high_arr = <np.ndarray>np.PyArray_FROM_OTF(high, np.NPY_UINT8, np.NPY_ALIGNED | np.NPY_FORCECAST)

    if size is not None:
        out_arr = <np.ndarray>np.empty(size, np.bool_)
    else:
        it = np.PyArray_MultiIterNew2(low_arr, high_arr)
        out_arr = <np.ndarray>np.empty(it.shape, np.bool_)

    it = np.PyArray_MultiIterNew3(low_arr, high_arr, out_arr)
    out_data = <bool_t *>np.PyArray_DATA(out_arr)
    cnt = np.PyArray_SIZE(out_arr)
    mask = last_rng = 0
    with lock, nogil:
        for i in range(cnt):
            low_v = (<uint8_t*>np.PyArray_MultiIter_DATA(it, 0))[0]
            high_v = (<uint8_t*>np.PyArray_MultiIter_DATA(it, 1))[0]
            rng = <bool_t>((high_v - 1) - low_v)
            off = <bool_t>(<uint8_t>low_v)

            if rng != last_rng:
                # Smallest bit mask >= max
                mask = <bool_t>_gen_mask(rng)

            out_data[i] = random_buffered_bounded_bool(state, off, rng, mask, &buf_rem, &buf)

            np.PyArray_MultiIter_NEXT(it)
    return out_arr


cdef object _rand_int32_broadcast(np.ndarray low, np.ndarray high, object size, brng_t *state, object lock):
    """Array path for smaller integer types"""
    cdef uint32_t rng, last_rng, off, val, mask, out_val
    cdef uint32_t buf
    cdef uint32_t *out_data
    cdef uint64_t low_v, high_v
    cdef np.ndarray low_arr, high_arr, out_arr
    cdef np.npy_intp i, cnt
    cdef np.broadcast it
    cdef int buf_rem = 0


    # Array path
    low_arr = <np.ndarray>low
    high_arr = <np.ndarray>high
    if np.any(np.less(low_arr, -0x80000000LL)):
        raise ValueError('low is out of bounds for int32')
    if np.any(np.greater(high_arr, 0x80000000LL)):
        raise ValueError('high is out of bounds for int32')
    if np.any(np.greater_equal(low_arr, high_arr)):
        raise ValueError('low >= high')

    low_arr = <np.ndarray>np.PyArray_FROM_OTF(low, np.NPY_INT64, np.NPY_ALIGNED | np.NPY_FORCECAST)
    high_arr = <np.ndarray>np.PyArray_FROM_OTF(high, np.NPY_INT64, np.NPY_ALIGNED | np.NPY_FORCECAST)

    if size is not None:
        out_arr = <np.ndarray>np.empty(size, np.int32)
    else:
        it = np.PyArray_MultiIterNew2(low_arr, high_arr)
        out_arr = <np.ndarray>np.empty(it.shape, np.int32)

    it = np.PyArray_MultiIterNew3(low_arr, high_arr, out_arr)
    out_data = <uint32_t *>np.PyArray_DATA(out_arr)
    cnt = np.PyArray_SIZE(out_arr)
    mask = last_rng = 0
    with lock, nogil:
        for i in range(cnt):
            low_v = (<uint64_t*>np.PyArray_MultiIter_DATA(it, 0))[0]
            high_v = (<uint64_t*>np.PyArray_MultiIter_DATA(it, 1))[0]
            rng = <uint32_t>((high_v - 1) - low_v)
            off = <uint32_t>(<uint64_t>low_v)

            if rng != last_rng:
                # Smallest bit mask >= max
                mask = <uint32_t>_gen_mask(rng)

            out_data[i] = random_buffered_bounded_uint32(state, off, rng, mask, &buf_rem, &buf)

            np.PyArray_MultiIter_NEXT(it)
    return out_arr


cdef object _rand_int16_broadcast(np.ndarray low, np.ndarray high, object size, brng_t *state, object lock):
    """Array path for smaller integer types"""
    cdef uint16_t rng, last_rng, off, val, mask, out_val
    cdef uint32_t buf
    cdef uint16_t *out_data
    cdef uint32_t low_v, high_v
    cdef np.ndarray low_arr, high_arr, out_arr
    cdef np.npy_intp i, cnt
    cdef np.broadcast it
    cdef int buf_rem = 0


    # Array path
    low_arr = <np.ndarray>low
    high_arr = <np.ndarray>high
    if np.any(np.less(low_arr, -0x8000LL)):
        raise ValueError('low is out of bounds for int16')
    if np.any(np.greater(high_arr, 0x8000LL)):
        raise ValueError('high is out of bounds for int16')
    if np.any(np.greater_equal(low_arr, high_arr)):
        raise ValueError('low >= high')

    low_arr = <np.ndarray>np.PyArray_FROM_OTF(low, np.NPY_INT32, np.NPY_ALIGNED | np.NPY_FORCECAST)
    high_arr = <np.ndarray>np.PyArray_FROM_OTF(high, np.NPY_INT32, np.NPY_ALIGNED | np.NPY_FORCECAST)

    if size is not None:
        out_arr = <np.ndarray>np.empty(size, np.int16)
    else:
        it = np.PyArray_MultiIterNew2(low_arr, high_arr)
        out_arr = <np.ndarray>np.empty(it.shape, np.int16)

    it = np.PyArray_MultiIterNew3(low_arr, high_arr, out_arr)
    out_data = <uint16_t *>np.PyArray_DATA(out_arr)
    cnt = np.PyArray_SIZE(out_arr)
    mask = last_rng = 0
    with lock, nogil:
        for i in range(cnt):
            low_v = (<uint32_t*>np.PyArray_MultiIter_DATA(it, 0))[0]
            high_v = (<uint32_t*>np.PyArray_MultiIter_DATA(it, 1))[0]
            rng = <uint16_t>((high_v - 1) - low_v)
            off = <uint16_t>(<uint32_t>low_v)

            if rng != last_rng:
                # Smallest bit mask >= max
                mask = <uint16_t>_gen_mask(rng)

            out_data[i] = random_buffered_bounded_uint16(state, off, rng, mask, &buf_rem, &buf)

            np.PyArray_MultiIter_NEXT(it)
    return out_arr


cdef object _rand_int8_broadcast(np.ndarray low, np.ndarray high, object size, brng_t *state, object lock):
    """Array path for smaller integer types"""
    cdef uint8_t rng, last_rng, off, val, mask, out_val
    cdef uint32_t buf
    cdef uint8_t *out_data
    cdef uint16_t low_v, high_v
    cdef np.ndarray low_arr, high_arr, out_arr
    cdef np.npy_intp i, cnt
    cdef np.broadcast it
    cdef int buf_rem = 0


    # Array path
    low_arr = <np.ndarray>low
    high_arr = <np.ndarray>high
    if np.any(np.less(low_arr, -0x80LL)):
        raise ValueError('low is out of bounds for int8')
    if np.any(np.greater(high_arr, 0x80LL)):
        raise ValueError('high is out of bounds for int8')
    if np.any(np.greater_equal(low_arr, high_arr)):
        raise ValueError('low >= high')

    low_arr = <np.ndarray>np.PyArray_FROM_OTF(low, np.NPY_INT16, np.NPY_ALIGNED | np.NPY_FORCECAST)
    high_arr = <np.ndarray>np.PyArray_FROM_OTF(high, np.NPY_INT16, np.NPY_ALIGNED | np.NPY_FORCECAST)

    if size is not None:
        out_arr = <np.ndarray>np.empty(size, np.int8)
    else:
        it = np.PyArray_MultiIterNew2(low_arr, high_arr)
        out_arr = <np.ndarray>np.empty(it.shape, np.int8)

    it = np.PyArray_MultiIterNew3(low_arr, high_arr, out_arr)
    out_data = <uint8_t *>np.PyArray_DATA(out_arr)
    cnt = np.PyArray_SIZE(out_arr)
    mask = last_rng = 0
    with lock, nogil:
        for i in range(cnt):
            low_v = (<uint16_t*>np.PyArray_MultiIter_DATA(it, 0))[0]
            high_v = (<uint16_t*>np.PyArray_MultiIter_DATA(it, 1))[0]
            rng = <uint8_t>((high_v - 1) - low_v)
            off = <uint8_t>(<uint16_t>low_v)

            if rng != last_rng:
                # Smallest bit mask >= max
                mask = <uint8_t>_gen_mask(rng)

            out_data[i] = random_buffered_bounded_uint8(state, off, rng, mask, &buf_rem, &buf)

            np.PyArray_MultiIter_NEXT(it)
    return out_arr



cdef object _rand_uint64_broadcast(object low, object high, object size, brng_t *state, object lock):
    """Array path for 64-bit integer types"""
    cdef np.ndarray low_arr, high_arr, out_arr, highm1_arr
    cdef np.npy_intp i, cnt, n
    cdef np.broadcast it
    cdef object closed_upper
    cdef uint64_t *out_data
    cdef uint64_t *highm1_data
    cdef uint64_t low_v, high_v
    cdef uint64_t rng, last_rng, val, mask, off, out_val

    low_arr = <np.ndarray>low
    high_arr = <np.ndarray>high

    if np.any(np.less(low_arr, 0x0ULL)):
        raise ValueError('low is out of bounds for uint64')

    highm1_arr = <np.ndarray>np.empty_like(high_arr, dtype=np.uint64)
    highm1_data = <uint64_t *>np.PyArray_DATA(highm1_arr)
    cnt = np.PyArray_SIZE(high_arr)
    flat = high_arr.flat
    for i in range(cnt):
        closed_upper = int(flat[i]) - 1
        if closed_upper > 0xFFFFFFFFFFFFFFFFULL:
            raise ValueError('high is out of bounds for uint64')
        if closed_upper < 0x0ULL:
            raise ValueError('low >= high')
        highm1_data[i] = <uint64_t>closed_upper

    if np.any(np.greater(low_arr, highm1_arr)):
        raise ValueError('low >= high')

    high_arr = highm1_arr
    low_arr = <np.ndarray>np.PyArray_FROM_OTF(low, np.NPY_UINT64, np.NPY_ALIGNED | np.NPY_FORCECAST)

    if size is not None:
        out_arr = <np.ndarray>np.empty(size, np.uint64)
    else:
        it = np.PyArray_MultiIterNew2(low_arr, high_arr)
        out_arr = <np.ndarray>np.empty(it.shape, np.uint64)

    it = np.PyArray_MultiIterNew3(low_arr, high_arr, out_arr)
    out_data = <uint64_t *>np.PyArray_DATA(out_arr)
    n = np.PyArray_SIZE(out_arr)
    mask = last_rng = 0
    with lock, nogil:
        for i in range(n):
            low_v = (<uint64_t*>np.PyArray_MultiIter_DATA(it, 0))[0]
            high_v = (<uint64_t*>np.PyArray_MultiIter_DATA(it, 1))[0]
            rng = <uint64_t>(high_v - low_v) # No -1 here since implemented above
            off = <uint64_t>(<uint64_t>low_v)

            if rng != last_rng:
                mask = _gen_mask(rng)
            out_data[i] = random_bounded_uint64(state, off, rng, mask)

            np.PyArray_MultiIter_NEXT(it)

    return out_arr

cdef object _rand_int64_broadcast(object low, object high, object size, brng_t *state, object lock):
    """Array path for 64-bit integer types"""
    cdef np.ndarray low_arr, high_arr, out_arr, highm1_arr
    cdef np.npy_intp i, cnt, n
    cdef np.broadcast it
    cdef object closed_upper
    cdef uint64_t *out_data
    cdef int64_t *highm1_data
    cdef int64_t low_v, high_v
    cdef uint64_t rng, last_rng, val, mask, off, out_val

    low_arr = <np.ndarray>low
    high_arr = <np.ndarray>high

    if np.any(np.less(low_arr, -0x8000000000000000LL)):
        raise ValueError('low is out of bounds for int64')

    highm1_arr = <np.ndarray>np.empty_like(high_arr, dtype=np.int64)
    highm1_data = <int64_t *>np.PyArray_DATA(highm1_arr)
    cnt = np.PyArray_SIZE(high_arr)
    flat = high_arr.flat
    for i in range(cnt):
        closed_upper = int(flat[i]) - 1
        if closed_upper > 0x7FFFFFFFFFFFFFFFLL:
            raise ValueError('high is out of bounds for int64')
        if closed_upper < -0x8000000000000000LL:
            raise ValueError('low >= high')
        highm1_data[i] = <int64_t>closed_upper

    if np.any(np.greater(low_arr, highm1_arr)):
        raise ValueError('low >= high')

    high_arr = highm1_arr
    low_arr = <np.ndarray>np.PyArray_FROM_OTF(low, np.NPY_INT64, np.NPY_ALIGNED | np.NPY_FORCECAST)

    if size is not None:
        out_arr = <np.ndarray>np.empty(size, np.int64)
    else:
        it = np.PyArray_MultiIterNew2(low_arr, high_arr)
        out_arr = <np.ndarray>np.empty(it.shape, np.int64)

    it = np.PyArray_MultiIterNew3(low_arr, high_arr, out_arr)
    out_data = <uint64_t *>np.PyArray_DATA(out_arr)
    n = np.PyArray_SIZE(out_arr)
    mask = last_rng = 0
    with lock, nogil:
        for i in range(n):
            low_v = (<int64_t*>np.PyArray_MultiIter_DATA(it, 0))[0]
            high_v = (<int64_t*>np.PyArray_MultiIter_DATA(it, 1))[0]
            rng = <uint64_t>(high_v - low_v) # No -1 here since implemented above
            off = <uint64_t>(<int64_t>low_v)

            if rng != last_rng:
                mask = _gen_mask(rng)
            out_data[i] = random_bounded_uint64(state, off, rng, mask)

            np.PyArray_MultiIter_NEXT(it)

    return out_arr



cdef object _rand_uint64(object low, object high, object size, brng_t *state, object lock):
    """
    _rand_uint64(low, high, size, *state, lock)

    Return random np.uint64 integers between `low` and `high`, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.uint64 type.

    Parameters
    ----------
    low : int or array-like
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int or array-like
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    state : basic random state
        State to use in the core random number generators
    lock : threading.Lock
        Lock to prevent multiple using a single generator simultaneously

    Returns
    -------
    out : python scalar or ndarray of np.uint64
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.
    """
    cdef np.ndarray out_arr, low_arr, high_arr
    cdef uint64_t rng, off, out_val
    cdef uint64_t *out_data
    cdef np.npy_intp i, n, cnt

    if size is not None:
        if (np.prod(size) == 0):
            return np.empty(size, dtype=np.uint64)

    low_arr = <np.ndarray>np.array(low, copy=False)
    high_arr = <np.ndarray>np.array(high, copy=False)
    low_ndim = np.PyArray_NDIM(low_arr)
    high_ndim = np.PyArray_NDIM(high_arr)
    if ((low_ndim == 0 or (low_ndim==1 and low_arr.size==1 and size is not None)) and
            (high_ndim == 0 or (high_ndim==1 and high_arr.size==1 and size is not None))):
        low = int(low_arr)
        high = int(high_arr)
        high -= 1

        if low < 0x0ULL:
            raise ValueError("low is out of bounds for uint64")
        if high > 0xFFFFFFFFFFFFFFFFULL:
            raise ValueError("high is out of bounds for uint64")
        if low > high:  # -1 already subtracted, closed interval
            raise ValueError("low >= high")

        rng = <uint64_t>(high - low)
        off = <uint64_t>(<uint64_t>low)
        if size is None:
            with lock:
                random_bounded_uint64_fill(state, off, rng, 1, &out_val)
            return np.uint64(<uint64_t>out_val)
        else:
            out_arr = <np.ndarray>np.empty(size, np.uint64)
            cnt = np.PyArray_SIZE(out_arr)
            out_data = <uint64_t *>np.PyArray_DATA(out_arr)
            with lock, nogil:
                random_bounded_uint64_fill(state, off, rng, cnt, out_data)
            return out_arr
    return _rand_uint64_broadcast(low_arr, high_arr, size, state, lock)

cdef object _rand_uint32(object low, object high, object size, brng_t *state, object lock):
    """
    _rand_uint32(low, high, size, *state, lock)

    Return random np.uint32 integers between `low` and `high`, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.uint32 type.

    Parameters
    ----------
    low : int or array-like
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int or array-like
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    state : basic random state
        State to use in the core random number generators
    lock : threading.Lock
        Lock to prevent multiple using a single generator simultaneously

    Returns
    -------
    out : python scalar or ndarray of np.uint32
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.
    """
    cdef np.ndarray out_arr, low_arr, high_arr
    cdef uint32_t rng, off, out_val
    cdef uint32_t *out_data
    cdef np.npy_intp i, n, cnt

    if size is not None:
        if (np.prod(size) == 0):
            return np.empty(size, dtype=np.uint32)

    low_arr = <np.ndarray>np.array(low, copy=False)
    high_arr = <np.ndarray>np.array(high, copy=False)
    low_ndim = np.PyArray_NDIM(low_arr)
    high_ndim = np.PyArray_NDIM(high_arr)
    if ((low_ndim == 0 or (low_ndim==1 and low_arr.size==1 and size is not None)) and
            (high_ndim == 0 or (high_ndim==1 and high_arr.size==1 and size is not None))):
        low = int(low_arr)
        high = int(high_arr)
        high -= 1

        if low < 0x0UL:
            raise ValueError("low is out of bounds for uint32")
        if high > 0XFFFFFFFFUL:
            raise ValueError("high is out of bounds for uint32")
        if low > high:  # -1 already subtracted, closed interval
            raise ValueError("low >= high")

        rng = <uint32_t>(high - low)
        off = <uint32_t>(<uint32_t>low)
        if size is None:
            with lock:
                random_bounded_uint32_fill(state, off, rng, 1, &out_val)
            return np.uint32(<uint32_t>out_val)
        else:
            out_arr = <np.ndarray>np.empty(size, np.uint32)
            cnt = np.PyArray_SIZE(out_arr)
            out_data = <uint32_t *>np.PyArray_DATA(out_arr)
            with lock, nogil:
                random_bounded_uint32_fill(state, off, rng, cnt, out_data)
            return out_arr
    return _rand_uint32_broadcast(low_arr, high_arr, size, state, lock)

cdef object _rand_uint16(object low, object high, object size, brng_t *state, object lock):
    """
    _rand_uint16(low, high, size, *state, lock)

    Return random np.uint16 integers between `low` and `high`, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.uint16 type.

    Parameters
    ----------
    low : int or array-like
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int or array-like
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    state : basic random state
        State to use in the core random number generators
    lock : threading.Lock
        Lock to prevent multiple using a single generator simultaneously

    Returns
    -------
    out : python scalar or ndarray of np.uint16
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.
    """
    cdef np.ndarray out_arr, low_arr, high_arr
    cdef uint16_t rng, off, out_val
    cdef uint16_t *out_data
    cdef np.npy_intp i, n, cnt

    if size is not None:
        if (np.prod(size) == 0):
            return np.empty(size, dtype=np.uint16)

    low_arr = <np.ndarray>np.array(low, copy=False)
    high_arr = <np.ndarray>np.array(high, copy=False)
    low_ndim = np.PyArray_NDIM(low_arr)
    high_ndim = np.PyArray_NDIM(high_arr)
    if ((low_ndim == 0 or (low_ndim==1 and low_arr.size==1 and size is not None)) and
            (high_ndim == 0 or (high_ndim==1 and high_arr.size==1 and size is not None))):
        low = int(low_arr)
        high = int(high_arr)
        high -= 1

        if low < 0x0UL:
            raise ValueError("low is out of bounds for uint16")
        if high > 0XFFFFUL:
            raise ValueError("high is out of bounds for uint16")
        if low > high:  # -1 already subtracted, closed interval
            raise ValueError("low >= high")

        rng = <uint16_t>(high - low)
        off = <uint16_t>(<uint16_t>low)
        if size is None:
            with lock:
                random_bounded_uint16_fill(state, off, rng, 1, &out_val)
            return np.uint16(<uint16_t>out_val)
        else:
            out_arr = <np.ndarray>np.empty(size, np.uint16)
            cnt = np.PyArray_SIZE(out_arr)
            out_data = <uint16_t *>np.PyArray_DATA(out_arr)
            with lock, nogil:
                random_bounded_uint16_fill(state, off, rng, cnt, out_data)
            return out_arr
    return _rand_uint16_broadcast(low_arr, high_arr, size, state, lock)

cdef object _rand_uint8(object low, object high, object size, brng_t *state, object lock):
    """
    _rand_uint8(low, high, size, *state, lock)

    Return random np.uint8 integers between `low` and `high`, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.uint8 type.

    Parameters
    ----------
    low : int or array-like
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int or array-like
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    state : basic random state
        State to use in the core random number generators
    lock : threading.Lock
        Lock to prevent multiple using a single generator simultaneously

    Returns
    -------
    out : python scalar or ndarray of np.uint8
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.
    """
    cdef np.ndarray out_arr, low_arr, high_arr
    cdef uint8_t rng, off, out_val
    cdef uint8_t *out_data
    cdef np.npy_intp i, n, cnt

    if size is not None:
        if (np.prod(size) == 0):
            return np.empty(size, dtype=np.uint8)

    low_arr = <np.ndarray>np.array(low, copy=False)
    high_arr = <np.ndarray>np.array(high, copy=False)
    low_ndim = np.PyArray_NDIM(low_arr)
    high_ndim = np.PyArray_NDIM(high_arr)
    if ((low_ndim == 0 or (low_ndim==1 and low_arr.size==1 and size is not None)) and
            (high_ndim == 0 or (high_ndim==1 and high_arr.size==1 and size is not None))):
        low = int(low_arr)
        high = int(high_arr)
        high -= 1

        if low < 0x0UL:
            raise ValueError("low is out of bounds for uint8")
        if high > 0XFFUL:
            raise ValueError("high is out of bounds for uint8")
        if low > high:  # -1 already subtracted, closed interval
            raise ValueError("low >= high")

        rng = <uint8_t>(high - low)
        off = <uint8_t>(<uint8_t>low)
        if size is None:
            with lock:
                random_bounded_uint8_fill(state, off, rng, 1, &out_val)
            return np.uint8(<uint8_t>out_val)
        else:
            out_arr = <np.ndarray>np.empty(size, np.uint8)
            cnt = np.PyArray_SIZE(out_arr)
            out_data = <uint8_t *>np.PyArray_DATA(out_arr)
            with lock, nogil:
                random_bounded_uint8_fill(state, off, rng, cnt, out_data)
            return out_arr
    return _rand_uint8_broadcast(low_arr, high_arr, size, state, lock)

cdef object _rand_bool(object low, object high, object size, brng_t *state, object lock):
    """
    _rand_bool(low, high, size, *state, lock)

    Return random np.bool integers between `low` and `high`, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.bool type.

    Parameters
    ----------
    low : int or array-like
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int or array-like
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    state : basic random state
        State to use in the core random number generators
    lock : threading.Lock
        Lock to prevent multiple using a single generator simultaneously

    Returns
    -------
    out : python scalar or ndarray of np.bool
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.
    """
    cdef np.ndarray out_arr, low_arr, high_arr
    cdef bool_t rng, off, out_val
    cdef bool_t *out_data
    cdef np.npy_intp i, n, cnt

    if size is not None:
        if (np.prod(size) == 0):
            return np.empty(size, dtype=np.bool)

    low_arr = <np.ndarray>np.array(low, copy=False)
    high_arr = <np.ndarray>np.array(high, copy=False)
    low_ndim = np.PyArray_NDIM(low_arr)
    high_ndim = np.PyArray_NDIM(high_arr)
    if ((low_ndim == 0 or (low_ndim==1 and low_arr.size==1 and size is not None)) and
            (high_ndim == 0 or (high_ndim==1 and high_arr.size==1 and size is not None))):
        low = int(low_arr)
        high = int(high_arr)
        high -= 1

        if low < 0x0UL:
            raise ValueError("low is out of bounds for bool")
        if high > 0x1UL:
            raise ValueError("high is out of bounds for bool")
        if low > high:  # -1 already subtracted, closed interval
            raise ValueError("low >= high")

        rng = <bool_t>(high - low)
        off = <bool_t>(<bool_t>low)
        if size is None:
            with lock:
                random_bounded_bool_fill(state, off, rng, 1, &out_val)
            return np.bool_(<bool_t>out_val)
        else:
            out_arr = <np.ndarray>np.empty(size, np.bool)
            cnt = np.PyArray_SIZE(out_arr)
            out_data = <bool_t *>np.PyArray_DATA(out_arr)
            with lock, nogil:
                random_bounded_bool_fill(state, off, rng, cnt, out_data)
            return out_arr
    return _rand_bool_broadcast(low_arr, high_arr, size, state, lock)

cdef object _rand_int64(object low, object high, object size, brng_t *state, object lock):
    """
    _rand_int64(low, high, size, *state, lock)

    Return random np.int64 integers between `low` and `high`, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.int64 type.

    Parameters
    ----------
    low : int or array-like
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int or array-like
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    state : basic random state
        State to use in the core random number generators
    lock : threading.Lock
        Lock to prevent multiple using a single generator simultaneously

    Returns
    -------
    out : python scalar or ndarray of np.int64
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.
    """
    cdef np.ndarray out_arr, low_arr, high_arr
    cdef uint64_t rng, off, out_val
    cdef uint64_t *out_data
    cdef np.npy_intp i, n, cnt

    if size is not None:
        if (np.prod(size) == 0):
            return np.empty(size, dtype=np.int64)

    low_arr = <np.ndarray>np.array(low, copy=False)
    high_arr = <np.ndarray>np.array(high, copy=False)
    low_ndim = np.PyArray_NDIM(low_arr)
    high_ndim = np.PyArray_NDIM(high_arr)
    if ((low_ndim == 0 or (low_ndim==1 and low_arr.size==1 and size is not None)) and
            (high_ndim == 0 or (high_ndim==1 and high_arr.size==1 and size is not None))):
        low = int(low_arr)
        high = int(high_arr)
        high -= 1

        if low < -0x8000000000000000LL:
            raise ValueError("low is out of bounds for int64")
        if high > 0x7FFFFFFFFFFFFFFFL:
            raise ValueError("high is out of bounds for int64")
        if low > high:  # -1 already subtracted, closed interval
            raise ValueError("low >= high")

        rng = <uint64_t>(high - low)
        off = <uint64_t>(<int64_t>low)
        if size is None:
            with lock:
                random_bounded_uint64_fill(state, off, rng, 1, &out_val)
            return np.int64(<int64_t>out_val)
        else:
            out_arr = <np.ndarray>np.empty(size, np.int64)
            cnt = np.PyArray_SIZE(out_arr)
            out_data = <uint64_t *>np.PyArray_DATA(out_arr)
            with lock, nogil:
                random_bounded_uint64_fill(state, off, rng, cnt, out_data)
            return out_arr
    return _rand_int64_broadcast(low_arr, high_arr, size, state, lock)

cdef object _rand_int32(object low, object high, object size, brng_t *state, object lock):
    """
    _rand_int32(low, high, size, *state, lock)

    Return random np.int32 integers between `low` and `high`, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.int32 type.

    Parameters
    ----------
    low : int or array-like
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int or array-like
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    state : basic random state
        State to use in the core random number generators
    lock : threading.Lock
        Lock to prevent multiple using a single generator simultaneously

    Returns
    -------
    out : python scalar or ndarray of np.int32
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.
    """
    cdef np.ndarray out_arr, low_arr, high_arr
    cdef uint32_t rng, off, out_val
    cdef uint32_t *out_data
    cdef np.npy_intp i, n, cnt

    if size is not None:
        if (np.prod(size) == 0):
            return np.empty(size, dtype=np.int32)

    low_arr = <np.ndarray>np.array(low, copy=False)
    high_arr = <np.ndarray>np.array(high, copy=False)
    low_ndim = np.PyArray_NDIM(low_arr)
    high_ndim = np.PyArray_NDIM(high_arr)
    if ((low_ndim == 0 or (low_ndim==1 and low_arr.size==1 and size is not None)) and
            (high_ndim == 0 or (high_ndim==1 and high_arr.size==1 and size is not None))):
        low = int(low_arr)
        high = int(high_arr)
        high -= 1

        if low < -0x80000000L:
            raise ValueError("low is out of bounds for int32")
        if high > 0x7FFFFFFFL:
            raise ValueError("high is out of bounds for int32")
        if low > high:  # -1 already subtracted, closed interval
            raise ValueError("low >= high")

        rng = <uint32_t>(high - low)
        off = <uint32_t>(<int32_t>low)
        if size is None:
            with lock:
                random_bounded_uint32_fill(state, off, rng, 1, &out_val)
            return np.int32(<int32_t>out_val)
        else:
            out_arr = <np.ndarray>np.empty(size, np.int32)
            cnt = np.PyArray_SIZE(out_arr)
            out_data = <uint32_t *>np.PyArray_DATA(out_arr)
            with lock, nogil:
                random_bounded_uint32_fill(state, off, rng, cnt, out_data)
            return out_arr
    return _rand_int32_broadcast(low_arr, high_arr, size, state, lock)

cdef object _rand_int16(object low, object high, object size, brng_t *state, object lock):
    """
    _rand_int16(low, high, size, *state, lock)

    Return random np.int16 integers between `low` and `high`, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.int16 type.

    Parameters
    ----------
    low : int or array-like
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int or array-like
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    state : basic random state
        State to use in the core random number generators
    lock : threading.Lock
        Lock to prevent multiple using a single generator simultaneously

    Returns
    -------
    out : python scalar or ndarray of np.int16
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.
    """
    cdef np.ndarray out_arr, low_arr, high_arr
    cdef uint16_t rng, off, out_val
    cdef uint16_t *out_data
    cdef np.npy_intp i, n, cnt

    if size is not None:
        if (np.prod(size) == 0):
            return np.empty(size, dtype=np.int16)

    low_arr = <np.ndarray>np.array(low, copy=False)
    high_arr = <np.ndarray>np.array(high, copy=False)
    low_ndim = np.PyArray_NDIM(low_arr)
    high_ndim = np.PyArray_NDIM(high_arr)
    if ((low_ndim == 0 or (low_ndim==1 and low_arr.size==1 and size is not None)) and
            (high_ndim == 0 or (high_ndim==1 and high_arr.size==1 and size is not None))):
        low = int(low_arr)
        high = int(high_arr)
        high -= 1

        if low < -0x8000L:
            raise ValueError("low is out of bounds for int16")
        if high > 0x7FFFL:
            raise ValueError("high is out of bounds for int16")
        if low > high:  # -1 already subtracted, closed interval
            raise ValueError("low >= high")

        rng = <uint16_t>(high - low)
        off = <uint16_t>(<int16_t>low)
        if size is None:
            with lock:
                random_bounded_uint16_fill(state, off, rng, 1, &out_val)
            return np.int16(<int16_t>out_val)
        else:
            out_arr = <np.ndarray>np.empty(size, np.int16)
            cnt = np.PyArray_SIZE(out_arr)
            out_data = <uint16_t *>np.PyArray_DATA(out_arr)
            with lock, nogil:
                random_bounded_uint16_fill(state, off, rng, cnt, out_data)
            return out_arr
    return _rand_int16_broadcast(low_arr, high_arr, size, state, lock)

cdef object _rand_int8(object low, object high, object size, brng_t *state, object lock):
    """
    _rand_int8(low, high, size, *state, lock)

    Return random np.int8 integers between `low` and `high`, inclusive.

    Return random integers from the "discrete uniform" distribution in the
    closed interval [`low`, `high`).  If `high` is None (the default),
    then results are from [0, `low`). On entry the arguments are presumed
    to have been validated for size and order for the np.int8 type.

    Parameters
    ----------
    low : int or array-like
        Lowest (signed) integer to be drawn from the distribution (unless
        ``high=None``, in which case this parameter is the *highest* such
        integer).
    high : int or array-like
        If provided, the largest (signed) integer to be drawn from the
        distribution (see above for behavior if ``high=None``).
    size : int or tuple of ints
        Output shape.  If the given shape is, e.g., ``(m, n, k)``, then
        ``m * n * k`` samples are drawn.  Default is None, in which case a
        single value is returned.
    state : basic random state
        State to use in the core random number generators
    lock : threading.Lock
        Lock to prevent multiple using a single generator simultaneously

    Returns
    -------
    out : python scalar or ndarray of np.int8
          `size`-shaped array of random integers from the appropriate
          distribution, or a single such random int if `size` not provided.
    """
    cdef np.ndarray out_arr, low_arr, high_arr
    cdef uint8_t rng, off, out_val
    cdef uint8_t *out_data
    cdef np.npy_intp i, n, cnt

    if size is not None:
        if (np.prod(size) == 0):
            return np.empty(size, dtype=np.int8)

    low_arr = <np.ndarray>np.array(low, copy=False)
    high_arr = <np.ndarray>np.array(high, copy=False)
    low_ndim = np.PyArray_NDIM(low_arr)
    high_ndim = np.PyArray_NDIM(high_arr)
    if ((low_ndim == 0 or (low_ndim==1 and low_arr.size==1 and size is not None)) and
            (high_ndim == 0 or (high_ndim==1 and high_arr.size==1 and size is not None))):
        low = int(low_arr)
        high = int(high_arr)
        high -= 1

        if low < -0x80L:
            raise ValueError("low is out of bounds for int8")
        if high > 0x7FL:
            raise ValueError("high is out of bounds for int8")
        if low > high:  # -1 already subtracted, closed interval
            raise ValueError("low >= high")

        rng = <uint8_t>(high - low)
        off = <uint8_t>(<int8_t>low)
        if size is None:
            with lock:
                random_bounded_uint8_fill(state, off, rng, 1, &out_val)
            return np.int8(<int8_t>out_val)
        else:
            out_arr = <np.ndarray>np.empty(size, np.int8)
            cnt = np.PyArray_SIZE(out_arr)
            out_data = <uint8_t *>np.PyArray_DATA(out_arr)
            with lock, nogil:
                random_bounded_uint8_fill(state, off, rng, cnt, out_data)
            return out_arr
    return _rand_int8_broadcast(low_arr, high_arr, size, state, lock)
