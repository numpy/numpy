# -*- encoding: utf-8 -*-

"""
shm.py

Shared memory array implementation for numpy which delegates all the nasty stuff
to multiprocessing.sharedctypes.

Copyright (c) 2010, David Baddeley
Copyright (c) 2010-2012, Chris Lee-Messer
Copyright (c) 2012-2016, Matěj Týč
All rights reserved.
"""

# Licensced under the BSD liscence ...
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# Neither the name of the <ORGANIZATION> nor the names of its contributors may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
from multiprocessing import sharedctypes
from numpy import ctypeslib


class shmarray(np.ndarray):
    """
    subclass of ndarray with overridden pickling functions which record
    dtype, shape etc...
    Defer pickling of the underlying data to the original data source.

    Doesn't actually handle allocation of the shared memory - this is done in
    empty, and zeros, ones, (or create_copy) are the functions
    which should be used for creating a new shared memory array.
    """
    def __new__(cls, shape, dtype=float, buffer=None,
                offset=0, strides=None, order=None):

        assert buffer is not None, \
            "You have omitted the essential 'buffer' kwarg"

        # some magic (copied from np.ctypeslib) to make sure the ctypes array
        # has the array interface
        tp = type(buffer)
        try:
            tp.__array_interface__
        except AttributeError:
            ctypeslib.prep_array(tp)

        obj = np.ndarray.__new__(cls, shape, dtype, buffer,
                                 offset, strides, order)

        # keep track of the underlying storage
        # this may not be strictly necessary
        # as the same info should be stored in .base
        obj.ctypesArray = buffer

        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.ctypesArray = getattr(obj, 'ctypesArray', None)


def empty(shape, dtype='d', alignment=32):
    """
    Create an uninitialised shared array.

    Warning
    -------
    Avoid object arrays, as these will almost certainly break as the objects
    themselves won't be stored in shared memory, only the pointers

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array
    dtype : data-type, optional
        Desired output data-type
    alignment : int, optional
        The array memory alignment (in bytes)

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data of the given shape and dtype.
    """
    shape = np.atleast_1d(shape).astype('i')
    dtype = np.dtype(dtype)

    # we're going to use a flat ctypes array
    N = np.prod(shape) + alignment
    # The upper bound of size we want to allocate to be certain
    #  that we can take an aligned array of the right size from it.
    N_bytes_big = N * dtype.itemsize
    # The final (= right) size of the array
    N_bytes_right = np.prod(shape) * dtype.itemsize

    # what.
    dtype_raw = 'b'

    # We create the big array first
    # N_bytes_right has to be exactly int (and not e.g. np.int64)
    buf = sharedctypes.RawArray(dtype_raw, int(N_bytes_big))

    sa = shmarray((N_bytes_big,), dtype_raw, buf)

    # We pick the first index of the new array that is aligned
    # If the address of the first element is 1 and we want 8-alignment, the
    #  first aligned index of the array is going to be 7 == -1 % 8
    start_index = -sa.ctypes.data % alignment
    end_index = start_index + N_bytes_right
    # Finally, we take the (aligned) subarray and reshape it.
    sa = sa[start_index:end_index].view(dtype).reshape(shape)

    return sa


def zeros(shape, dtype='d'):
    """
    Create an shared array initialised to zeros.

    Warning
    -------
    Avoid object arrays, as these will almost certainly break as the objects
    themselves won't be stored in shared memory, only the pointers

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array
    dtype : data-type, optional
        Desired output data-type
    alignment : int, optional
        The array memory alignment (in bytes)

    Returns
    -------
    out : ndarray
        Array of (arbitrary) data initialized to zeros
        of the given shape and dtype.
    """
    sa = empty(shape, dtype=dtype)

    # contrary to the documentation, sharedctypes.RawArray does
    # NOT always return an array which is initialised to zero - do it ourselves
    # http://code.google.com/p/python-multiprocessing/issues/detail?id=25
    sa[:] = np.zeros(1, dtype)
    return sa


def ones(shape, dtype='d'):
    """
    Create an shared array initialised to ones.

    Warning
    -------
    Avoid object arrays, as these will almost certainly break as the objects
    themselves won't be stored in shared memory, only the pointers

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array
    dtype : data-type, optional
        Desired output data-type
    alignment : int, optional
        The array memory alignment (in bytes)

    Returns
    -------
    out : ndarray
        Array of (arbitrary) data initialized to ones
        of the given shape and dtype.
    """
    sa = empty(shape, dtype=dtype)

    sa[:] = np.ones(1, dtype)
    return sa


def copy(a, alignment=32):
    """
    Create a a shared copy of an array

    Warning
    -------
    Avoid copying object arrays, as these will almost certainly break as
    the objects themselves won't be stored in shared memory, only the pointers

    Parameters
    ----------
    a : array_like
        Input data.
    alignment : int, optional
        The array memory alignment (in bytes)

    Returns
    -------
    out : ndarray
        Shared array interpretation of `a`.
    """
    # create an empty array
    b = empty(a.shape, a.dtype, alignment)

    # copy contents across
    b[:] = a[:]

    return b
