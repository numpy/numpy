Putting the Inner Loop in Cython
================================

Those who want really good performance out of their low level operations
should strongly consider directly using the iteration API provided
in C, but for those who are not comfortable with C or C++, Cython
is a good middle ground with reasonable performance tradeoffs. For
the :class:`~numpy.nditer` object, this means letting the iterator take care
of broadcasting, dtype conversion, and buffering, while giving the inner
loop to Cython.

For our example, we'll create a sum of squares function. To start,
let's implement this function in straightforward Python. We want to
support an 'axis' parameter similar to the numpy :func:`sum` function,
so we will need to construct a list for the `op_axes` parameter.
Here's how this looks.

.. admonition:: Example

    >>> def axis_to_axeslist(axis, ndim):
    ...     if axis is None:
    ...         return [-1] * ndim
    ...     else:
    ...         if type(axis) is not tuple:
    ...             axis = (axis,)
    ...         axeslist = [1] * ndim
    ...         for i in axis:
    ...             axeslist[i] = -1
    ...         ax = 0
    ...         for i in range(ndim):
    ...             if axeslist[i] != -1:
    ...                 axeslist[i] = ax
    ...                 ax += 1
    ...         return axeslist
    ...
    >>> def sum_squares_py(arr, axis=None, out=None):
    ...     axeslist = axis_to_axeslist(axis, arr.ndim)
    ...     it = np.nditer([arr, out], flags=['reduce_ok',
    ...                                       'buffered', 'delay_bufalloc'],
    ...                 op_flags=[['readonly'], ['readwrite', 'allocate']],
    ...                 op_axes=[None, axeslist],
    ...                 op_dtypes=['float64', 'float64'])
    ...     with it:
    ...         it.operands[1][...] = 0
    ...         it.reset()
    ...         for x, y in it:
    ...             y[...] += x*x
    ...         return it.operands[1]
    ...
    >>> a = np.arange(6).reshape(2,3)
    >>> sum_squares_py(a)
    array(55.)
    >>> sum_squares_py(a, axis=-1)
    array([  5.,  50.])

To Cython-ize this function, we replace the inner loop (y[...] += x*x) with
Cython code that's specialized for the float64 dtype. With the
'external_loop' flag enabled, the arrays provided to the inner loop will
always be one-dimensional, so very little checking needs to be done.

Here's the listing of sum_squares.pyx::

    import numpy as np
    cimport numpy as np
    cimport cython

    def axis_to_axeslist(axis, ndim):
        if axis is None:
            return [-1] * ndim
        else:
            if type(axis) is not tuple:
                axis = (axis,)
            axeslist = [1] * ndim
            for i in axis:
                axeslist[i] = -1
            ax = 0
            for i in range(ndim):
                if axeslist[i] != -1:
                    axeslist[i] = ax
                    ax += 1
            return axeslist

    @cython.boundscheck(False)
    def sum_squares_cy(arr, axis=None, out=None):
        cdef np.ndarray[double] x
        cdef np.ndarray[double] y
        cdef int size
        cdef double value

        axeslist = axis_to_axeslist(axis, arr.ndim)
        it = np.nditer([arr, out], flags=['reduce_ok', 'external_loop',
                                          'buffered', 'delay_bufalloc'],
                    op_flags=[['readonly'], ['readwrite', 'allocate']],
                    op_axes=[None, axeslist],
                    op_dtypes=['float64', 'float64'])
        with it:
            it.operands[1][...] = 0
            it.reset()
            for xarr, yarr in it:
                x = xarr
                y = yarr
                size = x.shape[0]
                for i in range(size):
                   value = x[i]
                   y[i] = y[i] + value * value
            return it.operands[1]

On this machine, building the .pyx file into a module looked like the
following, but you may have to find some Cython tutorials to tell you
the specifics for your system configuration.::

    $ cython sum_squares.pyx
    $ gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -I/usr/include/python2.7 -fno-strict-aliasing -o sum_squares.so sum_squares.c

Running this from the Python interpreter produces the same answers
as our native Python/NumPy code did.

.. admonition:: Example

    >>> from sum_squares import sum_squares_cy #doctest: +SKIP
    >>> a = np.arange(6).reshape(2,3)
    >>> sum_squares_cy(a) #doctest: +SKIP
    array(55.0)
    >>> sum_squares_cy(a, axis=-1) #doctest: +SKIP
    array([  5.,  50.])

Doing a little timing in IPython shows that the reduced overhead and
memory allocation of the Cython inner loop is providing a very nice
speedup over both the straightforward Python code and an expression
using NumPy's built-in sum function.::

    >>> a = np.random.rand(1000,1000)

    >>> timeit sum_squares_py(a, axis=-1)
    10 loops, best of 3: 37.1 ms per loop

    >>> timeit np.sum(a*a, axis=-1)
    10 loops, best of 3: 20.9 ms per loop

    >>> timeit sum_squares_cy(a, axis=-1)
    100 loops, best of 3: 11.8 ms per loop

    >>> np.all(sum_squares_cy(a, axis=-1) == np.sum(a*a, axis=-1))
    True

    >>> np.all(sum_squares_py(a, axis=-1) == np.sum(a*a, axis=-1))
    True
