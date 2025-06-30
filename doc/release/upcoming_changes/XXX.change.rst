`np.linspace` on descending integers now use ceil
-------------------------------------------------
When using a `int` dtype in `numpy.linspace` on a descending interval,
previously float values would be unconditionally rounded towards `-inf`.
Now `numpy.ceil` is used instead, which rounds toward ``+inf``. This changes
the results for descending intervals. For example, the following would
previously give::

    >>> np.linspace(3, -2, 10, endpoint=False, dtype=int)
    array([ 3,  2,  2,  1,  1,  0,  0, -1, -1, -2])

and now results in::

    >>> np.linspace(3, -2, 10, endpoint=False, dtype=int)
    array([ 3,  3,  2,  2,  1,  1,  0,  0, -1, -1])
