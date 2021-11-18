Note that since Python 3.6 the builtin tracemalloc module can be used to
track allocations inside numpy.
Numpy places its CPU memory allocations into the `np.lib.tracemalloc_domain`
domain.
See https://docs.python.org/3/library/tracemalloc.html.

The tool that used to be here has been deprecated.
