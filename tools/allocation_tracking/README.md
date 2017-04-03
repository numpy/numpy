Example for using the `PyDataMem_SetEventHook` to track allocations inside numpy.

`alloc_hook.pyx` implements a hook in Cython that calls back into a python
function. `track_allocations.py` uses it for a simple listing of allocations.
It can be built with the `setup.py` file in this folder.

Note that since Python 3.6 the builtin tracemalloc module can be used to
track allocations inside numpy.
Numpy places its CPU memory allocations into the `np.lib.tracemalloc_domain`
domain.
See https://docs.python.org/3/library/tracemalloc.html.
