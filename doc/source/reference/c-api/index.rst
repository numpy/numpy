.. _c-api:

###########
NumPy C-API
###########

.. sectionauthor:: Travis E. Oliphant

|    Beware of the man who won't be bothered with details.
|    --- *William Feather, Sr.*

|    The truth is out there.
|    --- *Chris Carter, The X Files*


NumPy provides a C-API to enable users to extend the system and get
access to the array object for use in other routines. The best way to
truly understand the C-API is to read the source code. If you are
unfamiliar with (C) source code, however, this can be a daunting
experience at first. Be assured that the task becomes easier with
practice, and you may be surprised at how simple the C-code can be to
understand. Even if you don't think you can write C-code from scratch,
it is much easier to understand and modify already-written source code
than create it *de novo*.

Python extensions are especially straightforward to understand because
they all have a very similar structure. Admittedly, NumPy is not a
trivial extension to Python, and may take a little more snooping to
grasp. This is especially true because of the code-generation
techniques, which simplify maintenance of very similar code, but can
make the code a little less readable to beginners. Still, with a
little persistence, the code can be opened to your understanding. It
is my hope, that this guide to the C-API can assist in the process of
becoming familiar with the compiled-level work that can be done with
NumPy in order to squeeze that last bit of necessary speed out of your
code.

.. _cython-c-api:

Using the C-API from Cython
===========================

The NumPy C API can be used from `Cython <https://cython.readthedocs.io/>`__.
Cython translates Python or Cython
source into C code, which is then compiled into a native Python extension
module. The extension still runs as part of Python rather than becoming an
independent C program. NumPy provides Cython declaration files (``.pxd`` files),
including `numpy/__init__.cython-30.pxd
<https://github.com/numpy/numpy/blob/main/numpy/__init__.cython-30.pxd>`__, that
give Cython access to NumPy C-API types and functions. These declarations require
Cython 3.0 or newer.

In a Cython source file, ``cimport numpy`` loads these declarations at compile
time, while a regular ``import numpy`` imports the Python package at runtime.
The regular import is only needed when the extension also uses Python-level
NumPy functions, such as ``numpy.asarray`` or ``numpy.empty``. The following
example uses NumPy's C API to expose memory allocated in C as a NumPy array:

.. code-block:: cython

   from libc.stdlib cimport free, malloc

   cimport numpy as cnp


   cnp.import_array()


   cdef class DoubleBuffer:
       cdef double *data
       cdef cnp.npy_intp size

       def __cinit__(self, cnp.npy_intp size):
           if size <= 0:
               raise ValueError("size must be positive")

           self.data = <double *>malloc(size * sizeof(double))
           if self.data == NULL:
               raise MemoryError()
           self.size = size

       def __dealloc__(self):
           free(self.data)

       def as_array(self):
           cdef cnp.ndarray arr = cnp.PyArray_SimpleNewFromData(
               1, &self.size, cnp.NPY_DOUBLE, self.data
           )
           cnp.set_array_base(arr, self)
           return arr

The ``DoubleBuffer`` class owns the memory allocated by ``malloc``.
``cnp.PyArray_SimpleNewFromData`` creates a one-dimensional NumPy array that
uses this memory without copying it. Since NumPy did not allocate the memory,
``cnp.set_array_base`` makes the ``DoubleBuffer`` object the array's base and
keeps it alive while the array is using its memory.

Here, ``cnp.npy_intp``, ``cnp.ndarray``, ``cnp.NPY_DOUBLE``,
``cnp.PyArray_SimpleNewFromData``, and ``cnp.set_array_base`` come from the
declarations loaded by ``cimport``. The example does not require a regular
NumPy import because it does not use Python-level NumPy functions. Calling
``cnp.import_array()`` initializes the NumPy C-API when the extension module is
imported.

.. note::

   For efficient access to array elements, typed memoryviews are generally
   preferred when a NumPy-specific C-API operation is not needed. Typed
   memoryviews use Python's buffer protocol and therefore work with NumPy
   arrays and other compatible buffer providers. They do not require
   ``cimport numpy`` unless the code also uses NumPy-specific declarations.
   See `Cython's typed memoryview documentation
   <https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html>`__.

For an example of using Cython to create a NumPy ufunc, see the `Cython section
of NumPy's ufunc tutorial
<https://numpy.org/devdocs/user/c-info.ufunc-tutorial.html#cython>`__.

For a longer introduction to building Cython extensions that use NumPy, see
`Cython's Working with NumPy tutorial
<https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html>`__.

.. currentmodule:: numpy-c-api

.. toctree::
   :maxdepth: 2

   types-and-structures
   config
   dtype
   array
   iterator
   ufunc
   generalized-ufuncs
   strings
   coremath
   datetimes
   deprecations
   data_memory
