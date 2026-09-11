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

The NumPy C API can be used from Cython. Cython translates Python or Cython
source into C code, which is then compiled into a native Python extension
module. The extension still runs as part of Python rather than becoming an
independent C program. NumPy provides Cython declaration files (``.pxd`` files)
that give Cython access to NumPy C-API types and functions. These declarations
require Cython 3.0 or newer.

In a Cython source file, ``cimport numpy`` loads these declarations at compile
time, while a regular ``import numpy`` imports the Python package at runtime.
The regular import is only needed when the extension also uses Python-level
NumPy functions, such as ``numpy.asarray`` or ``numpy.empty``. For example:

.. code-block:: cython

   import numpy as np
   cimport numpy as cnp

   cnp.import_array()

   def ensure_2d(obj):
       cdef cnp.ndarray arr = np.asarray(obj)
       if cnp.PyArray_NDIM(arr) != 2:
           raise ValueError("expected a two-dimensional array")
       return arr

Here, ``np.asarray`` is looked up through NumPy's Python API at runtime, while
``cnp.ndarray`` and ``cnp.PyArray_NDIM`` come from the declarations loaded by
``cimport``. Calling ``cnp.import_array()`` initializes the NumPy C-API when
the extension module is imported. Cython 3 can add this call automatically
when it is needed, but explicitly calling it is recommended.

.. note::

   For efficient access to array elements, typed memoryviews are generally
   preferred over the older NumPy-specific ``cnp.ndarray[...]`` buffer syntax.
   Typed memoryviews use Python's buffer protocol and therefore work with
   NumPy arrays and other compatible buffer providers. They do not require
   ``cimport numpy`` unless the code also uses NumPy-specific declarations.
   See `Cython's typed memoryview documentation
   <https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html>`__.

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
