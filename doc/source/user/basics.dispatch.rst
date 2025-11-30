.. _basics.dispatch:

*******************************
Writing custom array containers
*******************************

Numpy's dispatch mechanism, introduced in numpy version v1.16 is the
recommended approach for writing custom N-dimensional array containers that are
compatible with the numpy API and provide custom implementations of numpy
functionality. Applications include `dask <https://docs.dask.org/en/stable/>`_
arrays, an N-dimensional array distributed across multiple nodes, and `cupy
<https://docs-cupy.chainer.org/en/stable/>`_ arrays, an N-dimensional array on
a GPU.

For comprehensive documentation on writing custom array containers, please see:

- :ref:`Interoperability with NumPy <basics.interoperability>` - the main guide
  covering ``__array_ufunc__`` and ``__array_function__`` protocols
- :ref:`Special attributes and methods <special-attributes-and-methods>` - see
  ``class.__array__()`` for documentation and example implementing the
  ``__array__()`` method

Numpy provides some utilities to aid testing of custom array containers that
implement the ``__array_ufunc__`` and ``__array_function__`` protocols in the
``numpy.testing.overrides`` namespace.

To check if a Numpy function can be overridden via ``__array_ufunc__``, you can
use :func:`~numpy.testing.overrides.allows_array_ufunc_override`:

>>> from numpy.testing.overrides import allows_array_ufunc_override
>>> allows_array_ufunc_override(np.add)
True

Similarly, you can check if a function can be overridden via
``__array_function__`` using
:func:`~numpy.testing.overrides.allows_array_function_override`.

Lists of every overridable function in the Numpy API are also available via
:func:`~numpy.testing.overrides.get_overridable_numpy_array_functions` for
functions that support the ``__array_function__`` protocol and
:func:`~numpy.testing.overrides.get_overridable_numpy_ufuncs` for functions that
support the ``__array_ufunc__`` protocol. Both functions return sets of
functions that are present in the Numpy public API. User-defined ufuncs or
ufuncs defined in other libraries that depend on Numpy are not present in
these sets.

Refer to the `dask source code <https://github.com/dask/dask>`_ and
`cupy source code <https://github.com/cupy/cupy>`_  for more fully-worked
examples of custom array containers.

See also :doc:`NEP 18<neps:nep-0018-array-function-protocol>`.
