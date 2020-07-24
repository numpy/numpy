"""
============================
Typing (:mod:`numpy.typing`)
============================

.. warning::

  Some of the types in this module rely on features only present in
  the standard library in Python 3.8 and greater. If you want to use
  these types in earlier versions of Python, you should install the
  typing-extensions_ package.

Large parts of the NumPy API have PEP-484-style type annotations. In
addition, the following type aliases are available for users.

- ``typing.ArrayLike``: objects that can be converted to arrays
- ``typing.DtypeLike``: objects that can be converted to dtypes

Roughly speaking, ``typing.ArrayLike`` is "objects that can be used as
inputs to ``np.array``" and ``typing.DtypeLike`` is "objects that can
be used as inputs to ``np.dtype``".

.. _typing-extensions: https://pypi.org/project/typing-extensions/

Differences from the runtime NumPy API
--------------------------------------

NumPy is very flexible. Trying to describe the full range of
possibilities statically would result in types that are not very
helpful. For that reason, the typed NumPy API is often stricter than
the runtime NumPy API. This section describes some notable
differences.

ArrayLike
~~~~~~~~~

The ``ArrayLike`` type tries to avoid creating object arrays. For
example,

.. code-block:: python

    >>> np.array(x**2 for x in range(10))
    array(<generator object <genexpr> at 0x10c004cd0>, dtype=object)

is valid NumPy code which will create a 0-dimensional object
array. Type checkers will complain about the above example when using
the NumPy types however. If you really intended to do the above, then
you can either use a ``# type: ignore`` comment:

.. code-block:: python

    >>> np.array(x**2 for x in range(10))  # type: ignore

or explicitly type the array like object as ``Any``:

.. code-block:: python

    >>> from typing import Any
    >>> array_like: Any = (x**2 for x in range(10))
    >>> np.array(array_like)
    array(<generator object <genexpr> at 0x1192741d0>, dtype=object)

ndarray
~~~~~~~

It's possible to mutate the dtype of an array at runtime. For example,
the following code is valid:

.. code-block:: python

    >>> x = np.array([1, 2])
    >>> x.dtype = np.bool_

This sort of mutation is not allowed by the types. Users who want to
write statically typed code should insted use the `numpy.ndarray.view`
method to create a view of the array with a different dtype.

dtype
~~~~~

The ``DTypeLike`` type tries to avoid creation of dtype objects using
dictionary of fields like below:

.. code-block:: python

    >>> x = np.dtype({"field1": (float, 1), "field2": (int, 3)})

Although this is valid Numpy code, the type checker will complain about it,
since its usage is discouraged.
Please see : https://numpy.org/devdocs/reference/arrays.dtypes.html

"""
from ._array_like import _SupportsArray, ArrayLike
from ._shape import _Shape, _ShapeLike
from ._dtype_like import DtypeLike
