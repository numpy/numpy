"""
============================
Typing (:mod:`numpy.typing`)
============================

Large parts of the NumPy API have :pep:`484`-style type annotations. In
addition a number of type aliases are available to users, most prominently
the two below:

- `ArrayLike`: objects that can be converted to arrays
- `DTypeLike`: objects that can be converted to dtypes

See the :ref:`user guide on static typing <basics.typing>`.

.. _typing-extensions: https://pypi.org/project/typing-extensions/

.. _typing.differences:

Differences from the runtime NumPy API
--------------------------------------

NumPy is very flexible. Trying to describe the full range of
possibilities statically would result in types that are not very
helpful. For that reason, the typed NumPy API is often stricter than
the runtime NumPy API. This section describes some notable
differences.

ArrayLike
~~~~~~~~~

The `ArrayLike` type tries to avoid creating object arrays. For
example,

.. code-block:: python

    >>> np.array(x**2 for x in range(10))
    array(<generator object <genexpr> at ...>, dtype=object)

is valid NumPy code which will create a 0-dimensional object
array. Type-checkers will complain about the above example when using
the NumPy types however. If you really intended to do the above, then
you can either use a ``# type: ignore`` comment:

.. code-block:: python

    >>> np.array(x**2 for x in range(10))  # type: ignore

or explicitly type the array like object as `~typing.Any`:

.. code-block:: python

    >>> from typing import Any
    >>> array_like: Any = (x**2 for x in range(10))
    >>> np.array(array_like)
    array(<generator object <genexpr> at ...>, dtype=object)

DTypeLike
~~~~~~~~~

The `DTypeLike` type tries to avoid creation of dtype objects using
dictionary of fields like below:

.. code-block:: python

    >>> x = np.dtype({"field1": (float, 1), "field2": (int, 3)})

Although this is valid NumPy code, the type-checker will complain about it,
since its usage is discouraged.
Please see : :ref:`Data type objects <arrays.dtypes>`

Number precision
~~~~~~~~~~~~~~~~

The precision of `numpy.number` subclasses is treated as an invariant generic
parameter (see :class:`~NBitBase`), simplifying the annotating of processes
involving precision-based casting.

.. code-block:: python

    >>> from typing import TypeVar
    >>> import numpy as np
    >>> import numpy.typing as npt

    >>> T = TypeVar("T", bound=npt.NBitBase)
    >>> def func(a: np.floating[T], b: np.floating[T]) -> np.floating[T]:
    ...     ...

.. deprecated:: 2.3
    The :class:`~numpy.typing.NBitBase` helper is deprecated and will be
    removed in a future release. Prefer expressing precision relationships via
    ``typing.overload`` or ``TypeVar`` definitions bounded by concrete scalar
    classes. For example:

    .. code-block:: python

        from typing import TypeVar
        import numpy as np

        S = TypeVar("S", bound=np.floating)

        def func(a: S, b: S) -> S:
            ...

    or in the case of different input types mapping to different output types:

   .. code-block:: python

        from typing import overload
        import numpy as np

        @overload
        def phase(x: np.complex64) -> np.float32: ...
        @overload
        def phase(x: np.complex128) -> np.float64: ...
        @overload
        def phase(x: np.clongdouble) -> np.longdouble: ...
        def phase(x: np.complexfloating) -> np.floating:
            ...

Timedelta64
~~~~~~~~~~~

The `~numpy.timedelta64` class is not considered a subclass of
`~numpy.signedinteger`, the former only inheriting from `~numpy.generic`
while static type checking.

Record array dtypes
~~~~~~~~~~~~~~~~~~~

The dtype of `numpy.recarray`, and the :ref:`routines.array-creation.rec`
functions in general, can be specified in one of two ways:

* Directly via the ``dtype`` argument.
* With up to five helper arguments that operate via `numpy.rec.format_parser`:
  ``formats``, ``names``, ``titles``, ``aligned`` and ``byteorder``.

These two approaches are currently typed as being mutually exclusive,
*i.e.* if ``dtype`` is specified than one may not specify ``formats``.
While this mutual exclusivity is not (strictly) enforced during runtime,
combining both dtype specifiers can lead to unexpected or even downright
buggy behavior.

Scalar type sizes
~~~~~~~~~~~~~~~~~

NumPy's stubs assume a 64-bit platform, and a 32-bit C ``int``:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Scalar types
     - Type-checkers see
   * - ``np.intp``, ``np.int_``, ``np.longlong``
     - ``np.int64``
   * - ``np.uintp``, ``np.uint``, ``np.ulonglong``
     - ``np.uint64``
   * - ``np.long``
     - ``np.int64``, or ``np.int32`` on Windows
   * - ``np.ulong``
     - ``np.uint64``, or ``np.uint32`` on Windows
   * - ``np.intc``
     - ``np.int32``
   * - ``np.uintc``
     - ``np.uint32``

``np.long`` and ``np.ulong`` depend on `sys.platform`.

On 32-bit platforms, ``np.intp``, ``np.int_``, ``np.long``, and their unsigned
counterparts are 32-bit at runtime. On the rare platforms where C ``int`` isn't
32-bit, ``np.intc`` and ``np.uintc`` have a different size. On these platforms, the
types above are incorrect.

The sizes of `np.longdouble <numpy.longdouble>` and
`np.clongdouble <numpy.clongdouble>` depend on the platform, so they are separate
types, not aliases of other scalar types. ``np.float96``, ``np.float128``,
``np.complex192``, and ``np.complex256`` are aliases of these two types. The stubs
always define all four, but at runtime a platform has at most one of ``np.float96``
and ``np.float128``, and at most one of ``np.complex192`` and ``np.complex256``. For
example, x86-64 Linux has ``np.float128`` but not ``np.float96``. On some platforms,
such as Windows and arm64 macOS, ``np.longdouble`` is a 64-bit float, like
``np.float64``, and none of the four aliases exist. Type-checkers still treat
``np.longdouble`` and ``np.float64`` as different types there.

Supported type-checkers
-----------------------

NumPy's stubs follow the `Python typing specification`_, rather than the behavior of
a specific type-checker, and support all modern type-checkers. No plugins are needed.

Where a type-checker deviates from the typing specification in a way that would
degrade inference, the stubs can contain workarounds, such as a specific order of
overloads, or extra overloads for a specific type-checker. This explains some unusual
overloads in the stubs.

These deviations can also affect your code. For example, if an argument is
(partially) ``Any``, several overloads can match it. If several of these remain after
the specification's filtering steps, and their return types aren't equivalent, the
call is ambiguous (see `overload call evaluation`_). The specification then requires
a result type that is assignable to each of these return types, such as ``Any``. Some
type-checkers pick the first matching overload instead, so type-checkers can infer
different types, especially for arrays with an unknown dtype. Arrays with a known
dtype avoid most of these differences.

If you find an incorrect type, or type-checkers that disagree about a type, please
`open an issue <https://github.com/numpy/numpy/issues>`__.

.. _Python typing specification: https://typing.python.org/en/latest/spec/
.. _overload call evaluation:
   https://typing.python.org/en/latest/spec/overload.html#step-5

API
---

.. rubric:: ndarray

The `numpy.ndarray` class is a `generic type`_ that accepts two type arguments:

1. The type of `numpy.ndarray.shape`, which must be a `tuple` of `int`, e.g.
   ``tuple[int, int]`` (2-D shape) or ``tuple[()]`` (0-D shape).
   The default shape is ``tuple[Any, ...]``, which represents an unknown shape with
   *any* number of dimensions.
   Currently, ``Literal`` ints or other more specific types are not supported.
2. The type of `numpy.ndarray.dtype`, which must be a subtype of `numpy.dtype` such as
   ``numpy.dtype[numpy.float64]``. If omitted, it will default to ``numpy.dtype[Any]``.

.. code-block:: python

    >>> import numpy as np

    >>> type ImageRGB = np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]
    >>> type Vector[S: np.generic] = np.ndarray[tuple[int], np.dtype[S]]

.. _generic type: https://typing.python.org/en/latest/spec/generics.html

"""
# NOTE: The API section will be appended with additional entries
# further down in this file

# pyright: reportDeprecated=false

from numpy._typing import (  # type: ignore[deprecated]
    Array0D,
    Array1D,
    Array2D,
    Array3D,
    Array4D,
    ArrayLike,
    DTypeLike,
    NBitBase,
    NDArray,
)

__all__ = [
    "Array0D",
    "Array1D",
    "Array2D",
    "Array3D",
    "Array4D",
    "ArrayLike",
    "DTypeLike",
    "NBitBase",
    "NDArray",
]


__DIR = __all__ + [k for k in globals() if k.startswith("__") and k.endswith("__")]
__DIR_SET = frozenset(__DIR)


def __dir__() -> list[str]:
    return __DIR

def __getattr__(name: str) -> object:
    if name == "NBitBase":
        import warnings

        # Deprecated in NumPy 2.3, 2025-05-01
        warnings.warn(
            "`NBitBase` is deprecated and will be removed from numpy.typing in the "
            "future. Use `@typing.overload` or a `TypeVar` with a scalar-type as upper "
            "bound, instead. (deprecated in NumPy 2.3)",
            DeprecationWarning,
            stacklevel=2,
        )
        return NBitBase

    if name in __DIR_SET:
        return globals()[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __doc__ is not None:
    from numpy._typing._add_docstring import _docstrings
    __doc__ += _docstrings
    __doc__ += '\n.. autoclass:: numpy.typing.NBitBase\n'
    del _docstrings

from numpy._pytesttester import PytestTester

test = PytestTester(__name__)
del PytestTester
