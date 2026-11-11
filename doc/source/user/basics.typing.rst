.. _basics.typing:

*************
Static typing
*************

The ``numpy`` package includes `type stubs`_ (``.pyi`` files) for static type-checkers.
The stubs have no effect at runtime. This page covers type inference, annotations, and
shape-typing.

The public typing aliases, which can also be used at runtime, are documented in
`numpy.typing`, together with the assumptions that the stubs make about the platform and
the type-checker.

This page is specific to NumPy, and isn't an introduction to static typing in Python.
For that, see the `Python typing documentation`_.

.. note::

   ``reveal_type(x)  # T`` means that static type-checkers infer the type ``T`` for
   ``x``, see |reveal_type|_.


Introduction
============

Static type-checkers use NumPy's stubs to determine, or *infer*, the type of each
expression, without running the code. This also works for code without annotations:

.. code-block:: python

    import numpy as np

    x = np.linspace(0, 1, 50)
    reveal_type(x)                 # npt.Array1D[np.float64]

    grid = x.reshape(5, 10)
    reveal_type(grid)              # npt.Array2D[np.float64]
    reveal_type(grid.sum(axis=0))  # npt.Array1D[np.float64]
    reveal_type(grid > 0.5)        # npt.Array2D[np.bool]
    reveal_type(grid[0, 0])        # np.float64

Here ``npt`` is `numpy.typing`, and ``npt.Array1D[np.float64]`` is the type of a 1-d
array with dtype ``np.float64``. The inferred types describe the number of dimensions
and the dtype of each array, see :ref:`basics.typing.shape-typing`. ``npt.Array1D`` and
``npt.Array2D`` are new in NumPy 2.6. Type-checkers can display an equivalent form, such
as ``np.ndarray[tuple[int], np.dtype[np.float64]]`` for ``npt.Array1D[np.float64]``.

With these types, type-checkers report mistakes before the code is run:

.. code-block:: python

    rows, cols = x.shape  # error: x is 1-d, so its shape has 1 value
    np.zeros(3, 4)        # error: 4 isn't a dtype; a 2-d shape must be a tuple
    x[0.5]                # error: a float isn't a valid index
    len(x.mean())         # error: the mean of x is a scalar

They can also report the use of deprecated functionality:

.. code-block:: python

    grid.shape = (10, 5)  # deprecated: use reshape instead
    np.fix(x)             # deprecated: use np.trunc instead

The types of some functions depend on the values of their arguments. For example, for
commonly used subscripts, the number of dimensions of the result of
`np.einsum <numpy.einsum>` is inferred from the subscripts:

.. code-block:: python

    reveal_type(np.einsum("ij,kj->ik", grid, grid))  # npt.Array2D[np.float64]
    reveal_type(np.einsum("ij,ij->i", grid, grid))   # npt.Array1D[np.float64]
    reveal_type(np.einsum("ij,ij->", grid, grid))    # np.float64


Summary
=======

The main recommendations of this page are:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Use
     - Instead of
     - Details
   * - ``dtype=np.float64``
     - ``dtype=float`` or ``dtype="float64"``
     - `dtype arguments`_
   * - ``npt.Array2D[np.float64]``, if the number of dimensions is known
     - ``np.ndarray``
     - `Function annotations`_, `Shape-typing`_
   * - ``npt.NDArray[np.float64]``, if the number of dimensions is unknown
     - ``np.ndarray[tuple[int, ...], ...]`` or ``np.ndarray[Any, ...]``
     - `Unknown number of dimensions`_
   * - Concrete return types, such as ``npt.Array2D[np.float64]``
     - ``npt.ArrayLike`` or ``npt.DTypeLike`` as return types
     - `ArrayLike and DTypeLike inputs`_
   * - :class:`int` axis sizes, as in ``tuple[int, int]``
     - ``Literal`` axis sizes
     - `Literal axis sizes`_


Type inference
==============

``dtype`` arguments
-------------------

To specify a dtype, pass a NumPy scalar type, such as `np.float64 <numpy.float64>`, as
the ``dtype`` argument. A dtype with a known scalar type, such as the ``dtype``
attribute of an ``npt.NDArray[np.float64]``, also works. The following arrays have the
same dtype at runtime, but due to limitations in Python's type system, only the first
has a known dtype for type-checkers:

.. code-block:: python

    import numpy as np

    a = np.zeros((2, 3), dtype=np.float64)   # yes
    reveal_type(a)                           # npt.Array2D[np.float64]

    b = np.eye(3, dtype=float)               # no
    reveal_type(b)                           # npt.Array2D[Any]

    c = np.zeros(5, dtype="float")           # no
    reveal_type(c)                           # npt.Array1D[Any]

    d = np.ones((2, 3, 4), dtype="float64")  # no
    reveal_type(d)                           # npt.Array3D[Any]

``Any`` is a `gradual form`_, so type-checkers don't check the dtype of these arrays,
only their number of dimensions in this case. ``Any`` is best avoided, because
type-checkers can detect fewer mistakes in code that uses it. Other functions and
methods with a ``dtype`` parameter behave the same way, and a scalar result becomes
``Any`` altogether:

.. code-block:: python

    import numpy as np

    x = np.zeros((2, 3))

    reveal_type(x.astype(np.float32))     # npt.Array2D[np.float32]
    reveal_type(x.astype("float32"))      # npt.Array2D[Any]

    reveal_type(x.sum(dtype=np.float32))  # np.float32
    reveal_type(x.sum(dtype=float))       # Any

Some commonly used functions also infer the dtype from a few of the other spellings,
but most functions don't. If the default dtype is what you need, omit the ``dtype``:

.. code-block:: python

    import numpy as np

    reveal_type(np.linspace(0, 1, 5))  # npt.Array1D[np.float64]
    reveal_type(np.eye(3))             # npt.Array2D[np.float64]
    reveal_type(np.ones((2, 3, 4)))    # npt.Array3D[np.float64]

Shape arguments
---------------

The number of dimensions is inferred from the ``shape`` argument, which can be an
:class:`int` or a tuple of :class:`int`:

.. code-block:: python

    import numpy as np


    def f(n: int, shape: tuple[int, ...]) -> None:
        reveal_type(np.zeros(n))       # npt.Array1D[np.float64]
        reveal_type(np.zeros((n, 3)))  # npt.Array2D[np.float64]

        # the length of this tuple is not known
        reveal_type(np.zeros(shape))   # npt.NDArray[np.float64]


Annotations
===========

Function annotations
--------------------

Annotate the parameter and return types of functions as precisely as possible, and let
the type-checker infer the types of local variables. For example, this function fits a
straight line through a set of points:

.. code-block:: python

    import numpy as np
    import numpy.typing as npt


    def fit_line(
        x: npt.Array1D[np.float64],
        y: npt.Array1D[np.float64],
    ) -> tuple[np.float64, np.float64]:
        design = np.column_stack([x, np.ones_like(x)])
        reveal_type(design)               # npt.Array2D[np.float64]

        solution, *_ = np.linalg.lstsq(design, y)
        reveal_type(solution)             # npt.Array1D[np.float64]

        slope, intercept = solution
        return slope, intercept


    x = np.linspace(0.0, 1.0, 50)
    y = 3 * x + 1

    slope, intercept = fit_line(x, y)
    fit_line(np.column_stack([x, y]), y)  # error: 2-d array, expected a 1-d array

Only the signature of ``fit_line`` is annotated. The types of ``design`` and
``solution`` are inferred. For example, the stubs declare that
`np.column_stack <numpy.column_stack>` returns a 2-d array when it's given 1-d arrays.

The last call passes a 2-d array as ``x``. At runtime, ``np.column_stack`` and
`np.linalg.lstsq <numpy.linalg.lstsq>` accept it, and the error only occurs at
``slope, intercept = solution``, as a ``ValueError``. A static type-checker reports the
error at the call, before the code is run.

Writing ``npt.Array1D[np.float64]`` takes no more effort than writing
``npt.NDArray[np.float64]`` or ``np.ndarray``, but only ``npt.Array1D`` lets
type-checkers report this error. `np.ndarray <numpy.ndarray>` without type arguments is
equivalent to ``npt.NDArray[Any]``: it has an unknown shape and dtype, and so do most
results that are computed from it:

.. code-block:: python

    import numpy as np
    import numpy.typing as npt


    def f(
        untyped: np.ndarray,
        floating: npt.NDArray[np.floating],
        float64: npt.NDArray[np.float64],
    ) -> None:
        reveal_type(untyped.mean())   # Any
        reveal_type(floating.mean())  # np.floating
        reveal_type(float64.mean())   # np.float64

Abstract scalar types such as `np.floating <numpy.floating>` are useful in parameter
types and as `upper bounds <upper bound_>`__ of type parameters. Return types should be
concrete, such as `np.float64 <numpy.float64>`, when possible.

Scalar type promotion
---------------------

The dtype of a result often depends on the dtypes of the inputs, following NumPy's
:ref:`promotion rules <arrays.promotion>`, as defined in :ref:`NEP 50 <NEP50>`. To
describe this in annotations, use type parameters or overloads. If the result has the
same dtype as one of the parameters, use a type parameter with an upper bound:

.. code-block:: python

    import numpy as np
    import numpy.typing as npt


    def first_column[ScalarT: np.generic](
        a: npt.Array2D[ScalarT],
    ) -> npt.Array1D[ScalarT]:
        return a[:, 0]

Otherwise, use `overloads`_. They are more verbose, but each combination of dtypes can
have its own return type. For example, this ``midpoint`` follows the promotion rules for
`np.float32 <numpy.float32>` and ``np.float64``:

.. code-block:: python

    from typing import overload

    import numpy as np
    import numpy.typing as npt


    @overload
    def midpoint(
        a: npt.Array1D[np.float32],
        b: npt.Array1D[np.float32],
    ) -> npt.Array1D[np.float32]: ...
    @overload
    def midpoint(
        a: npt.Array1D[np.float32],
        b: npt.Array1D[np.float64],
    ) -> npt.Array1D[np.float64]: ...
    @overload
    def midpoint(
        a: npt.Array1D[np.float64],
        b: npt.Array1D[np.float64 | np.float32],
    ) -> npt.Array1D[np.float64]: ...
    def midpoint(
        a: npt.Array1D[np.float64 | np.float32],
        b: npt.Array1D[np.float64 | np.float32],
    ) -> npt.Array1D[np.float64 | np.float32]:
        return (a + b) / 2


    x32 = np.zeros(3, dtype=np.float32)
    x64 = np.zeros(3, dtype=np.float64)

    reveal_type(midpoint(x32, x32))  # npt.Array1D[np.float32]
    reveal_type(midpoint(x32, x64))  # npt.Array1D[np.float64]
    reveal_type(midpoint(x64, x32))  # npt.Array1D[np.float64]

``ArrayLike`` and ``DTypeLike`` inputs
--------------------------------------

Use `~numpy.typing.ArrayLike` for parameters that accept array-like objects, and
`~numpy.typing.DTypeLike` for parameters that accept dtype-like objects. ``DTypeLike``
doesn't include ``None``, so write ``npt.DTypeLike | None`` for an optional ``dtype``
parameter. Convert array-like inputs with, for example, `np.asarray <numpy.asarray>`,
which also accepts the ``dtype``:

.. code-block:: python

    from typing import Any

    import numpy as np
    import numpy.typing as npt


    def load_table(
        values: npt.ArrayLike,
        dtype: npt.DTypeLike | None = None,
    ) -> npt.Array2D[Any]:
        table = np.asarray(values, dtype=dtype)
        reveal_type(table)  # npt.NDArray[Any]

        if table.ndim != 2:
            raise ValueError("expected a 2-d array")

        return table

Type-checkers don't `narrow`_ the type of ``table`` based on the ``ndim`` check. The
``return`` is accepted because an array with an unknown number of dimensions is
`assignable`_ to an array type with any number of dimensions, see
:ref:`basics.typing.unknown-rank`.

The dtype of ``table`` is unknown, because ``DTypeLike`` can describe any dtype. With a
fixed dtype, such as ``np.asarray(values, dtype=np.float64)``, the type of ``table``
would be ``npt.NDArray[np.float64]``. So use ``DTypeLike`` only as a fallback. Where
possible, accept more specific types, such as ``type[np.float32]`` and
``type[np.float64]`` in `overloads`_, or ``type[ScalarT]`` with a type parameter, so
that type-checkers can infer the dtype of the result.

``ArrayLike`` and ``DTypeLike`` exclude some inputs that NumPy accepts at runtime. For
example, ``ArrayLike`` excludes generators, which NumPy converts to an object array, and
``DTypeLike`` excludes some dictionaries of fields. See :ref:`typing.differences` for
details.

.. important::

   Only use ``ArrayLike`` and ``DTypeLike`` as parameter types, never as return types.
   They are wide unions, for example of Python scalars, nested sequences, strings, and
   types, so callers can't do anything with such a return value without converting or
   narrowing it first. Return a concrete type, such as ``npt.Array2D[np.float64]`` or
   ``np.dtype[np.float64]``, instead.


.. _basics.typing.shape-typing:

Shape-typing
============

`np.ndarray <numpy.ndarray>` is a `generic class`_ with two type parameters: the
shape-type, which is the type of `np.ndarray.shape <numpy.ndarray.shape>`, and the dtype
type, which is the type of `np.ndarray.dtype <numpy.ndarray.dtype>`. For example, the
``shape`` of an ``np.ndarray[tuple[int, int], np.dtype[np.float64]]`` has type
``tuple[int, int]``, and its ``dtype`` has type ``np.dtype[np.float64]``.

`numpy.typing` provides `type aliases`_ for the common cases. Each alias takes a scalar
type ``ScalarT`` as type argument, and is equivalent to
``np.ndarray[ShapeT, np.dtype[ScalarT]]``, with the shape-type ``ShapeT`` from this
table:

.. list-table::
   :header-rows: 1
   :widths: auto

   * - Alias
     - ``ShapeT``
     - Typical use
   * - `~numpy.typing.Array0D`
     - ``tuple[()]``
     - A single value
   * - `~numpy.typing.Array1D`
     - ``tuple[int]``
     - A vector or time series
   * - `~numpy.typing.Array2D`
     - ``tuple[int, int]``
     - A matrix or table
   * - `~numpy.typing.Array3D`
     - ``tuple[int, int, int]``
     - An RGB image
   * - `~numpy.typing.Array4D`
     - ``tuple[int, int, int, int]``
     - A batch of RGB images
   * - `~numpy.typing.NDArray`
     - ``tuple[Any, ...]``
     - An unknown number of dimensions

.. versionadded:: 2.6
   ``Array0D``, ``Array1D``, ``Array2D``, ``Array3D``, and ``Array4D``.

The shape-type describes the number of dimensions of an array, not the size of each
dimension: ``Array2D`` is any 2-d array. See :ref:`basics.typing.literals` for the
reasons.

``NDArray[ScalarT]`` is assignable to each of ``Array0D[ScalarT]``,
``Array1D[ScalarT]``, ``Array2D[ScalarT]``, ``Array3D[ScalarT]``, and
``Array4D[ScalarT]``, and each of these is assignable to ``NDArray[ScalarT]``. Use
``NDArray`` when the number of dimensions isn't known statically, see
:ref:`basics.typing.unknown-rank`.

Shape-type propagation
----------------------

Type-checkers infer the number of dimensions of the results of indexing, reductions,
and reshaping:

.. code-block:: python

    import numpy as np
    import numpy.typing as npt


    def f(a: npt.Array2D[np.float64]) -> None:
        reveal_type(a.shape)              # tuple[int, int]
        rows, cols = a.shape

        row = a[0]
        reveal_type(row)                  # npt.Array1D[np.float64]
        element = a[0, 0]
        reveal_type(element)              # np.float64

        reveal_type(a.mean())             # np.float64
        reveal_type(a.mean(axis=0))       # npt.Array1D[np.float64]
        reveal_type(a.ravel())            # npt.Array1D[np.float64]
        reveal_type(a.reshape(-1, 2, 2))  # npt.Array3D[np.float64]

        rows, cols, depth = a.shape       # error: 2 values, expected 3

If the number of dimensions is unknown, the inferred result types are often less
precise:

.. code-block:: python

    import numpy as np
    import numpy.typing as npt


    def f(a: npt.NDArray[np.float64]) -> None:
        reveal_type(a.shape)                       # tuple[Any, ...]

        row_or_element = a[0]
        reveal_type(row_or_element)                # Any

        reveal_type(a.sum(axis=0, keepdims=True))  # npt.NDArray[np.float64]

        # the number of dimensions of these results doesn't depend on the input
        reveal_type(a.ravel())                     # npt.Array1D[np.float64]
        reveal_type(a.reshape(-1, 2))              # npt.Array2D[np.float64]

Shape-typed functions
---------------------

A `type statement`_ can give a shape-typed array a name:

.. code-block:: python

    import numpy as np
    import numpy.typing as npt

    type ImageRGB = npt.Array3D[np.uint8]

    LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


    def to_grayscale(image: ImageRGB) -> npt.Array2D[np.float32]:
        scaled = image.astype(np.float32) / 255
        reveal_type(scaled)       # npt.Array3D[np.float32]

        red_channel = scaled[..., 0]
        reveal_type(red_channel)  # npt.Array2D[np.float32]

        # the result of this operation is not fully inferred (see below)
        return scaled @ LUMA_WEIGHTS

To preserve the number of dimensions, use a type parameter for the shape-type. Its upper
bound is ``tuple[int, ...]``, which every shape-type is assignable to, including
``tuple[Any, ...]``:

.. code-block:: python

    import numpy as np
    import numpy.typing as npt


    def clip_unit[ShapeT: tuple[int, ...]](
        a: np.ndarray[ShapeT, np.dtype[np.float64]],
    ) -> np.ndarray[ShapeT, np.dtype[np.float64]]:
        return a.clip(0.0, 1.0)


    def f(a: npt.Array1D[np.float64], b: npt.Array2D[np.float64]) -> None:
        reveal_type(clip_unit(a))  # npt.Array1D[np.float64]
        reveal_type(clip_unit(b))  # npt.Array2D[np.float64]

.. _basics.typing.unknown-rank:

Unknown number of dimensions
----------------------------

The shape-type of ``NDArray`` is ``tuple[Any, ...]``, which is
`assignable to and from tuples of any length`_. This makes it possible to add
shape-types to existing code one function at a time. However, type-checkers can't report
errors in the number of dimensions of an ``NDArray``.

.. warning::

   Never use ``tuple[int, ...]`` as the shape-type of an array, for example to mean "any
   number of dimensions". Unlike ``tuple[Any, ...]``, it is a `fully static type`_, so
   it isn't assignable to fixed-length shape-types such as ``tuple[int, int]``. An array
   with this shape-type can't be passed to functions that expect an ``Array1D``,
   ``Array2D``, and so on, which leads to type errors throughout the code. Use
   ``NDArray`` or ``tuple[Any, ...]`` instead.

   As the type of a shape value, such as a ``shape`` parameter, or as the upper bound of
   a shape-type parameter, ``tuple[int, ...]`` is fine.

   Also avoid ``Any`` as the shape-type, as in ``np.ndarray[Any]``. It is more gradual
   than ``tuple[Any, ...]``, without any advantage: the ``shape`` attribute then has
   type ``Any``, which is assignable to :class:`str` and other wrong types.

For example:

.. code-block:: python

    from typing import Any

    import numpy as np
    import numpy.typing as npt


    def solve(a: npt.Array2D[np.float64]) -> None: ...


    def f(
        vector: npt.Array1D[np.float64],
        unknown: npt.NDArray[np.float64],
        wrong: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
        too_gradual: np.ndarray[Any, np.dtype[np.float64]],
    ) -> None:
        solve(unknown)
        solve(vector)                   # error: 1-d array, expected a 2-d array
        solve(wrong)                    # error: not assignable to tuple[int, int]
        label: str = too_gradual.shape  # accepted, because the shape is Any

Limitations
-----------

Some operations aren't shape-typed. Operations on arrays with different numbers of
dimensions, such as ``Array2D - Array1D`` or ``Array3D @ Array1D``, usually result in
an unknown number of dimensions, and sometimes in an unknown dtype. The same applies to
functions where the number of dimensions of the result depends on runtime values, such
as `np.squeeze <numpy.squeeze>`.

In these cases, annotate the return type or the variable, as in ``to_grayscale`` above.
``keepdims=True`` can also help, because it preserves the number of dimensions:

.. code-block:: python

    import numpy as np
    import numpy.typing as npt


    def standardize(a: npt.Array2D[np.float64]) -> npt.Array2D[np.float64]:
        mean = a.mean(axis=0, keepdims=True)
        std = a.std(axis=0, keepdims=True)
        reveal_type(mean)      # npt.Array2D[np.float64]

        centered = a - mean
        reveal_type(centered)  # npt.Array2D[np.float64]
        return centered / std

Most functions are shape-typed for up to 4 dimensions. Except for operations on arrays
with the same number of dimensions, results with more dimensions usually have an unknown
number of dimensions. There is no ``Array5D``; write
``np.ndarray[tuple[int, int, int, int, int], np.dtype[np.float64]]`` instead.

Shape-types don't describe the size of each dimension, see
:ref:`basics.typing.literals`. Other ways to support exact sizes are being investigated,
but there are no concrete plans yet.

.. _basics.typing.literals:

``Literal`` axis sizes
----------------------

NumPy doesn't support |Literal|_ axis sizes in shape-types, such as
``tuple[Literal[2], Literal[3]]``. Only :class:`int` is supported. If you use literal
axis sizes anyway, NumPy can't guarantee correct type inference.

The reasons are:

- **Inference:** The type system has no arithmetic on literals, and many sizes depend on
  runtime values. For example, ``np.ones(12).reshape((3, -1))`` has shape ``(3, 4)``,
  but a shape-type taken from the arguments would be ``tuple[Literal[3], Literal[-1]]``.
  Slicing, boolean indexing, concatenation, and broadcasting have the same problem. Only
  a few functions, such as ``np.ones``, could infer correct literal sizes, so most
  results would have :class:`int` sizes anyway.
- **Assignability:** ``int`` isn't assignable to a ``Literal`` type, so most arrays
  couldn't be passed to a function that requires literal sizes without a |cast|_ or a
  |type-ignore|_ comment.
- **Stub behavior:** The stubs assume that shape-types only describe the number of
  dimensions. For example, transposing returns the same shape-type, so the transpose of
  a ``(2, 3)`` array with a literal shape-type would still claim the shape ``(2, 3)``
  instead of ``(3, 2)``.
- **Type-checker differences:** Type-checkers handle literals differently, while NumPy
  aims for the same inference in all of them. Literal shape-types would also make error
  messages longer.

.. _Python typing documentation: https://typing.python.org/en/latest/
.. _type stubs: https://typing.python.org/en/latest/spec/distributing.html#stub-files
.. _reveal_type: https://typing.python.org/en/latest/spec/directives.html#reveal-type
.. _gradual form:
   https://typing.python.org/en/latest/spec/glossary.html#term-gradual-form
.. _fully static type:
   https://typing.python.org/en/latest/spec/glossary.html#term-fully-static-type
.. _narrow: https://typing.python.org/en/latest/spec/glossary.html#term-narrow
.. _assignable: https://typing.python.org/en/latest/spec/glossary.html#term-assignable
.. _overloads: https://typing.python.org/en/latest/spec/overload.html
.. _generic class:
   https://typing.python.org/en/latest/spec/generics.html#user-defined-generic-types
.. _type aliases: https://typing.python.org/en/latest/spec/aliases.html
.. _type statement: https://typing.python.org/en/latest/spec/aliases.html#type-statement
.. _upper bound:
   https://typing.python.org/en/latest/spec/generics.html#type-variables-with-an-upper-bound
.. _assignable to and from tuples of any length:
   https://typing.python.org/en/latest/spec/tuples.html#tuple-type-form
.. _Literal: https://typing.python.org/en/latest/spec/literal.html
.. _cast: https://typing.python.org/en/latest/spec/directives.html#cast
.. _type-ignore:
   https://typing.python.org/en/latest/spec/directives.html#type-ignore-comments

.. |reveal_type| replace:: ``reveal_type``
.. |Literal| replace:: ``Literal``
.. |cast| replace:: ``cast``
.. |type-ignore| replace:: ``# type: ignore``
