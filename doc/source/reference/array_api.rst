.. _array_api:

********************************
Array API Standard Compatibility
********************************

.. note::

   The ``numpy.array_api`` module is still experimental. See `NEP 47
   <https://numpy.org/neps/nep-0047-array-api-standard.html>`__.

NumPy includes a reference implementation of the `array API standard
<https://data-apis.org/array-api/latest/>`__ in ``numpy.array_api``. `NEP 47
<https://numpy.org/neps/nep-0047-array-api-standard.html>`__ describes the
motivation and scope for implementing the array API standard in NumPy.

The ``numpy.array_api`` module serves as a minimal, reference implementation
of the array API standard. In being minimal, the module only implements those
things that are explicitly required by the specification. Certain things are
allowed by the specification but are explicitly disallowed in
``numpy.array_api``. This is so that the module can serve as a reference
implementation for users of the array API standard. Any consumer of the array
API can test their code against ``numpy.array_api`` and be sure that they
aren't using any features that aren't guaranteed by the spec, and which may
not be present in other conforming libraries.

The ``numpy.array_api`` module is not documented here. For a listing of the
functions present in the array API specification, refer to the `array API
standard <https://data-apis.org/array-api/latest/>`__. The ``numpy.array_api``
implementation is functionally complete, so all functionality described in the
standard is implemented.

.. _array_api-differences:

Table of Differences between ``numpy.array_api`` and ``numpy``
==============================================================

This table outlines the primary differences between ``numpy.array_api`` from
the main ``numpy`` namespace. There are three types of differences:

1. **Strictness**. Things that are only done so that ``numpy.array_api`` is a
   strict, minimal implementation. They aren't actually required by the spec,
   and other conforming libraries may not follow them. In most cases, spec
   does not specify or require any behavior outside of the given domain. The
   main ``numpy`` namespace would not need to change in any way to be
   spec-compatible for these.

2. **Compatible**. Things that could be added to the main ``numpy`` namespace
   without breaking backwards compatibility.

3. **Breaking**. Things that would break backwards compatibility if
   implemented in the main ``numpy`` namespace.

Name Differences
----------------

Many functions have been renamed in the spec from NumPy. These are otherwise
identical in behavior, and are thus all **compatible** changes, unless
otherwise noted.

.. _array_api-name-changes:

Function Name Changes
~~~~~~~~~~~~~~~~~~~~~

The following functions are named differently in the array API

.. list-table::
   :header-rows: 1

   * - Array API name
     - NumPy namespace name
     - Notes
   * - ``acos``
     - ``arccos``
     -
   * - ``acosh``
     - ``arccosh``
     -
   * - ``asin``
     - ``arcsin``
     -
   * - ``asinh``
     - ``arcsinh``
     -
   * - ``atan``
     - ``arctan``
     -
   * - ``atan2``
     - ``arctan2``
     -
   * - ``atanh``
     - ``arctanh``
     -
   * - ``bitwise_left_shift``
     - ``left_shift``
     -
   * - ``bitwise_invert``
     - ``invert``
     -
   * - ``bitwise_right_shift``
     - ``right_shift``
     -
   * - ``bool``
     - ``bool_``
     - This is **breaking** because ``np.bool`` is currently a deprecated
       alias for the built-in ``bool``.
   * - ``concat``
     - ``concatenate``
     -
   * - ``matrix_norm`` and ``vector_norm``
     - ``norm``
     - ``matrix_norm`` and ``vector_norm`` each do a limited subset of what
       ``np.norm`` does.
   * - ``permute_dims``
     - ``transpose``
     - Unlike ``np.transpose``, the ``axis`` keyword-argument to
       ``permute_dims`` is required.
   * - ``pow``
     - ``power``
     -
   * - ``unique_all``, ``unique_counts``, ``unique_inverse``, and
       ``unique_values``
     - ``unique``
     - Each is equivalent to ``np.unique`` with certain flags set.


Function instead of method
~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``astype`` is a function in the array API, whereas it is a method on
  ``ndarray`` in ``numpy``.


``linalg`` Namespace Differences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These functions are in the ``linalg`` sub-namespace in the array API, but are
only in the top-level namespace in NumPy:

- ``cross``
- ``diagonal``
- ``matmul`` (*)
- ``outer``
- ``tensordot`` (*)
- ``trace``

(*): These functions are also in the top-level namespace in the array API.

Keyword Argument Renames
~~~~~~~~~~~~~~~~~~~~~~~~

The following functions have keyword arguments that have been renamed. The
functionality of the keyword argument is identical unless otherwise stated.
Renamed keyword arguments with the same semantic definition may be considered
either **compatible** or **breaking**, depending on how the change is
implemented.

Note, this page does not list function keyword arguments that are in the main
``numpy`` namespace but not in the array API. Such keyword arguments are
omitted from ``numpy.array_api`` for **strictness**, as the spec allows
functions to include additional keyword arguments from those required.

.. list-table::
   :header-rows: 1

   * - Function
     - Array API keyword name
     - NumPy keyword name
     - Notes
   * - ``argsort`` and ``sort``
     - ``stable``
     - ``kind``
     - The definitions of ``stable`` and ``kind`` differ, as do the default
       values. The change of the default value makes this **breaking**. See
       :ref:`array_api-set-functions-differences`.
   * - ``matrix_rank``
     - ``rtol``
     - ``tol``
     - The definitions of ``rtol`` and ``tol`` differ, as do the default
       values. The change of the default value makes this **breaking**. See
       :ref:`array_api-linear-algebra-differences`.
   * - ``pinv``
     - ``rtol``
     - ``rcond``
     - The definitions of ``rtol`` and ``rcond`` are the same, but their
       default values differ, making this **breaking**. See
       :ref:`array_api-linear-algebra-differences`.
   * - ``std`` and ``var``
     - ``correction``
     - ``ddof``
     -
   * - ``reshape``
     - ``shape``
     - ``newshape``
     - The argument may be passed as a positional or keyword argument for both
       NumPy and the array API.

.. _array_api-type-promotion-differences:

Type Promotion Differences
--------------------------

Type promotion is the biggest area where NumPy deviates from the spec. The
most notable difference is that NumPy does value-based casting in many cases.
The spec explicitly disallows value-based casting. In the array API, the
result type of any operation is always determined entirely by the input types,
independently of values or shapes.

.. list-table::
   :header-rows: 1

   * - Feature
     - Type
     - Notes
   * - Limited set of dtypes.
     - **Strictness**
     - ``numpy.array_api`` only implements those `dtypes that are required by
       the spec
       <https://data-apis.org/array-api/latest/API_specification/data_types.html>`__.
   * - Operators (like ``+``) with Python scalars only accept matching
       scalar types.
     - **Strictness**
     - For example, ``<int32 array> + 1.0`` is not allowed. See `the spec
       rules for mixing arrays and Python scalars
       <https://data-apis.org/array-api/latest/API_specification/type_promotion.html#mixing-arrays-with-python-scalars>`__.
   * - Operators (like ``+``) with Python scalars always return the same dtype
       as the array.
     - **Breaking**
     - For example, ``numpy.array_api.asarray(0., dtype=float32) + 1e64`` is a
       ``float32`` array.
   * - In-place operators are disallowed when the left-hand side would be
       promoted.
     - **Breaking**
     - Example: ``a = np.array(1, dtype=np.int8); a += np.array(1, dtype=np.int16)``. The spec explicitly disallows this.
   * - In-place operators are disallowed when the right-hand side operand
       cannot broadcast to the shape of the left-hand side operand.
     - **Strictness**
     - This so-called "reverse broadcasting" should not be allowed. Example:
       ``a = np.empty((2, 3, 4)); a += np.empty((3, 4))`` should error. See
       https://github.com/numpy/numpy/issues/10404.
   * - ``int`` promotion for operators is only specified for integers within
       the bounds of the dtype.
     - **Strictness**
     - ``numpy.array_api`` fallsback to ``np.ndarray`` behavior (either
       cast or raise ``OverflowError``).
   * - ``__pow__`` and ``__rpow__`` do not do value-based casting for 0-D
       arrays.
     - **Breaking**
     - For example, ``np.array(0., dtype=float32)**np.array(0.,
       dtype=float64)`` is ``float32``. Note that this is value-based casting
       on 0-D arrays, not scalars.
   * - No cross-kind casting.
     - **Strictness**
     - Namely, boolean, integer, and floating-point data types do not cast to
       each other, except explicitly with ``astype`` (this is separate from
       the behavior with Python scalars).
   * - No casting unsigned integer dtypes to floating dtypes (e.g., ``int64 +
       uint64 -> float64``.
     - **Strictness**
     -
   * - ``can_cast`` and ``result_type`` are restricted.
     - **Strictness**
     - The ``numpy.array_api`` implementations disallow cross-kind casting.
   * - ``sum`` and ``prod`` always upcast ``float32`` to ``float64`` when
       ``dtype=None``.
     - **Breaking**
     -

Indexing Differences
--------------------

The spec requires only a subset of indexing, but all indexing rules in the
spec are compatible with NumPy's more broad indexing rules.

.. list-table::
   :header-rows: 1

   * - Feature
     - Type
     - Notes
   * - No implicit ellipses (``...``).
     - **Strictness**
     - If an index does not include an ellipsis, all axes must be indexed.
   * - The start and stop of a slice may not be out of bounds.
     - **Strictness**
     - For a slice ``i:j:k``, only the following are allowed:

       - ``i`` or ``j`` omitted (``None``).
       - ``-n <= i <= max(0, n - 1)``.
       - For ``k > 0`` or ``k`` omitted (``None``), ``-n <= j <= n``.
       - For ``k < 0``, ``-n - 1 <= j <= max(0, n - 1)``.
   * - Boolean array indices are only allowed as the sole index.
     - **Strictness**
     -
   * - Integer array indices are not allowed at all.
     - **Strictness**
     - With the exception of 0-D arrays, which are treated like integers.

.. _array_api-type-strictness:

Type Strictness
---------------

Functions in ``numpy.array_api`` restrict their inputs to only those dtypes
that are explicitly required by the spec, even when the wrapped corresponding
NumPy function would allow a broader set. Here, we list each function and the
dtypes that are allowed in ``numpy.array_api``. These are **strictness**
differences because the spec does not require that other dtypes result in an
error. The categories here are defined as follows:

- **Floating-point**: ``float32`` or ``float64``.
- **Integer**: Any signed or unsigned integer dtype (``int8``, ``int16``,
  ``int32``, ``int64``, ``uint8``, ``uint16``, ``uint32``, or ``uint64``).
- **Boolean**: ``bool``.
- **Integer or boolean**: Any signed or unsigned integer dtype, or ``bool``.
  For two-argument functions, both arguments must be integer or both must be
  ``bool``.
- **Numeric**: Any integer or floating-point dtype. For two-argument
  functions, both arguments must be integer or both must be
  floating-point.
- **All**: Any of the above dtype categories. For two-argument functions, both
  arguments must be the same kind (integer, floating-point, or boolean).

In all cases, the return dtype is chosen according to `the rules outlined in
the spec
<https://data-apis.org/array-api/latest/API_specification/type_promotion.html>`__,
and does not differ from NumPy's return dtype for any of the allowed input
dtypes, except in the cases mentioned specifically in the subsections below.

Elementwise Functions
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Function Name
     - Dtypes
   * - ``abs``
     - Numeric
   * - ``acos``
     - Floating-point
   * - ``acosh``
     - Floating-point
   * - ``add``
     - Numeric
   * - ``asin`` (*)
     - Floating-point
   * - ``asinh`` (*)
     - Floating-point
   * - ``atan`` (*)
     - Floating-point
   * - ``atan2`` (*)
     - Floating-point
   * - ``atanh`` (*)
     - Floating-point
   * - ``bitwise_and``
     - Integer or boolean
   * - ``bitwise_invert``
     - Integer or boolean
   * - ``bitwise_left_shift`` (*)
     - Integer
   * - ``bitwise_or``
     - Integer or boolean
   * - ``bitwise_right_shift`` (*)
     - Integer
   * - ``bitwise_xor``
     - Integer or boolean
   * - ``ceil``
     - Numeric
   * - ``cos``
     - Floating-point
   * - ``cosh``
     - Floating-point
   * - ``divide``
     - Floating-point
   * - ``equal``
     - All
   * - ``exp``
     - Floating-point
   * - ``expm1``
     - Floating-point
   * - ``floor``
     - Numeric
   * - ``floor_divide``
     - Numeric
   * - ``greater``
     - Numeric
   * - ``greater_equal``
     - Numeric
   * - ``isfinite``
     - Numeric
   * - ``isinf``
     - Numeric
   * - ``isnan``
     - Numeric
   * - ``less``
     - Numeric
   * - ``less_equal``
     - Numeric
   * - ``log``
     - Floating-point
   * - ``logaddexp``
     - Floating-point
   * - ``log10``
     - Floating-point
   * - ``log1p``
     - Floating-point
   * - ``log2``
     - Floating-point
   * - ``logical_and``
     - Boolean
   * - ``logical_not``
     - Boolean
   * - ``logical_or``
     - Boolean
   * - ``logical_xor``
     - Boolean
   * - ``multiply``
     - Numeric
   * - ``negative``
     - Numeric
   * - ``not_equal``
     - All
   * - ``positive``
     - Numeric
   * - ``pow`` (*)
     - Numeric
   * - ``remainder``
     - Numeric
   * - ``round``
     - Numeric
   * - ``sign``
     - Numeric
   * - ``sin``
     - Floating-point
   * - ``sinh``
     - Floating-point
   * - ``sqrt``
     - Floating-point
   * - ``square``
     - Numeric
   * - ``subtract``
     - Numeric
   * - ``tan``
     - Floating-point
   * - ``tanh``
     - Floating-point
   * - ``trunc``
     - Numeric

(*) These functions have different names from the main ``numpy`` namespace.
See :ref:`array_api-name-changes`.

Creation Functions
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Function Name
     - Dtypes
   * - ``meshgrid``
     - Any (all input dtypes must be the same)


Linear Algebra Functions
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Function Name
     - Dtypes
   * - ``cholesky``
     - Floating-point
   * - ``cross``
     - Numeric
   * - ``det``
     - Floating-point
   * - ``diagonal``
     - Any
   * - ``eigh``
     - Floating-point
   * - ``eighvals``
     - Floating-point
   * - ``inv``
     - Floating-point
   * - ``matmul``
     - Numeric
   * - ``matrix_norm`` (*)
     - Floating-point
   * - ``matrix_power``
     - Floating-point
   * - ``matrix_rank``
     - Floating-point
   * - ``matrix_transpose`` (**)
     - Any
   * - ``outer``
     - Numeric
   * - ``pinv``
     - Floating-point
   * - ``qr``
     - Floating-point
   * - ``slogdet``
     - Floating-point
   * - ``solve``
     - Floating-point
   * - ``svd``
     - Floating-point
   * - ``svdvals`` (**)
     - Floating-point
   * - ``tensordot``
     - Numeric
   * - ``trace``
     - Numeric
   * - ``vecdot`` (**)
     - Numeric
   * - ``vector_norm`` (*)
     - Floating-point

(*) These functions are split from ``norm`` from the main ``numpy`` namespace.
See :ref:`array_api-name-changes`.

(**) These functions are new in the array API and are not in the main
``numpy`` namespace.

Array Object
~~~~~~~~~~~~

All the special ``__operator__`` methods on the array object behave
identically to their corresponding functions (see `the spec
<https://data-apis.org/array-api/latest/API_specification/array_object.html#methods>`__
for a list of which methods correspond to which functions). The exception is
that operators explicitly allow Python scalars according to the `rules
outlined in the spec
<https://data-apis.org/array-api/latest/API_specification/type_promotion.html#mixing-arrays-with-python-scalars>`__
(see :ref:`array_api-type-promotion-differences`).


Array Object Differences
------------------------

.. list-table::
   :header-rows: 1

   * - Feature
     - Type
     - Notes
   * - No array scalars
     - **Strictness**
     - The spec does not have array scalars, only 0-D arrays. However, other
       than the promotion differences outlined in
       :ref:`array_api-type-promotion-differences`, scalars duck type as 0-D
       arrays for the purposes of the spec. The are immutable, but the spec
       `does not require mutability
       <https://data-apis.org/array-api/latest/design_topics/copies_views_and_mutation.html>`__.
   * - ``bool()``, ``int()``, and ``float()`` only work on 0-D arrays.
     - **Strictness**
     - See https://github.com/numpy/numpy/issues/10404.
   * - ``__imatmul__``
     - **Compatible**
     - ``np.ndarray`` does not currently implement ``__imatmul``. Note that
       ``a @= b`` should only defined when it does not change the shape of
       ``a``.
   * - The ``mT`` attribute for matrix transpose.
     - **Compatible**
     - See `the spec definition
       <https://data-apis.org/array-api/latest/API_specification/generated/signatures.array_object.array.mT.html>`__
       for ``mT``.
   * - The ``T`` attribute should error if the input is not 2-dimensional.
     - **Breaking**
     - See `the note in the spec
       <https://data-apis.org/array-api/latest/API_specification/generated/signatures.array_object.array.T.html>`__.
   * - New method ``to_device`` and attribute ``device``
     - **Compatible**
     - The methods would effectively not do anything since NumPy is CPU only

Creation Functions Differences
------------------------------

.. list-table::
   :header-rows: 1

   * - Feature
     - Type
     - Notes
   * - ``copy`` keyword argument to ``asarray``
     - **Compatible**
     -
   * - New ``device`` keyword argument to all array creation functions
       (``asarray``, ``arange``, ``empty``, ``empty_like``, ``eye``, ``full``,
       ``full_like``, ``linspace``, ``ones``, ``ones_like``, ``zeros``, and
       ``zeros_like``).
     - **Compatible**
     - ``device`` would effectively do nothing, since NumPy is CPU only.

Elementwise Functions Differences
---------------------------------

.. list-table::
   :header-rows: 1

   * - Feature
     - Type
     - Notes
   * - Various functions have been renamed.
     - **Compatible**
     - See :ref:`array_api-name-changes`.
   * - Elementwise functions are only defined for given input type
       combinations.
     - **Strictness**
     - See :ref:`array_api-type-strictness`.
   * - ``bitwise_left_shift`` and ``bitwise_right_shift`` are only defined for
       ``x2`` nonnegative.
     - **Strictness**
     -
   * - ``ceil``, ``floor``, and ``trunc`` return an integer with integer
       input.
     - **Breaking**
     - ``np.ceil``, ``np.floor``, and ``np.trunc`` return a floating-point
       dtype on integer dtype input.

.. _array_api-linear-algebra-differences:

Linear Algebra Differences
--------------------------

.. list-table::
   :header-rows: 1

   * - Feature
     - Type
     - Notes
   * - ``cholesky`` includes an ``upper`` keyword argument.
     - **Compatible**
     -
   * - ``cross`` does not allow size 2 vectors (only size 3).
     - **Breaking**
     -
   * - ``diagonal`` operates on the last two axes.
     - **Breaking**
     - Strictly speaking this can be **compatible** because ``diagonal`` is
       moved to the ``linalg`` namespace.
   * - ``eigh``, ``qr``, ``slogdet`` and ``svd`` return a named tuple.
     - **Compatible**
     - The corresponding ``numpy`` functions return a ``tuple``, with the
       resulting arrays in the same order.
   * - New functions ``matrix_norm`` and ``vector_norm``.
     - **Compatible**
     - The ``norm`` function has been omitted from the array API and split
       into ``matrix_norm`` for matrix norms and ``vector_norm`` for vector
       norms. Note that ``vector_norm`` supports any number of axes, whereas
       ``np.linalg.norm`` only supports a single axis for vector norms.
   * - ``matrix_rank`` has an ``rtol`` keyword argument instead of ``tol``.
     - **Breaking**
     - In the array API, ``rtol`` filters singular values smaller than
       ``rtol * largest_singular_value``. In ``np.linalg.matrix_rank``,
       ``tol`` filters singular values smaller than ``tol``. Furthermore, the
       default value for ``rtol`` is ``max(M, N) * eps``, whereas the default
       value of ``tol`` in ``np.linalg.matrix_rank`` is ``S.max() *
       max(M, N) * eps``, where ``S`` is the singular values of the input. The
       new flag name is compatible but the default change is breaking
   * - ``matrix_rank`` does not support 1-dimensional arrays.
     - **Breaking**
     -
   * - New function ``matrix_transpose``.
     - **Compatible**
     - Unlike ``np.transpose``, ``matrix_transpose`` only transposes the last
       two axes. See `the spec definition
       <https://data-apis.org/array-api/latest/API_specification/generated/signatures.linear_algebra_functions.matrix_transpose.html#signatures.linear_algebra_functions.matrix_transpose>`__
   * - ``outer`` only supports 1-dimensional arrays.
     - **Breaking**
     - The spec currently only specifies behavior on 1-D arrays but future
       behavior will likely be to broadcast, rather than flatten, which is
       what ``np.outer`` does.
   * - ``pinv`` has an ``rtol`` keyword argument instead of ``rcond``
     - **Breaking**
     - The meaning of ``rtol`` and ``rcond`` is the same, but the default
       value for ``rtol`` is ``max(M, N) * eps``, whereas the default value
       for ``rcond`` is ``1e-15``. The new flag name is compatible but the
       default change is breaking.
   * - ``solve`` only accepts ``x2`` as a vector when it is exactly
       1-dimensional.
     - **Breaking**
     - The ``np.linalg.solve`` behavior is ambiguous. See `this numpy issue
       <https://github.com/numpy/numpy/issues/15349>`__ and `this array API
       specification issue
       <https://github.com/data-apis/array-api/issues/285>`__ for more
       details.
   * - New function ``svdvals``.
     - **Compatible**
     - Equivalent to ``np.linalg.svd(compute_uv=False)``.
   * - The ``axis`` keyword to ``tensordot`` must be a tuple.
     - **Compatible**
     - In ``np.tensordot``, it can also be an array or array-like.
   * - ``trace`` operates on the last two axes.
     - **Breaking**
     - ``np.trace`` operates on the first two axes by default. Note that the
       array API ``trace`` does not allow specifying which axes to operate on.

Manipulation Functions Differences
----------------------------------

.. list-table::
   :header-rows: 1

   * - Feature
     - Type
     - Notes
   * - Various functions have been renamed
     - **Compatible**
     - See :ref:`array_api-name-changes`.
   * - ``concat`` has different default casting rules from ``np.concatenate``
     - **Strictness**
     - No cross-kind casting. No value-based casting on scalars (when axis=None).
   * - ``stack`` has different default casting rules from ``np.stack``
     - **Strictness**
     - No cross-kind casting.
   * - New function ``permute_dims``.
     - **Compatible**
     - Unlike ``np.transpose``, the ``axis`` keyword argument to
       ``permute_dims`` is required.
   * - ``reshape`` function has a ``copy`` keyword argument
     - **Compatible**
     - See https://github.com/numpy/numpy/issues/9818.

Set Functions Differences
-------------------------

.. list-table::
   :header-rows: 1

   * - Feature
     - Type
     - Notes
   * - New functions ``unique_all``, ``unique_counts``, ``unique_inverse``,
       and ``unique_values``.
     - **Compatible**
     - See :ref:`array_api-name-changes`.
   * - The four ``unique_*`` functions return a named tuple.
     - **Compatible**
     -
   * - ``unique_all`` and ``unique_indices`` return indices with the same
       shape as ``x``.
     - **Compatible**
     - See https://github.com/numpy/numpy/issues/20638.

.. _array_api-set-functions-differences:

Set Functions Differences
-------------------------

.. list-table::
   :header-rows: 1

   * - Feature
     - Type
     - Notes
   * - ``argsort`` and ``sort`` have a ``stable`` keyword argument instead of
       ``kind``.
     - **Breaking**
     - ``stable`` is a boolean keyword argument, defaulting to ``True``.
       ``kind`` takes a string, defaulting to ``"quicksort"``. ``stable=True``
       is equivalent to ``kind="stable"`` and ``kind=False`` is equivalent to
       ``kind="quicksort"``, although any sorting algorithm is allowed by the
       spec when ``stable=False``. The new flag name is compatible but the
       default change is breaking.
   * - ``argsort`` and ``sort`` have a ``descending`` keyword argument.
     - **Compatible**
     -

Statistical Functions Differences
---------------------------------

.. list-table::
   :header-rows: 1

   * - Feature
     - Type
     - Notes
   * - ``sum`` and ``prod`` always upcast ``float32`` to ``float64`` when
       ``dtype=None``.
     - **Breaking**
     -
   * - The ``std`` and ``var`` functions have a ``correction`` keyword
       argument instead of ``ddof``.
     - **Compatible**
     -

Other Differences
-----------------

.. list-table::
   :header-rows: 1

   * - Feature
     - Type
     - Notes
   * - Dtypes can only be spelled as dtype objects.
     - **Strictness**
     - For example, ``numpy.array_api.asarray([0], dtype='int32')`` is not
       allowed.
   * - ``asarray`` is not implicitly called in any function.
     - **Strictness**
     - The exception is Python operators, which accept Python scalars in
       certain cases (see :ref:`array_api-type-promotion-differences`).
   * - ``tril`` and ``triu`` require the input to be at least 2-D.
     - **Strictness**
     -
   * - finfo() return type uses ``float`` for the various attributes.
     - **Strictness**
     - The spec allows duck typing, so ``finfo`` returning dtype
       scalars is considered type compatible with ``float``.
   * - Positional arguments in every function are positional-only.
     - **Breaking**
     - See the spec for the exact signature of each function. Note that NumPy
       ufuncs already use positional-only arguments, but non-ufuncs like
       ``asarray`` generally do not.
