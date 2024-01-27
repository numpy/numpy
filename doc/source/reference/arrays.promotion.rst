.. currentmodule:: numpy

.. _arrays.promotion:

****************************
Data type promotion in NumPy
****************************

When mixing two different data types NumPy has to find the correct dtype for
the result of the operation.  This step is referred *promotion* or finding
the common dtype.
In most you do not have to know much about the details of promotion since
the promotion step generally ensures that the result of:
1. Combining multiple arrays e.g. with ``np.concatenate``
2. The evaluation of any mathematical operation

will return in a result that either matches its input dtypes or has a higher
precision.
The ``dtype`` of the result can be found using the `result_type` and
`promote_types` functions for most (but not all) operations.

These are examples, for howNumPy behaves for both arrays and scalar
operations.  Here the result matches the input:

  >>> np.int8(1) + np.int8(1)
  np.int8(2)

While mixing two different ``dtypes`` normally results in the larger of both:

  >>> np.int8(4) + np.int64(8)
  np.int64(12)
  >>> np.int16(3) + np.float64(3)
  np.float64(6.0)

In the majority of cases, this should not lead to surprises.
However, especially if you work with non-default datatypes like low
precision integers or floats or unsigned integers there are some details
of NumPy promotion rules which may be relevant to know.
These detailed rules also do not always match those of many other languages.[#hist-reasons]_

For Numerical values, you can think of promotion happening in three kinds:
``unsigned integers < signed integers < float < complex``,
where the result will always be the highest kind of any of the inputs.
Further, the result will always have a precision higher or equivalent than
any of the inputs which leads to two some examples which may be unexpected:
1. When mixing floating point numbers and integers, the integer ``dtype`` may
   force the result to a higher precision floating point.
2. When mixing unsigned and signed integers, the result will be a higher
   precision than both inputs.  And an unfortunate surprise is that
   ``int64`` and ``uint64`` can return floating point values.

Please see the `Numerical promotion` section and image below for details
on both.

Detailed behavior of Python scalars
-----------------------------------
Since NumPy 2.0,[#NEP50]_ an important case in our promotion rules is that while
NumPy usually never loses precision, it explicitly allows this for the Python
numerical scalars of type ``int``, ``float``, and ``complex``.

Python integers have arbitrary precision, and float and complex numbers
have the same precision as NumPy's `float64` and `complex128`.
However, unlike NumPy arrays and scalars, they do not have an explicit
``dtype`` attached.  Because of this, NumPy assigns them the "kind" but
ignores their actual precision when working with arrays of a lower precision
dtype you do not expect this to change the result:

  >>> arr_float32 = np.array([1, 2.5, 2.1], dtype="float32")
  >>> arr_float32 + 10.0
  array([4. , 5.5, 5.1], dtype=float32)
  >>> arr_int16 = np.array([3, 5, 7], dtype="int16")
  >>> arr_int16 + 10
  array([13, 15, 17], dtype=int16)

In both cases the result precision is dictated only by the NumPy dtype.
And because of that, the ``arr_float32 + 3.0`` behaves the same as
``arr_float32 + np.float32(3.0)`` and ``arr_int16 + 10`` is found via
``arr_int16 + np.int16(10.)``.
Although, when mixing NumPy integers and Python ``float`` or ``complex``
the result always gives the default ``float64`` or ``complex128``:

  >> np.int16(1) + 1.0
  np.float64(2.0)

Users should be aware that while the above is typically convenient, it can
also lead to surprising behaviors when working with low precision data types.

First, since the Python value is converted to a NumPy one, operations can
fail with an error when the result seems obvious.
``np.int8(1) + 1000`` cannot reasonably return an ``int8`` result when the
``1000`` cannot be stored as an ``int8`` at all.
In this case, an error is raised for integers:

  >>> np.int8(1) + 1000
  Traceback (most recent call last):
    ...
  OverflowError: Python integer 1000 out of bounds for int8
  >>> np.int64(1) * 10**100
  Traceback (most recent call last):
  ...
  OverflowError: Python int too large to convert to C long

And overflows are possible for floating points:

  >>> np.float32(1) + 1e300
  RuntimeWarning: overflow encountered in cast
  np.float32(inf)

Second, since the Python float or integer precision is always ignored, a low
precision NumPy scalar will keep using it's lower precision unless explicitly
converted to a Python scalar via ``int()``, ``float()``, or ``scalar.item()``
or to a higher precision NumPy dtype.
This lower precision may be detrimental to some calculations or lead to
incorrect results especially for integer overflows:

  >>> np.int8(100) + 100
  RuntimeWarning: overflow encountered in scalar add
  np.int8(-56)

Overflows and gives an unexpected result.  NumPy gives a warning for scalars
but not for arrays: ``np.array(100, dtype="uint8") + 100`` will *not* warn.

Numerical promotion
-------------------

The following images shows the numerical promotion rules with the kinds
on the vertical axes and the precision on the horizontal one.

.. figure:: figures/nep-0050-promotion-no-fonts.svg
    :figclass: align-center

Promotion is always found along the lines in the schema from left to right
and down to up.
With the following specific rules or observations:
1. When a Python ``float`` or ``complex`` interacts with a NumPy integer
   the result will be ``float64`` or ``complex128`` (yellow border).
   NumPy booleans will also be cast to the default integer.[#default-int]
2. The precision is drawn such that ``float16 < int16 < uint16`` because
   large ``uint16`` do not fit ``int16`` and large ``int16`` will lose precision
   when stored in a ``float16``.
   This pattern however is broken since NumPy always considers ``float64``
   and ``complex128`` to be acceptable promotion results for any integer
   value.
3. A special case the above leads to is that NumPy will promote many
   combinations of signed and unsigned integers to ``float64`` because no
   integer dtype can hold both inputs.
   This can unfortunately be 

The precision here comes from the bit size of the numerical value but
an ``int32`` cannot always be stored in a ``float32`` without loss of
precision, this leads to the 


Notable or special promotion behaviors
--------------------------------------

In NumPy promotion refers to what specific functions do with the result and
in some cases, this means that NumPy may deviate from what the `np.result_type`
would give.

Behavior of ``sum`` and ``prod``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**``np.sum`` and ``np.prod``:** Will alway return the default integer type
when summing over integer values (or booleans).  This is usually an ``int64``.
The reason for this is that integer summations are otherwise very likely
to overflow and give confusing results.
This rule also applies to the underlying ``np.add.reduce``, ``np.multiply.reduce``
and other reduction methods.

Notable behavior with NumPy or Python integer scalars
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NumPy promotion refers to the result dtype and operation precision,
but the operation will sometimes dictate that result.
Division always returns floating point values and comparison always booleans.

This leads to some special behaviors which may appear as "exceptions" to
the rules:
* NumPy comparisons with Python integers or mixed precision integers always
  return the correct result.  The inputs will never be cast in a way which
  loses precision.
* Equality comparisons between types which cannot be promoted will be
  considered all ``False`` (equality) or all ``True`` (not-equal).
* Unary math functions like ``np.sin`` that always return floating point
  values, accept any Python integer input by converting it to ``float64``.
* Division always returns floating point values and thus also allows divisions
  between any NumPy integer with any Python integer value by casting both
  to ``float64``.

It should be noted that while these exceptions can apply to more functions,
NumPy may choose to not implement them in all such cases.

Promotion of non-numerical datatypes
------------------------------------

NumPy extends the promotion to non-numerical types, although in most cases
promotion is not well defined and simply rejected.

The following rules apply:
* NumPy byte strings (``np.bytes_``) can be promoted to unicode strings
  (``np.str_``).  A ``result_type``
* For some purposes NumPy will promote almost any other datatype to strings.
  This applies to array creation or concatenation.
* The array constructers like ``np.array()`` will use ``object`` dtype when
  there is no viable promotion.
* Structured dtypes can promote when their field names and order matches.
  In that case all fields are promoted individually.
* NumPy ``timedelta`` can in some cases promote with integers.

.. note::
    Some of these rules describe the state as of NumPy 2.0 and details
    may be surprising and are under consideration to be change in the future.
    However, changes always have to be weighed against backwards compatibility
    concerns.

Details of promoted ``dtype`` instances
---------------------------------------
The above promotion mainly refers to the behavior when mixing different DType
classes.
A ``dtype`` instance attached to an array can carry additional information
such as byte-order, metadata, string length, or exact structured dtype layout.

While the string length or field names of a structured dtype are important,
NumPy considers byte-order, metadata, and the exact layout of a structured
dtype as storage details.
During promotion NumPy does *not* take these storage details into account:
* Byte-order is converted to native byte-order.
* Metadata attached to the dtype may or may not be preserved.
* Resulting structured dtypes will be packed (but aligned if inputs were).

This behaviors is the best behavior for most programs where storage details
are not relevant to the final results, but the use of incorrect byte-order
could drastically slow down evaluation.


.. [#hist-reasons]: To a large degree, this may just be for choices made early
   on in NumPy's predecessors even.  You may find some more details also in
   `NEP 50 <NEP50>`.

.. [#NEP50]: See also `NEP 50 <NEP50>` which changed the rules for NumPy 2.0.
   previous versions of NumPy would sometimes return higher precision results
   based on the input value of Python scalars.
   Further, previous versions of NumPy would typically ignore the higher
   precision of NumPy scalars or 0-D arrays for promotion purposes.

.. [#default-int]: The default integer is marked as ``int64`` in the schema
   but is ``int32`` on 32bit platforms.  However, normal PCs are 64bit.
