.. sectionauthor:: adapted from "Guide to NumPy" by Travis E. Oliphant

.. currentmodule:: numpy

.. _ufuncs:

************************************
Universal functions (:class:`ufunc`)
************************************

.. seealso:: :ref:`ufuncs-basics`

A universal function (or :term:`ufunc` for short) is a function that
operates on :class:`ndarrays <numpy.ndarray>` in an element-by-element fashion,
supporting :ref:`array broadcasting <ufuncs.broadcasting>`, :ref:`type
casting <ufuncs.casting>`, and several other standard features. That
is, a ufunc is a ":term:`vectorized <vectorization>`" wrapper for a function
that takes a fixed number of specific inputs and produces a fixed number of
specific outputs. For detailed information on universal functions, see
:ref:`ufuncs-basics`.

:class:`ufunc`
==============

.. autosummary::
   :toctree: generated/

   numpy.ufunc

.. _ufuncs.kwargs:

Optional keyword arguments
--------------------------

All ufuncs take optional keyword arguments. Most of these represent
advanced usage and will not typically be used.

.. index::
   pair: ufunc; keyword arguments

.. rubric:: *out*

The first output can be provided as either a positional or a keyword
parameter. Keyword 'out' arguments are incompatible with positional
ones.

The 'out' keyword argument is expected to be a tuple with one entry per
output (which can be None for arrays to be allocated by the ufunc).
For ufuncs with a single output, passing a single array (instead of a
tuple holding a single array) is also valid.

If 'out' is None (the default), a uninitialized output array is created,
which will be filled in the ufunc.  At the end, this array is returned
unless it is zero-dimensional, in which case it is converted to a scalar;
this conversion can be avoided by passing in ``out=...``. This can also be 
spelled `out=Ellipsis` if you think that is clearer.

Note that the output is filled only in the places that the broadcast
'where' is True. If 'where' is the scalar True (the default), then this
corresponds to all elements of the output, but in other cases, the
elements not explicitly filled are left with their uninitialized values.

Operations where ufunc input and output operands have memory overlap are
defined to be the same as for equivalent operations where there
is no memory overlap.  Operations affected make temporary copies
as needed to eliminate data dependency.  As detecting these cases
is computationally expensive, a heuristic is used, which may in rare
cases result in needless temporary copies.  For operations where the
data dependency is simple enough for the heuristic to analyze,
temporary copies will not be made even if the arrays overlap, if it
can be deduced copies are not necessary.  As an example,
``np.add(a, b, out=a)`` will not involve copies.

.. rubric:: *where*

Accepts a boolean array which is broadcast together with the operands.
Values of True indicate to calculate the ufunc at that position, values
of False indicate to leave the value in the output alone. This argument
cannot be used for generalized ufuncs as those take non-scalar input.

Note that if an uninitialized return array is created, values of False
will leave those values **uninitialized**.

.. rubric:: *axes*

A list of tuples with indices of axes a generalized ufunc should operate
on. For instance, for a signature of ``(i,j),(j,k)->(i,k)`` appropriate
for matrix multiplication, the base elements are two-dimensional matrices
and these are taken to be stored in the two last axes of each argument.
The corresponding axes keyword would be ``[(-2, -1), (-2, -1), (-2, -1)]``.
For simplicity, for generalized ufuncs that operate on 1-dimensional arrays
(vectors), a single integer is accepted instead of a single-element tuple,
and for generalized ufuncs for which all outputs are scalars, the output
tuples can be omitted.

.. rubric:: *axis*

A single axis over which a generalized ufunc should operate. This is a
short-cut for ufuncs that operate over a single, shared core dimension,
equivalent to passing in ``axes`` with entries of ``(axis,)`` for each
single-core-dimension argument and ``()`` for all others.  For instance,
for a signature ``(i),(i)->()``, it is equivalent to passing in
``axes=[(axis,), (axis,), ()]``.

.. rubric:: *keepdims*

If this is set to `True`, axes which are reduced over will be left in the
result as a dimension with size one, so that the result will broadcast
correctly against the inputs. This option can only be used for generalized
ufuncs that operate on inputs that all have the same number of core
dimensions and with outputs that have no core dimensions, i.e., with
signatures like ``(i),(i)->()`` or ``(m,m)->()``. If used, the location of
the dimensions in the output can be controlled with ``axes`` and ``axis``.

.. rubric:: *casting*

May be 'no', 'equiv', 'safe', 'same_kind', or 'unsafe'.
See :func:`can_cast` for explanations of the parameter values.

Provides a policy for what kind of casting is permitted. For compatibility
with previous versions of NumPy, this defaults to 'unsafe' for numpy < 1.7.
In numpy 1.7 a transition to 'same_kind' was begun where ufuncs produce a
DeprecationWarning for calls which are allowed under the 'unsafe'
rules, but not under the 'same_kind' rules. From numpy 1.10 and
onwards, the default is 'same_kind'.

.. rubric:: *order*

Specifies the calculation iteration order/memory layout of the output array.
Defaults to 'K'. 'C' means the output should be C-contiguous, 'F' means
F-contiguous, 'A' means F-contiguous if the inputs are F-contiguous and
not also not C-contiguous, C-contiguous otherwise, and 'K' means to match
the element ordering of the inputs as closely as possible.

.. rubric:: *dtype*

Overrides the DType of the output arrays the same way as the *signature*.
This should ensure a matching precision of the calculation.  The exact
calculation DTypes chosen may depend on the ufunc and the inputs may be
cast to this DType to perform the calculation.

.. rubric:: *subok*

Defaults to true. If set to false, the output will always be a strict
array, not a subtype.

.. rubric:: *signature*

Either a Dtype, a tuple of DTypes, or a special signature string
indicating the input and output types of a ufunc.

This argument allows the user to specify exact DTypes to be used for the
calculation.  Casting will be used as necessary. The actual DType of the
input arrays is not considered unless ``signature`` is ``None`` for
that array.

When all DTypes are fixed, a specific loop is chosen or an error raised
if no matching loop exists.
If some DTypes are not specified and left ``None``, the behaviour may
depend on the ufunc.
At this time, a list of available signatures is provided by the **types**
attribute of the ufunc.  (This list may be missing DTypes not defined
by NumPy.)

The ``signature`` only specifies the DType class/type.  For example, it
can specify that the operation should be ``datetime64`` or ``float64``
operation.  It does not specify the ``datetime64`` time-unit or the
``float64`` byte-order.

For backwards compatibility this argument can also be provided as *sig*,
although the long form is preferred.  Note that this should not be
confused with the generalized ufunc :ref:`signature <details-of-signature>`
that is stored in the **signature** attribute of the of the ufunc object.


Attributes
----------

There are some informational attributes that universal functions
possess. None of the attributes can be set.

.. index::
   pair: ufunc; attributes


============  =================================================================
**__doc__**   A docstring for each ufunc. The first part of the docstring is
              dynamically generated from the number of outputs, the name, and
              the number of inputs. The second part of the docstring is
              provided at creation time and stored with the ufunc.

**__name__**  The name of the ufunc.
============  =================================================================

.. autosummary::
   :toctree: generated/

   ufunc.nin
   ufunc.nout
   ufunc.nargs
   ufunc.ntypes
   ufunc.types
   ufunc.identity
   ufunc.signature

.. _ufuncs.methods:

Methods
-------

.. index::
   pair: ufunc; methods

.. autosummary::
   :toctree: generated/

   ufunc.reduce
   ufunc.accumulate
   ufunc.reduceat
   ufunc.outer
   ufunc.at


.. warning::

    A reduce-like operation on an array with a data-type that has a
    range "too small" to handle the result will silently wrap. One
    should use `dtype` to increase the size of the data-type over which
    reduction takes place.


Available ufuncs
================

There are currently more than 60 universal functions defined in
:mod:`numpy` on one or more types, covering a wide variety of
operations. Some of these ufuncs are called automatically on arrays
when the relevant infix notation is used (*e.g.*, :func:`add(a, b) <add>`
is called internally when ``a + b`` is written and *a* or *b* is an
:class:`ndarray`). Nevertheless, you may still want to use the ufunc
call in order to use the optional output argument(s) to place the
output(s) in an object (or objects) of your choice.

Recall that each ufunc operates element-by-element. Therefore, each scalar
ufunc will be described as if acting on a set of scalar inputs to
return a set of scalar outputs.

.. note::

    The ufunc still returns its output(s) even if you use the optional
    output argument(s).

Math operations
---------------

.. autosummary::

    add
    subtract
    multiply
    matmul
    divide
    logaddexp
    logaddexp2
    true_divide
    floor_divide
    negative
    positive
    power
    float_power
    remainder
    mod
    fmod
    divmod
    absolute
    fabs
    rint
    sign
    heaviside
    conj
    conjugate
    exp
    exp2
    log
    log2
    log10
    expm1
    log1p
    sqrt
    square
    cbrt
    reciprocal
    gcd
    lcm

.. tip::

    The optional output arguments can be used to help you save memory
    for large calculations. If your arrays are large, complicated
    expressions can take longer than absolutely necessary due to the
    creation and (later) destruction of temporary calculation
    spaces. For example, the expression ``G = A * B + C`` is equivalent to
    ``T1 = A * B; G = T1 + C; del T1``. It will be more quickly executed
    as ``G = A * B; add(G, C, G)`` which is the same as
    ``G = A * B; G += C``.


Trigonometric functions
-----------------------
All trigonometric functions use radians when an angle is called for.
The ratio of degrees to radians is :math:`180^{\circ}/\pi.`

.. autosummary::

    sin
    cos
    tan
    arcsin
    arccos
    arctan
    arctan2
    hypot
    sinh
    cosh
    tanh
    arcsinh
    arccosh
    arctanh
    degrees
    radians
    deg2rad
    rad2deg

Bit-twiddling functions
-----------------------

These function all require integer arguments and they manipulate the
bit-pattern of those arguments.

.. autosummary::

    bitwise_and
    bitwise_or
    bitwise_xor
    invert
    left_shift
    right_shift

Comparison functions
--------------------

.. autosummary::

    greater
    greater_equal
    less
    less_equal
    not_equal
    equal

.. warning::

    Do not use the Python keywords ``and`` and ``or`` to combine
    logical array expressions. These keywords will test the truth
    value of the entire array (not element-by-element as you might
    expect). Use the bitwise operators & and \| instead.

.. autosummary::

    logical_and
    logical_or
    logical_xor
    logical_not

.. warning::

    The bit-wise operators & and \| are the proper way to perform
    element-by-element array comparisons. Be sure you understand the
    operator precedence: ``(a > 2) & (a < 5)`` is the proper syntax because
    ``a > 2 & a < 5`` will result in an error due to the fact that ``2 & a``
    is evaluated first.

.. autosummary::

    maximum

.. tip::

    The Python function ``max()`` will find the maximum over a one-dimensional
    array, but it will do so using a slower sequence interface. The reduce
    method of the maximum ufunc is much faster. Also, the ``max()`` method
    will not give answers you might expect for arrays with greater than
    one dimension. The reduce method of minimum also allows you to compute
    a total minimum over an array.

.. autosummary::

    minimum

.. warning::

    the behavior of ``maximum(a, b)`` is different than that of ``max(a, b)``.
    As a ufunc, ``maximum(a, b)`` performs an element-by-element comparison
    of `a` and `b` and chooses each element of the result according to which
    element in the two arrays is larger. In contrast, ``max(a, b)`` treats
    the objects `a` and `b` as a whole, looks at the (total) truth value of
    ``a > b`` and uses it to return either `a` or `b` (as a whole). A similar
    difference exists between ``minimum(a, b)`` and ``min(a, b)``.

.. autosummary::

    fmax
    fmin

Floating functions
------------------

Recall that all of these functions work element-by-element over an
array, returning an array output. The description details only a
single operation.

.. autosummary::

    isfinite
    isinf
    isnan
    isnat
    fabs
    signbit
    copysign
    nextafter
    spacing
    modf
    ldexp
    frexp
    fmod
    floor
    ceil
    trunc
