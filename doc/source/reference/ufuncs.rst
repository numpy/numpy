.. sectionauthor:: adapted from "Guide to Numpy" by Travis E. Oliphant

.. _ufuncs:

************************************
Universal functions (:class:`ufunc`)
************************************

.. note: XXX: section might need to be made more reference-guideish...

.. currentmodule:: numpy

.. index: ufunc, universal function, arithmetic, operation

A universal function (or :term:`ufunc` for short) is a function that
operates on :class:`ndarrays <ndarray>` in an element-by-element fashion,
supporting :ref:`array broadcasting <ufuncs.broadcasting>`, :ref:`type
casting <ufuncs.casting>`, and several other standard features. That
is, a ufunc is a ":term:`vectorized`" wrapper for a function that
takes a fixed number of scalar inputs and produces a fixed number of
scalar outputs.

In Numpy, universal functions are instances of the
:class:`numpy.ufunc` class. Many of the built-in functions are
implemented in compiled C code, but :class:`ufunc` instances can also
be produced using the :func:`frompyfunc` factory function.


.. _ufuncs.broadcasting:

Broadcasting
============

.. index:: broadcasting

Each universal function takes array inputs and produces array outputs
by performing the core function element-wise on the inputs. Standard
broadcasting rules are applied so that inputs not sharing exactly the
same shapes can still be usefully operated on. Broadcasting can be
understood by four rules:

1. All input arrays with :attr:`ndim <ndarray.ndim>` smaller than the
   input array of largest :attr:`ndim <ndarray.ndim>` have 1's
   prepended to their shapes.

2. The size in each dimension of the output shape is the maximum of all
   the input shapes in that dimension.

3. An input can be used in the calculation if its shape in a particular
   dimension either matches the output shape or has value exactly 1.

4. If an input has a dimension size of 1 in its shape, the first data
   entry in that dimension will be used for all calculations along
   that dimension. In other words, the stepping machinery of the
   :term:`ufunc` will simply not step along that dimension when
   otherwise needed (the :term:`stride` will be 0 for that dimension).

Broadcasting is used throughout NumPy to decide how to handle non
equally-shaped arrays; for example all arithmetic operators (``+``,
``-``, ``*``, ...) between :class:`ndarrays <ndarray>` broadcast the
arrays before operation.

.. _arrays.broadcasting.broadcastable:

.. index:: broadcastable

A set of arrays is called ":term:`broadcastable`" to the same shape if
the above rules produce a valid result, *i.e.*, one of the following
is true:

1. The arrays all have exactly the same shape.

2. The arrays all have the same number of dimensions and the length of
   each dimensions is either a common length or 1.

3. The arrays that have too few dimensions can have their shapes prepended
   with a dimension of length 1 to satisfy property 2.

.. admonition:: Example

   If ``a.shape`` is (5,1), ``b.shape`` is (1,6), ``c.shape`` is (6,)
   and d.shape is ``()`` so that d is a scalar, then *a*, *b*, *c*,
   and *d* are all broadcastable to dimension (5,6); and

   - *a* acts like a (5,6) array where ``a[:,0]`` is broadcast to the other
     columns,

   - *b* acts like a (5,6) array where ``b[0,:]`` is broadcast
     to the other rows,

   - *c* acts like a (1,6) array and therefore like a (5,6) array
     where ``c[:]`` is broadcast to every row, and finally,

   - *d* acts like a (5,6) array where the single value is repeated.


.. _ufuncs.output-type:

Output type determination
=========================

The output of the ufunc (and its methods) is not necessarily an
:class:`ndarray`, if all input arguments are not :class:`ndarrays <ndarray>`.

All output arrays will be passed to the :obj:`__array_wrap__`
method of the input (besides :class:`ndarrays <ndarray>`, and scalars)
that defines it **and** has the highest :obj:`__array_priority__` of
any other input to the universal function. The default
:obj:`__array_priority__` of the ndarray is 0.0, and the default
:obj:`__array_priority__` of a subtype is 1.0. Matrices have
:obj:`__array_priority__` equal to 10.0.

The ufuncs can also all take output arguments. The output will be cast
if necessary to the provided output array. If a class with an
:obj:`__array__` method is used for the output, results will be
written to the object returned by :obj:`__array__`. Then, if the class
also has an :obj:`__array_wrap__` method, the returned
:class:`ndarray` result will be passed to that method just before
passing control back to the caller.

Use of internal buffers
=======================

.. index:: buffers

Internally, buffers are used for misaligned data, swapped data, and
data that has to be converted from one data type to another. The size
of the internal buffers is settable on a per-thread basis. There can
be up to :math:`2 (n_{\mathrm{inputs}} + n_{\mathrm{outputs}})`
buffers of the specified size created to handle the data from all the
inputs and outputs of a ufunc. The default size of the buffer is
10,000 elements. Whenever buffer-based calculation would be needed,
but all input arrays are smaller than the buffer size, those
misbehaved or incorrect typed arrays will be copied before the
calculation proceeds. Adjusting the size of the buffer may therefore
alter the speed at which ufunc calculations of various sorts are
completed. A simple interface for setting this variable is accessible
using the function

.. autosummary::
   :toctree: generated/

   setbufsize


Error handling
==============

.. index:: error handling

Universal functions can trip special floating point status registers
in your hardware (such as divide-by-zero). If available on your
platform, these registers will be regularly checked during
calculation. Error handling is controlled on a per-thread basis,
and can be configured using the functions

.. autosummary::
   :toctree: generated/

   seterr
   seterrcall

.. _ufuncs.casting:

Casting Rules
=============

.. index::
   pair: ufunc; casting rules

At the core of every ufunc is a one-dimensional strided loop that
implements the actual function for a specific type combination. When a
ufunc is created, it is given a static list of inner loops and a
corresponding list of type signatures over which the ufunc operates.
The ufunc machinery uses this list to determine which inner loop to
use for a particular case. You can inspect the :attr:`.types
<ufunc.types>` attribute for a particular ufunc to see which type
combinations have a defined inner loop and which output type they
produce (:ref:`character codes <arrays.scalars.character-codes>` are used in
that output for brevity).

Casting must be done on one or more of the inputs whenever the ufunc
does not have a core loop implementation for the input types provided.
If an implementation for the input types cannot be found, then the
algorithm searches for an implementation with a type signature to
which all of the inputs can be cast "safely." The first one it finds
in its internal list of loops is selected and performed with types
cast. Recall that internal copies during ufuncs (even for casting) are
limited to the size of an internal buffer which is user settable.

.. note::

    Universal functions in NumPy are flexible enough to have mixed type
    signatures. Thus, for example, a universal function could be defined
    that works with floating point and integer values. See :func:`ldexp`
    for an example.

By the above description, the casting rules are essentially
implemented by the question of when a data type can be cast "safely"
to another data type. The answer to this question can be determined in
Python with a function call: :func:`can_cast(fromtype, totype)
<can_cast>`. Figure shows the results of this call for my 32-bit
system on the 21 internally supported types. You can generate this
table for your system with code shown in that Figure.

.. admonition:: Figure

    Code segment showing the can cast safely table for a 32-bit system.

    >>> def print_table(ntypes):
    ...     print 'X',
    ...     for char in ntypes: print char,
    ...     print
    ...     for row in ntypes:
    ...         print row,
    ...         for col in ntypes:
    ...             print int(np.can_cast(row, col)),
    ...         print
    >>> print_table(np.typecodes['All'])
    X ? b h i l q p B H I L Q P f d g F D G S U V O
    ? 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    b 0 1 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1
    h 0 0 1 1 1 1 1 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1
    i 0 0 0 1 1 1 1 0 0 0 0 0 0 0 1 1 0 1 1 1 1 1 1
    l 0 0 0 1 1 1 1 0 0 0 0 0 0 0 1 1 0 1 1 1 1 1 1
    q 0 0 0 0 0 1 0 0 0 0 0 0 0 0 1 1 0 1 1 1 1 1 1
    p 0 0 0 1 1 1 1 0 0 0 0 0 0 0 1 1 0 1 1 1 1 1 1
    B 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    H 0 0 0 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    I 0 0 0 0 0 1 0 0 0 1 1 1 1 0 1 1 0 1 1 1 1 1 1
    L 0 0 0 0 0 1 0 0 0 1 1 1 1 0 1 1 0 1 1 1 1 1 1
    Q 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 0 1 1 1 1 1 1
    P 0 0 0 0 0 1 0 0 0 1 1 1 1 0 1 1 0 1 1 1 1 1 1
    f 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1
    d 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 0 1 1 1 1 1 1
    g 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1 1 1 1 1
    F 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1
    D 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1
    G 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1
    S 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1
    U 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1
    V 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1
    O 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1

You should note that, while included in the table for completeness,
the 'S', 'U', and 'V' types cannot be operated on by ufuncs. Also,
note that on a 64-bit system the integer types may have different
sizes resulting in a slightly altered table.

Mixed scalar-array operations use a different set of casting rules
that ensure that a scalar cannot upcast an array unless the scalar is
of a fundamentally different kind of data (*i.e.* under a different
hierachy in the data type hierarchy) than the array.  This rule
enables you to use scalar constants in your code (which as Python
types are interpreted accordingly in ufuncs) without worrying about
whether the precision of the scalar constant will cause upcasting on
your large (small precision) array.


:class:`ufunc`
==============

Optional keyword arguments
--------------------------

All ufuncs take optional keyword arguments. These represent rather
advanced usage and will likely not be used by most users.

.. index::
   pair: ufunc; keyword arguments

*sig*

    Either a data-type, a tuple of data-types, or a special signature
    string indicating the input and output types of a ufunc. This argument
    allows you to specify a specific signature for a the 1-d loop to use
    in the underlying calculation. If the loop specified does not exist
    for the ufunc, then a TypeError is raised. Normally a suitable loop is
    found automatically by comparing the input types with what is
    available and searching for a loop with data-types to which all inputs
    can be cast safely. This key-word argument lets you by-pass that
    search and choose a loop you want. A list of available signatures is
    available in the **types** attribute of the ufunc object.

*extobj*

    a list of length 1, 2, or 3 specifying the ufunc buffer-size, the
    error mode integer, and the error call-back function. Normally, these
    values are looked-up in a thread-specific dictionary. Passing them
    here bypasses that look-up and uses the low-level specification
    provided for the error-mode. This may be useful as an optimization for
    calculations requiring lots of ufuncs on small arrays in a loop.


Attributes
----------

There are some informational attributes that universal functions
possess. None of the attributes can be set.

.. index::
   pair: ufunc; attributes


============  =================================================================
**__doc__**   A docstring for each ufunc. The first part of the docstring is
              dynamically generated from the number of outputs, the name, and
              the number of inputs. The second part of the doc string is
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

Methods
-------

All ufuncs have 4 methods. However, these methods only make sense on
ufuncs that take two input arguments and return one output argument.
Attempting to call these methods on other ufuncs will cause a
:exc:`ValueError` . The reduce-like methods all take an *axis* keyword
and a *dtype* keyword, and the arrays must all have dimension >=
1. The *axis* keyword specifies which axis of the array the reduction
will take place over and may be negative, but must be an integer. The
*dtype* keyword allows you to manage a very common problem that arises
when naively using :ref:`{op}.reduce <ufunc.reduce>`. Sometimes you may
have an array of a certain data type and wish to add up all of its
elements, but the result does not fit into the data type of the
array. This commonly happens if you have an array of single-byte
integers. The *dtype* keyword allows you to alter the data type that the
reduction takes place over (and therefore the type of the
output). Thus, you can ensure that the output is a data type with
large-enough precision to handle your output. The responsibility of
altering the reduce type is mostly up to you. There is one exception:
if no *dtype* is given for a reduction on the "add" or "multiply"
operations, then if the input type is an integer (or boolean) data-
type and smaller than the size of the :class:`int_` data type, it will
be internally upcast to the :class:`int_` (or :class:`uint`) data
type.

.. index::
   pair: ufunc; methods

.. autosummary::
   :toctree: generated/

   ufunc.reduce
   ufunc.accumulate
   ufunc.reduceat
   ufunc.outer


.. warning::

    A reduce-like operation on an array with a data type that has
    range "too small "to handle the result will silently wrap. You
    should use dtype to increase the data type over which reduction
    takes place.


Available ufuncs
================

There are currently more than 60 universal functions defined in
:mod:`numpy` on one or more types, covering a wide variety of
operations. Some of these ufuncs are called automatically on arrays
when the relevant infix notation is used (*e.g.* :func:`add(a, b) <add>`
is called internally when ``a + b`` is written and *a* or *b* is an
:class:`ndarray`). Nonetheless, you may still want to use the ufunc
call in order to use the optional output argument(s) to place the
output(s) in an object (or in objects) of your choice.

Recall that each ufunc operates element-by-element. Therefore, each
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
    divide
    logaddexp
    logaddexp2
    true_divide
    floor_divide
    negative
    power
    remainder
    mod
    fmod
    absolute
    rint
    sign
    conj
    exp
    exp2
    log
    log2
    log10
    expm1
    log1p
    sqrt
    square
    reciprocal
    ones_like

.. tip::

    The optional output arguments can be used to help you save memory
    for large calculations. If your arrays are large, complicated
    expressions can take longer than absolutely necessary due to the
    creation and (later) destruction of temporary calculation
    spaces. For example, the expression ``G=a*b+c`` is equivalent to
    ``t1=A*B; G=T1+C; del t1``. It will be more quickly executed as
    ``G=A*B; add(G,C,G)`` which is the same as ``G=A*B; G+=C``.


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
    deg2rad
    rad2deg

Bit-twiddling functions
-----------------------

These function all need integer arguments and they maniuplate the bit-
pattern of those arguments.

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
    expect). Use the bitwise operators: & and \| instead.

.. autosummary::

    logical_and
    logical_or
    logical_xor
    logical_not

.. warning::

    The Bitwise operators (& and \|) are the proper way to combine
    element-by-element array comparisons. Be sure to understand the
    operator precedence: (a>2) & (a<5) is the proper syntax because a>2 &
    a<5 will result in an error due to the fact that 2 & a is evaluated
    first.

.. autosummary::

    maximum

.. tip::

    The Python function max() will find the maximum over a one-dimensional
    array, but it will do so using a slower sequence interface. The reduce
    method of the maximum ufunc is much faster. Also, the max() method
    will not give answers you might expect for arrays with greater than
    one dimension. The reduce method of minimum also allows you to compute
    a total minimum over an array.

.. autosummary::

    minimum

.. warning::

    the behavior of maximum(a,b) is than that of max(a,b). As a ufunc,
    maximum(a,b) performs an element-by-element comparison of a and b and
    chooses each element of the result according to which element in the
    two arrays is larger. In contrast, max(a,b) treats the objects a and b
    as a whole, looks at the (total) truth value of a>b and uses it to
    return either a or b (as a whole). A similar difference exists between
    minimum(a,b) and min(a,b).


Floating functions
------------------

Recall that all of these functions work element-by-element over an
array, returning an array output. The description details only a
single operation.

.. autosummary::

    isreal
    iscomplex
    isfinite
    isinf
    isnan
    signbit
    copysign
    modf
    ldexp
    frexp
    fmod
    floor
    ceil
    trunc
