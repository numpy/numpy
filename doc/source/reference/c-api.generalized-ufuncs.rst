.. _c-api.generalized-ufuncs:

==================================
Generalized Universal Function API
==================================

There is a general need for looping over not only functions on scalars
but also over functions on vectors (or arrays).
This concept is realized in NumPy by generalizing the universal functions
(ufuncs).  In regular ufuncs, the elementary function is limited to
element-by-element operations, whereas the generalized version (gufuncs)
supports "sub-array" by "sub-array" operations.  The Perl vector library PDL
provides a similar functionality and its terms are re-used in the following.

Each generalized ufunc has information associated with it that states
what the "core" dimensionality of the inputs is, as well as the
corresponding dimensionality of the outputs (the element-wise ufuncs
have zero core dimensions).  The list of the core dimensions for all
arguments is called the "signature" of a ufunc.  For example, the
ufunc numpy.add has signature ``(),()->()`` defining two scalar inputs
and one scalar output.

As examples of operation requiring more detail, consider the inner (dot) and
outer (cross) products of vectors.  The inner product is easily defined for any
vector dimension, and the corresponding function ``inner1d(a, b)`` [#]_ has a
signature of ``(i),(i)->()``.  This applies the inner product along the last
axis of each input, but keeps the remaining indices intact.  For example, where
``a`` is of shape ``(3, 5, N)`` and ``b`` is of shape ``(5, N)``, this will
return an output of shape ``(3, 5)``.  The underlying elementary function is
called ``3 * 5`` times.  In the signature, we specify one core dimension
``(i)`` for each input and zero core dimensions ``()`` for the output, since
the elementary function takes two 1-d arrays and returns a scalar.  By using
the same name ``i``, we specify that the two corresponding dimensions should be
of the same size.

Turning now to the cross product, that function is less easily defined for
arbitrary dimensions.  But for vectors with three components, it returns a
vector with three components, so a function ``cross1d(a, b)`` specialized for
those would have signature ``(3),(3)->(3)``, indicating that both inputs and
outputs have the same core dimension of fixed size ``3``.

The dimensions beyond the core dimensions are called "loop" dimensions.  In
the example for the inner product, this corresponds to ``(3, 5)``.

The signature determines how the dimensions of each input/output array are
split into core and loop dimensions:

#. Each dimension in the signature is matched to a dimension of the
   corresponding passed-in array, starting from the end of the shape tuple.
   These are the core dimensions, and, unless specifically indicated (see
   below), they must be present in the arrays, or an error will be raised.
#. Core dimensions assigned to the same label in the signature (e.g. the
   ``i`` in ``inner1d``'s ``(i),(i)->()``) must have exactly matching sizes,
   no broadcasting is performed.
#. The core dimensions are removed from all inputs and the remaining
   dimensions are broadcast together, defining the loop dimensions.
#. The shape of each output is determined from the loop dimensions plus the
   output's core dimensions

Typically, the size of all core dimensions in an output will be determined by
the size of a core dimension with the same label in an input array. This is
not a requirement, and it is possible to define a signature where a label
comes up for the first time in an output, although some precautions must be
taken when calling such a function. An example would be the function
``euclidean_pdist(a)``, with signature ``(n,d)->(p)``, that given an array of
``n`` ``d``-dimensional vectors, computes all unique pairwise Euclidean
distances among them. The output dimension ``p`` must therefore be equal to
``n * (n - 1) / 2``, but it is the caller's responsibility to pass in an
output array of the right size. If the size of a core dimension of an output
cannot be determined from a passed in input or output array, an error will be
raised.

.. note: Prior to NumPy 1.10.0, less strict checks were in place:
    missing core dimensions were created by prepending 1's to the
    shape as necessary, core dimensions with the same label were
    broadcast together, and undetermined dimensions were created with
    size 1.  Such behaviour now has to be specifically enabled.

.. [#] The source code for the functions mentioned in this text can be found
       under ``numpy/core/src/umath/_umath.tests.c.src``.

Definitions
-----------

Elementary Function
    Each ufunc consists of an elementary function that performs the
    most basic operation on the smallest portion of array arguments
    (e.g. adding two numbers is the most basic operation in adding two
    arrays).  The ufunc applies the elementary function multiple times
    on different parts of the arrays.  The input/output of elementary
    functions can be arrays; e.g., the elementary function of ``inner1d``
    takes two vectors as input.

Signature
    A signature is a string describing the input/output dimensions of
    the elementary function of a ufunc.  See section below for more
    details.

Core Dimension
    The dimensionality of each input/output of an elementary function
    is defined by its core dimensions (zero core dimensions correspond
    to a scalar input/output).  The core dimensions are mapped to the
    last dimensions of the input/output arrays.

Dimension Name
    A dimension name or size representing a core dimension in the signature.
    Different dimensions may share a name, indicating that they are of
    the same size.

Dimension Index
    A dimension index is an integer representing a dimension name or size.
    It enumerates the distinct dimensions according to the order of the first
    occurrence of each name or size in the signature.

.. _details-of-signature:

Details of Signature
--------------------

The signature defines "core" dimensionality of input and output
variables, and thereby also defines the contraction of the
dimensions.  The signature is represented by a string of the
following format:

* Core dimensions of each input or output array are represented by a
  list of dimension names in parentheses, ``(i_1,...,i_N)``; a scalar
  input/output is denoted by ``()``.  Instead of ``i_1``, ``i_2``,
  etc, one can use any valid Python variable name or a numerical size.
  Names or sizes can have ``?`` or ``|1`` appended to indicate they migth
  not be present or can be broadcast (see below).
* Dimension lists for different arguments are separated by ``","``.
  Input/output arguments are separated by ``"->"``.
* If one uses the same dimension name in multiple locations, this
  enforces the same size of the corresponding dimensions.

The formal syntax of signatures is as follows::

    <Signature>            ::= <Input arguments> "->" <Output arguments>
    <Input arguments>      ::= <Argument list>
    <Output arguments>     ::= <Argument list>
    <Argument list>        ::= nil | <Argument> | <Argument> "," <Argument list>
    <Argument>             ::= "(" <Core dimension list> ")"
    <Core dimension list>  ::= nil | <Core dimension> |
                               <Core dimension> "," <Core dimension list>
    <Core dimension>       ::= <Dimension name> <Dimension modifier>
    <Dimension name>       ::= valid Python variable name | valid integer
    <Dimension modifier>   ::= nil | "|1" | "?"

Notes:

#. All quotes are for clarity.
#. Unmodified core dimensions that share the same name must have the same size.
   Each dimension name typically corresponds to one level of looping in the
   elementary function's implementation.
#. White spaces are ignored.
#. An integer as a dimension name freezes that dimension to the value.
#. If a name if suffixed with the "|1" modifier, it is allowed to broadcast
   against other dimensions with the same name.  All input dimensions
   must share this modifier, while no output dimensions should have it.
#. If the name is suffixed with the "?" modifier, the dimension is a core
   dimension only if it exists on all inputs and outputs that share it;
   otherwise it is ignored (and replaced by a dimension of size 1 for the
   elementary function).

Here are some examples of signatures:

+----------------------------+-----------------------------------+
| Signature                  | Possible use                      |
+----------------------------+-----------------------------------+
| ``(),()->()``              | Addition                          |
+----------------------------+-----------------------------------+
| ``(i)->()``                | Sum over last axis                |
+----------------------------+-----------------------------------+
| ``(i|1),(i|1)->()``        | Test for equality along axis,     |
|                            | allowing comparison with a scalar |
+----------------------------+-----------------------------------+
| ``(i),(i)->()``            | inner vector product              |
+----------------------------+-----------------------------------+
| ``(m,n),(n,p)->(m,p)``     | matrix multiplication             |
+----------------------------+-----------------------------------+
| ``(n),(n,p)->(p)``         | vector-matrix multiplication      |
+----------------------------+-----------------------------------+
| ``(m,n),(n)->(m)``         | matrix-vector multiplication      |
+----------------------------+-----------------------------------+
| ``(m?,n),(n,p?)->(m?,p?)`` | all four of the above at once,    |
|                            | except vectors cannot have loop   |
|                            | dimensions (ie, like ``matmul``)  |
+----------------------------+-----------------------------------+
| ``(3),(3)->(3)``           | cross product for 3-vectors       |
+----------------------------+-----------------------------------+
| ``(i,t),(j,t)->(i,j)``     | inner over the last dimension,    |
|                            | outer over the second to last,    |
|                            | and loop/broadcast over the rest. |
+----------------------------+-----------------------------------+

C-API for implementing Elementary Functions
-------------------------------------------

For scalar elementary functions, one can use ``PyUFunc_FromFuncAndData``.
For a generalized ufunc with a signature, one should use
``PyUFunc_FromFuncAndDataAndSignature``.  The argument lists of the two
are the apart from a final argument specifying the signature as C string.

The functions implementing the elementary operations take the same arguments::

    ``void (*foo)(char **args, intp *dimensions, intp *steps, void *func)``.

When invoked, ``args`` is a list of length ``nargs`` containing
the data of all input/output arguments.  For a scalar elementary
function, ``steps`` is also of length ``nargs``, denoting the strides used
for the arguments. ``dimensions`` is a pointer to a single integer
defining the size of the axis to be looped over.

For a non-trivial signature, ``dimensions`` and ``steps`` provide information
on the core dimensions. The sizes of the core dimensions are stored in
``dimensions``, starting at the second entry.  Only one size is provided for
each distinct dimension name and the sizes are given following the order of
occurrence in the signature. The strides of all core dimensions are given in
``steps``, starting at entry ``nargs``, for all arguments in order.

For example, consider a ufunc with signature ``(i,j),(i)->()``.  In
this case, ``args`` will contain three pointers to the data of the
input/output arrays ``a``, ``b``, ``c``.  Furthermore, ``dimensions`` will be
``[N, I, J]`` to define the size of ``N`` of the loop and the sizes ``I`` and
``J`` for the core dimensions ``i`` and ``j``.  Finally, ``steps`` will be
``[a_N, b_N, c_N, a_i, a_j, b_i]``, containing all necessary strides.
