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

Another example is the function ``inner1d(a, b)`` with a signature of
``(i),(i)->()``.  This applies the inner product along the last axis of
each input, but keeps the remaining indices intact.
For example, where ``a`` is of shape ``(3, 5, N)`` and ``b`` is of shape
``(5, N)``, this will return an output of shape ``(3,5)``.
The underlying elementary function is called ``3 * 5`` times.  In the
signature, we specify one core dimension ``(i)`` for each input and zero core
dimensions ``()`` for the output, since it takes two 1-d arrays and
returns a scalar.  By using the same name ``i``, we specify that the two
corresponding dimensions should be of the same size.

The dimensions beyond the core dimensions are called "loop" dimensions.  In
the above example, this corresponds to ``(3, 5)``.

The signature determines how the dimensions of each input/output array are
split into core and loop dimensions:

#. Each dimension in the signature is matched to a dimension of the
   corresponding passed-in array, starting from the end of the shape tuple.
   These are the core dimensions, and they must be present in the arrays, or
   an error will be raised.
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

Note: Prior to NumPy 1.10.0, less strict checks were in place: missing core
dimensions were created by prepending 1's to the shape as necessary, core
dimensions with the same label were broadcast together, and undetermined
dimensions were created with size 1.


Definitions
-----------

Elementary Function
    Each ufunc consists of an elementary function that performs the
    most basic operation on the smallest portion of array arguments
    (e.g. adding two numbers is the most basic operation in adding two
    arrays).  The ufunc applies the elementary function multiple times
    on different parts of the arrays.  The input/output of elementary
    functions can be vectors; e.g., the elementary function of inner1d
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
    A dimension name represents a core dimension in the signature.
    Different dimensions may share a name, indicating that they are of
    the same size.

Dimension Index
    A dimension index is an integer representing a dimension name. It
    enumerates the dimension names according to the order of the first
    occurrence of each name in the signature.


Details of Signature
--------------------

The signature defines "core" dimensionality of input and output
variables, and thereby also defines the contraction of the
dimensions.  The signature is represented by a string of the
following format:

* Core dimensions of each input or output array are represented by a
  list of dimension names in parentheses, ``(i_1,...,i_N)``; a scalar
  input/output is denoted by ``()``.  Instead of ``i_1``, ``i_2``,
  etc, one can use any valid Python variable name.
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
    <Core dimension list>  ::= nil | <Dimension name> |
                               <Dimension name> "," <Core dimension list>
    <Dimension name>       ::= valid Python variable name


Notes:

#. All quotes are for clarity.
#. Core dimensions that share the same name must have the exact same size.
   Each dimension name typically corresponds to one level of looping in the
   elementary function's implementation.
#. White spaces are ignored.

Here are some examples of signatures:

+-------------+------------------------+-----------------------------------+
| add         | ``(),()->()``          |                                   |
+-------------+------------------------+-----------------------------------+
| inner1d     | ``(i),(i)->()``        |                                   |
+-------------+------------------------+-----------------------------------+
| sum1d       | ``(i)->()``            |                                   |
+-------------+------------------------+-----------------------------------+
| dot2d       | ``(m,n),(n,p)->(m,p)`` | matrix multiplication             |
+-------------+------------------------+-----------------------------------+
| outer_inner | ``(i,t),(j,t)->(i,j)`` | inner over the last dimension,    |
|             |                        | outer over the second to last,    |
|             |                        | and loop/broadcast over the rest. |
+-------------+------------------------+-----------------------------------+

C-API for implementing Elementary Functions
-------------------------------------------

The current interface remains unchanged, and ``PyUFunc_FromFuncAndData``
can still be used to implement (specialized) ufuncs, consisting of
scalar elementary functions.

One can use ``PyUFunc_FromFuncAndDataAndSignature`` to declare a more
general ufunc.  The argument list is the same as
``PyUFunc_FromFuncAndData``, with an additional argument specifying the
signature as C string.

Furthermore, the callback function is of the same type as before,
``void (*foo)(char **args, intp *dimensions, intp *steps, void *func)``.
When invoked, ``args`` is a list of length ``nargs`` containing
the data of all input/output arguments.  For a scalar elementary
function, ``steps`` is also of length ``nargs``, denoting the strides used
for the arguments. ``dimensions`` is a pointer to a single integer
defining the size of the axis to be looped over.

For a non-trivial signature, ``dimensions`` will also contain the sizes
of the core dimensions as well, starting at the second entry.  Only
one size is provided for each unique dimension name and the sizes are
given according to the first occurrence of a dimension name in the
signature.

The first ``nargs`` elements of ``steps`` remain the same as for scalar
ufuncs.  The following elements contain the strides of all core
dimensions for all arguments in order.

For example, consider a ufunc with signature ``(i,j),(i)->()``.  In
this case, ``args`` will contain three pointers to the data of the
input/output arrays ``a``, ``b``, ``c``.  Furthermore, ``dimensions`` will be
``[N, I, J]`` to define the size of ``N`` of the loop and the sizes ``I`` and ``J``
for the core dimensions ``i`` and ``j``.  Finally, ``steps`` will be
``[a_N, b_N, c_N, a_i, a_j, b_i]``, containing all necessary strides.
