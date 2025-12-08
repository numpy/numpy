.. _c-api.generalized-ufuncs:

==================================
Generalized universal function API
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
ufunc ``numpy.add`` has signature ``(),()->()`` defining two scalar inputs
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
``n * (n - 1) / 2``, but by default, it is the caller's responsibility to pass
in an output array of the right size. If the size of a core dimension of an output
cannot be determined from a passed in input or output array, an error will be
raised.  This can be changed by defining a ``PyUFunc_ProcessCoreDimsFunc`` function
and assigning it to the ``proces_core_dims_func`` field of the ``PyUFuncObject``
structure.  See below for more details.

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
    functions can be vectors; e.g., the elementary function of ``inner1d``
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

.. _details-of-signature:

Details of signature
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
    <Core dimension list>  ::= nil | <Core dimension> |
                               <Core dimension> "," <Core dimension list>
    <Core dimension>       ::= <Dimension name> <Dimension modifier>
    <Dimension name>       ::= valid Python variable name | valid integer
    <Dimension modifier>   ::= nil | "?"

Notes:

#. All quotes are for clarity.
#. Unmodified core dimensions that share the same name must have the same size.
   Each dimension name typically corresponds to one level of looping in the
   elementary function's implementation.
#. White spaces are ignored.
#. An integer as a dimension name freezes that dimension to the value.
#. If the name is suffixed with the "?" modifier, the dimension is a core
   dimension only if it exists on all inputs and outputs that share it;
   otherwise it is ignored (and replaced by a dimension of size 1 for the
   elementary function).

Here are some examples of signatures:

+-------------+----------------------------+-----------------------------------+
| name        | signature                  | common usage                      |
+=============+============================+===================================+
| add         | ``(),()->()``              | binary ufunc                      |
+-------------+----------------------------+-----------------------------------+
| sum1d       | ``(i)->()``                | reduction                         |
+-------------+----------------------------+-----------------------------------+
| inner1d     | ``(i),(i)->()``            | vector-vector multiplication      |
+-------------+----------------------------+-----------------------------------+
| matmat      | ``(m,n),(n,p)->(m,p)``     | matrix multiplication             |
+-------------+----------------------------+-----------------------------------+
| vecmat      | ``(n),(n,p)->(p)``         | vector-matrix multiplication      |
+-------------+----------------------------+-----------------------------------+
| matvec      | ``(m,n),(n)->(m)``         | matrix-vector multiplication      |
+-------------+----------------------------+-----------------------------------+
| matmul      | ``(m?,n),(n,p?)->(m?,p?)`` | combination of the four above     |
+-------------+----------------------------+-----------------------------------+
| outer_inner | ``(i,t),(j,t)->(i,j)``     | inner over the last dimension,    |
|             |                            | outer over the second to last,    |
|             |                            | and loop/broadcast over the rest. |
+-------------+----------------------------+-----------------------------------+
|  cross1d    | ``(3),(3)->(3)``           | cross product where the last      |
|             |                            | dimension is frozen and must be 3 |
+-------------+----------------------------+-----------------------------------+

.. _frozen:

The last is an instance of freezing a core dimension and can be used to
improve ufunc performance

C-API for implementing elementary functions
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

Customizing core dimension size processing
------------------------------------------

The optional function of type ``PyUFunc_ProcessCoreDimsFunc``, stored
on the ``process_core_dims_func`` attribute of the ufunc, provides the
author of the ufunc a "hook" into the processing of the core dimensions
of the arrays that were passed to the ufunc.  The two primary uses of
this "hook" are:

* Check that constraints on the core dimensions required
  by the ufunc are satisfied (and set an exception if they are not).
* Compute output shapes for any output core dimensions that were not
  determined by the input arrays.

As an example of the first use, consider the generalized ufunc ``minmax``
with signature ``(n)->(2)`` that simultaneously computes the minimum and
maximum of a sequence.  It should require that ``n > 0``, because
the minimum and maximum of a sequence with length 0 is not meaningful.
In this case, the ufunc author might define the function like this:

    .. code-block:: c

        int minmax_process_core_dims(PyUFuncObject *ufunc,
                                     npy_intp *core_dim_sizes)
        {
            npy_intp n = core_dim_sizes[0];
            if (n == 0) {
                PyErr_SetString(PyExc_ValueError,
                                "minmax requires the core dimension to "
                                "be at least 1.");
                return -1;
            }
            return 0;
        }

In this case, the length of the array ``core_dim_sizes`` will be 2.
The second value in the array will always be 2, so there is no need
for the function to inspect it.  The core dimension ``n`` is stored
in the first element.  The function sets an exception and returns -1
if it finds that ``n`` is 0.

The second use for the "hook" is to compute the size of output arrays
when the output arrays are not provided by the caller and one or more
core dimension of the output is not also an input core dimension.
If the ufunc does not have a function defined on the
``process_core_dims_func`` attribute, an unspecified output core
dimension size will result in an exception being raised.  With the
"hook" provided by ``process_core_dims_func``, the author of the ufunc
can set the output size to whatever is appropriate for the ufunc.

In the array passed to the "hook" function, core dimensions that
were not determined by the input are indicated by having the value -1
in the ``core_dim_sizes`` array.  The function can replace the -1 with
whatever value is appropriate for the ufunc, based on the core dimensions
that occurred in the input arrays.

.. warning::
    The function must never change a value in ``core_dim_sizes`` that
    is not -1 on input.  Changing a value that was not -1 will generally
    result in incorrect output from the ufunc, and could result in the
    Python interpreter crashing.

For example, consider the generalized ufunc ``conv1d`` for which
the elementary function computes the "full" convolution of two
one-dimensional arrays ``x`` and ``y`` with lengths ``m`` and ``n``,
respectively.  The output of this convolution has length ``m + n - 1``.
To implement this as a generalized ufunc, the signature is set to
``(m),(n)->(p)``, and in the "hook" function, if the core dimension
``p`` is found to be -1, it is replaced with ``m + n - 1``.  If ``p``
is *not* -1, it must be verified that the given value equals ``m + n - 1``.
If it does not, the function must set an exception and return -1.
For a meaningful result, the operation also requires that ``m + n``
is at least 1, i.e. both inputs can't have length 0.

Here's how that might look in code:

    .. code-block:: c

        int conv1d_process_core_dims(PyUFuncObject *ufunc,
                                     npy_intp *core_dim_sizes)
        {
            // core_dim_sizes will hold the core dimensions [m, n, p].
            // p will be -1 if the caller did not provide the out argument.
            npy_intp m = core_dim_sizes[0];
            npy_intp n = core_dim_sizes[1];
            npy_intp p = core_dim_sizes[2];
            npy_intp required_p = m + n - 1;

            if (m == 0 && n == 0) {
                // Disallow both inputs having length 0.
                PyErr_SetString(PyExc_ValueError,
                    "conv1d: both inputs have core dimension 0; the function "
                    "requires that at least one input has size greater than 0.");
                return -1;
            }
            if (p == -1) {
                // Output array was not given in the call of the ufunc.
                // Set the correct output size here.
                core_dim_sizes[2] = required_p;
                return 0;
            }
            // An output array *was* given.  Validate its core dimension.
            if (p != required_p) {
                PyErr_Format(PyExc_ValueError,
                        "conv1d: the core dimension p of the out parameter "
                        "does not equal m + n - 1, where m and n are the "
                        "core dimensions of the inputs x and y; got m=%zd "
                        "and n=%zd so p must be %zd, but got p=%zd.",
                        m, n, required_p, p);
                return -1;
            }
            return 0;
        }
