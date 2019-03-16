**************
What is NumPy?
**************

NumPy is the fundamental package for scientific computing in Python.
It is a Python library that provides a multidimensional array object,
various derived objects (such as masked arrays and matrices), and an
assortment of routines for fast operations on arrays, including
mathematical, logical, shape manipulation, sorting, selecting, I/O,
discrete Fourier transforms, basic linear algebra, basic statistical
operations, random simulation and much more.

At the core of the NumPy package, is the `ndarray` object.  This
encapsulates *n*-dimensional arrays of homogeneous data types, with
many operations being performed in compiled code for performance.
There are several important differences between NumPy arrays and the
standard Python sequences:

- NumPy arrays have a fixed size at creation, unlike Python lists
  (which can grow dynamically). Changing the size of an `ndarray` will
  create a new array and delete the original.

- The elements in a NumPy array are all required to be of the same
  data type, and thus will be the same size in memory.  The exception:
  one can have arrays of (Python, including NumPy) objects, thereby
  allowing for arrays of different sized elements.

- NumPy arrays facilitate advanced mathematical and other types of
  operations on large numbers of data.  Typically, such operations are
  executed more efficiently and with less code than is possible using
  Python's built-in sequences.

- A growing plethora of scientific and mathematical Python-based
  packages are using NumPy arrays; though these typically support
  Python-sequence input, they convert such input to NumPy arrays prior
  to processing, and they often output NumPy arrays.  In other words,
  in order to efficiently use much (perhaps even most) of today's
  scientific/mathematical Python-based software, just knowing how to
  use Python's built-in sequence types is insufficient - one also
  needs to know how to use NumPy arrays.

The points about sequence size and speed are particularly important in
scientific computing.  As a simple example, consider the case of
multiplying each element in a 1-D sequence with the corresponding
element in another sequence of the same length.  If the data are
stored in two Python lists, ``a`` and ``b``, we could iterate over
each element::

  c = []
  for i in range(len(a)):
      c.append(a[i]*b[i])

This produces the correct answer, but if ``a`` and ``b`` each contain
millions of numbers, we will pay the price for the inefficiencies of
looping in Python.  We could accomplish the same task much more
quickly in C by writing (for clarity we neglect variable declarations
and initializations, memory allocation, etc.)

::

  for (i = 0; i < rows; i++): {
    c[i] = a[i]*b[i];
  }

This saves all the overhead involved in interpreting the Python code
and manipulating Python objects, but at the expense of the benefits
gained from coding in Python.  Furthermore, the coding work required
increases with the dimensionality of our data. In the case of a 2-D
array, for example, the C code (abridged as before) expands to

::

  for (i = 0; i < rows; i++): {
    for (j = 0; j < columns; j++): {
      c[i][j] = a[i][j]*b[i][j];
    }
  }

NumPy gives us the best of both worlds: element-by-element operations
are the "default mode" when an `ndarray` is involved, but the
element-by-element operation is speedily executed by pre-compiled C
code.  In NumPy

::

  c = a * b

does what the earlier examples do, at near-C speeds, but with the code
simplicity we expect from something based on Python. Indeed, the NumPy
idiom is even simpler!  This last example illustrates two of NumPy's
features which are the basis of much of its power: vectorization and
broadcasting.

.. _whatis-vectorization:

Why is NumPy Fast?
------------------

Vectorization describes the absence of any explicit looping, indexing,
etc., in the code - these things are taking place, of course, just
"behind the scenes" in optimized, pre-compiled C code.  Vectorized
code has many advantages, among which are:

- vectorized code is more concise and easier to read

- fewer lines of code generally means fewer bugs

- the code more closely resembles standard mathematical notation
  (making it easier, typically, to correctly code mathematical
  constructs)

- vectorization results in more "Pythonic" code. Without
  vectorization, our code would be littered with inefficient and
  difficult to read ``for`` loops.

Broadcasting is the term used to describe the implicit
element-by-element behavior of operations; generally speaking, in
NumPy all operations, not just arithmetic operations, but
logical, bit-wise, functional, etc., behave in this implicit
element-by-element fashion, i.e., they broadcast.  Moreover, in the
example above, ``a`` and ``b`` could be multidimensional arrays of the
same shape, or a scalar and an array, or even two arrays of with
different shapes, provided that the smaller array is "expandable" to
the shape of the larger in such a way that the resulting broadcast is
unambiguous. For detailed "rules" of broadcasting see
`numpy.doc.broadcasting`.

Who Else Uses NumPy?
--------------------

NumPy fully supports an object-oriented approach, starting, once
again, with `ndarray`.  For example, `ndarray` is a class, possessing
numerous methods and attributes.  Many of its methods are mirrored by
functions in the outer-most NumPy namespace, allowing the programmer
to code in whichever paradigm they prefer. This flexibility has allowed the
NumPy array dialect and NumPy `ndarray` class to become the *de-facto* language
of multi-dimensional data interchange used in Python.
