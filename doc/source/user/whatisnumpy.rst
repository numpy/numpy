**************
What is NumPy?
**************

NumPy is the fundamental package for scientific computing in Python.  It is a
Python library that provides a multidimensional array object, various derived
objects (such as masked arrays and matrices), and an assortment of routines
for fast operations on arrays, including mathematical, logical, shape
manipulation, sorting, I/O, discrete Fourier transform, basic linear algebra,
basic statistical and random simulation, etc., etc., etc.

The core of the NumPy package, however, is the `ndarray` object.  This
encapsulates *n*-dimensional arrays of homogeneous data.  There are several
important differences between NumPy arrays and the standard Python sequences:

- Unlike Python lists, NumPy arrays have a fixed size - changing their size
  *will* create a new array and delete the original.

- Unlike Python tuples, the elements in a NumPy array are all required to be
  the same data type, and thus *will* be the same size in memory.  The
  exception: one can have arrays of (Python, including NumPy) objects, thereby
  allowing for arrays of different sized elements.

- NumPy arrays facilitate advanced mathematical and other types of operations
  on large amounts of data.  Typically, such operations are executed much
  faster and with less code compared to what is possible using Python's
  built-in sequences.

- A growing plethora of scientific and mathematical Python-based packages are
  keyed to using NumPy arrays; though these typically support Python-sequence
  input, they convert such input to NumPy arrays prior to processing, and they
  output NumPy arrays.  In other words, in order to *efficiently* use much
  (perhaps even most) of today's scientific/mathematical Python-based software, just
  knowing how to use Python's built-in sequence types is insufficient - one
  also needs to know how to use NumPy arrays.

The points about sequence size and speed are particularly important in
scientific computing.  For a simple example, consider the case of multiplying
every element in a 1-D sequence with the corresponding element in another
sequence of the same length.  If the data are in two Python lists, ``a`` and
``b``, we could iterate over each element::

  c = []
  for i in range(len(a)):
      c.append(a[i]*b[i])

This gives the right answer, but if ``a`` and ``b`` each contain millions of
numbers, we will be waiting a rather long time fot it.  We could accomplish
the same task much more quickly in C by writing (neglecting variable
declarations and initializations, memory allocation, etc.)

::

  for (i = 0; i < rows; i++): {
    c[i] = a[i]*b[i];
  }

This saves all the overhead involved in interpreting the Python code and
manipulating Python objects, but at the expense of our beloved Python benefits.
Furthermore, the coding work required increases with the dimensionality of our
data. In the case of a 2-D array, for example, the C code (abridged as before)
expands to

::

  for (i = 0; i < rows; i++): {
    for (j = 0; j < columns; j++): {
      c[i][j] = a[i][j]*b[i][j];
    }
  }

NumPy gives us the best of both worlds: such element-by-element operation is
the "default mode" when an `ndarray` is involved, but the element-by-element
operation is speedily executed by pre-compiled C code.  In NumPy

::

  c = a * b

does what the earlier examples do, at near-C speeds, but with the code
simplicity we expect from something based on Python (indeed, the NumPy
idiom is even simpler!)  This last example illustrates two of NumPy's
features which are the basis of much of its power: vectorization and
broadcasting.

Vectorization describes the absence of any *explicit* looping, indexing, etc.,
in the code - these things are taking place, of course, just "behind the
scenes" (in optimized, pre-compiled C code).  Vectorized code has many
advantages, among which are:

- vectorized code is more concise and easier to read

- fewer lines of code generally means fewer bugs

- the code more closely resembles standard mathematical notation (making it
  easier, typically, to correctly code written mathematics)

- vectorization results in more "Pythonic" code (without vectorization, our
  code would still be littered with ``for`` loops; though of course not absent
  in Python, generally speaking, we feel that if we have to use a ``for`` loop
  in Python, we must be doing something wrong!) :-)

Broadcasting, on the other hand, is the term used to describe the *implicit*
element-by-element behavior of operations; generally speaking, in NumPy all
"operations" (i.e., not just arithmetic operations, but logical, bit-wise,
function, etc.) behave in this implicit element-by-element fashion, i.e., they
"broadcast."  Moreover, in the example above, ``a`` and ``b`` could be
multidimensional arrays of the same shape, or a scalar and an array, or even
two arrays of different shapes, as long as the smaller one is "expandable" to
the shape of the larger in such a way that the resulting broadcast is
unambiguous and "makes sense" (the detailed "rules" of broadcasting are
described in `numpy.doc.broadcasting`).

Finally, in tune with the rest of Python, NumPy fully supports an
object-oriented approach, starting, once again, with `ndarray`.
Unexceptionally, `ndarray` is a class, possessing many, *many* attributes,
both method and "other."  Many, if not all, of its attributes duplicate
(indeed, call) functions in the outer-most NumPy namespace, so the programmer
has complete freedom to code in whichever paradigm she prefers and/or that
seems most appropriate to the task at hand.
