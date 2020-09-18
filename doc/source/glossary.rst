********
Glossary
********

.. glossary::


   (`n`,)
       A tuple with one element. The trailing comma distinguishes a one-element
       tuple from a parenthesized ``n``.


   -1
       Used as a dimension entry, ``-1`` instructs NumPy to choose the length
       that will keep the total number of elements the same.


   ``...``
       An :py:data:`Ellipsis`

       **When indexing an array**, shorthand that the missing axes, if they
       exist, are full slices.

           >>> a = np.arange(24).reshape(2,3,4)

           >>> a[...].shape
           (2, 3, 4)

           >>> a[...,0].shape
           (2, 3)

           >>> a[0,...].shape
           (3, 4)

           >>> a[0,...,0].shape
           (3,)

       It can be used at most once; ``a[...,0,...]`` raises an :exc:`IndexError`.

       **In printouts**, NumPy substitutes ``...`` for the middle elements of
       large arrays. To see the entire array, use `numpy.printoptions`


   ``:``
       The Python :term:`python:slice`
       operator. In ndarrays, slicing can be applied to every
       axis:

           >>> a = np.arange(24).reshape(2,3,4)
           >>> a
           array([[[ 0,  1,  2,  3],
                   [ 4,  5,  6,  7],
                   [ 8,  9, 10, 11]],
           <BLANKLINE>
                  [[12, 13, 14, 15],
                   [16, 17, 18, 19],
                   [20, 21, 22, 23]]])
           <BLANKLINE>
           >>> a[1:,-2:,:-1]
           array([[[16, 17, 18],
                   [20, 21, 22]]])

       Trailing slices can be omitted: ::

           >>> a[1] == a[1,:,:]
           array([[ True,  True,  True,  True],
                  [ True,  True,  True,  True],
                  [ True,  True,  True,  True]])

       In contrast to Python, where slicing creates a copy, in NumPy slicing
       creates a :term:`view`.

       For details, see :ref:`combining-advanced-and-basic-indexing`.


   ``<``
       In a dtype declaration, indicates that the data is
       :term:`little-endian` (the bracket is big on the right). ::

           >>> dt = np.dtype('<f')  # little-endian single-precision float


   ``>``
       In a dtype declaration, indicates that the data is
       :term:`big-endian` (the bracket is big on the left). ::

           >>> dt = np.dtype('>H')  # big-endian unsigned short


   advanced indexing
       Rather than using a :doc:`scalar <reference/arrays.scalars>` or slice as
       an index, an axis can be indexed with an array, providing fine-grained
       selection. This is known as :ref:`advanced indexing<advanced-indexing>`
       or "fancy indexing".


   along an axis
       Axes are defined for arrays with more than one dimension.  A
       2-dimensional array has two corresponding axes: the first running
       vertically downwards across rows (axis 0), and the second running
       horizontally across columns (axis 1).

       Many operations can take place along one of these axes.  For example,
       we can sum each row of an array, in which case we operate along
       columns, or axis 1::

         >>> x = np.arange(12).reshape((3,4))

         >>> x
         array([[ 0,  1,  2,  3],
                [ 4,  5,  6,  7],
                [ 8,  9, 10, 11]])

         >>> x.sum(axis=1)
         array([ 6, 22, 38])


   array
       A homogeneous container of numerical elements.  Each element in the
       array occupies a fixed amount of memory (hence homogeneous), and
       can be a numerical element of a single type (such as float, int
       or complex) or a combination (such as ``(float, int, float)``).  Each
       array has an associated data-type (or ``dtype``), which describes
       the numerical type of its elements::

         >>> x = np.array([1, 2, 3], float)

         >>> x
         array([ 1.,  2.,  3.])

         >>> x.dtype # floating point number, 64 bits of memory per element
         dtype('float64')


         # More complicated data type: each array element is a combination of
         # and integer and a floating point number
         >>> np.array([(1, 2.0), (3, 4.0)], dtype=[('x', np.int64), ('y', float)])
         array([(1, 2.), (3, 4.)], dtype=[('x', '<i8'), ('y', '<f8')])

       Fast element-wise operations, called a :term:`ufunc`, operate on arrays.


   array_like
       Any sequence that can be interpreted as an ndarray.  This includes
       nested lists, tuples, scalars and existing arrays.


   array scalar
       For uniformity in handling operands, NumPy treats
       a :doc:`scalar <reference/arrays.scalars>` as an array of zero
       dimension.


   axis

       Another term for an array dimension. Axes are numbered left to right;
       axis 0 is the first element in the shape tuple.

       In a two-dimensional vector, the elements of axis 0 are rows and the
       elements of axis 1 are columns.

       In higher dimensions, the picture changes. NumPy prints
       higher-dimensional vectors as replications of row-by-column building
       blocks, as in this three-dimensional vector:

           >>> a = np.arange(12).reshape(2,2,3)
           >>> a
           array([[[ 0,  1,  2],
                   [ 3,  4,  5]],
           <BLANKLINE>
                  [[ 6,  7,  8],
                   [ 9, 10, 11]]])

       ``a`` is depicted as a two-element array whose elements are 2x3 vectors.
       From this point of view, rows and columns are the final two axes,
       respectively, in any shape.

       This rule helps you anticipate how a vector will be printed, and
       conversely how to find the index of any of the printed elements. For
       instance, in the example, the last two values of 8's index must be 0 and
       2. Since 8 appears in the second of the two 2x3's, the first index must
       be 1:

           >>> a[1,0,2]
           8

       A convenient way to count dimensions in a printed vector is to
       count ``[`` symbols after the open-parenthesis. This is
       useful in distinguishing, say, a (1,2,3) shape from a (2,3) shape:

           >>> a = np.arange(6).reshape(2,3)
           >>> a.ndim
           2
           >>> a
           array([[0, 1, 2],
                  [3, 4, 5]])

           >>> a = np.arange(6).reshape(1,2,3)
           >>> a.ndim
           3
           >>> a
           array([[[0, 1, 2],
                   [3, 4, 5]]])


   .base

       If an array does not own its memory, then its
       :doc:`base <reference/generated/numpy.ndarray.base>` attribute
       returns the object whose memory the array is referencing. That object
       may be borrowing the memory from still another object, so the
       owning object may be ``a.base.base.base...``. Despite advice to the
       contrary, testing ``base`` is not a surefire way to determine if two
       arrays are :term:`view`\ s.


   big-endian
       When storing a multi-byte value in memory as a sequence of bytes, the
       sequence addresses/sends/stores the most significant byte first (lowest
       address) and the least significant byte last (highest address). Common in
       micro-processors and used for transmission of data over network protocols.


   BLAS
       `Basic Linear Algebra Subprograms <https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms>`_


   broadcast
       NumPy can do operations on arrays whose shapes are mismatched::

         >>> x = np.array([1, 2])
         >>> y = np.array([[3], [4]])

         >>> x
         array([1, 2])

         >>> y
         array([[3],
                [4]])

         >>> x + y
         array([[4, 5],
                [5, 6]])

       See `basics.broadcasting` for more information.


   C order
       See `row-major`


   column-major
       A way to represent items in a N-dimensional array in the 1-dimensional
       computer memory. In column-major order, the leftmost index "varies the
       fastest": for example the array::

            [[1, 2, 3],
             [4, 5, 6]]

       is represented in the column-major order as::

           [1, 4, 2, 5, 3, 6]

       Column-major order is also known as the Fortran order, as the Fortran
       programming language uses it.

   copy

       See :term:`view`.


   decorator
       An operator that transforms a function.  For example, a ``log``
       decorator may be defined to print debugging information upon
       function execution::

         >>> def log(f):
         ...     def new_logging_func(*args, **kwargs):
         ...         print("Logging call with parameters:", args, kwargs)
         ...         return f(*args, **kwargs)
         ...
         ...     return new_logging_func

       Now, when we define a function, we can "decorate" it using ``log``::

         >>> @log
         ... def add(a, b):
         ...     return a + b

       Calling ``add`` then yields:

       >>> add(1, 2)
       Logging call with parameters: (1, 2) {}
       3


   dictionary
       Resembling a language dictionary, which provides a mapping between
       words and descriptions thereof, a Python dictionary is a mapping
       between two objects::

         >>> x = {1: 'one', 'two': [1, 2]}

       Here, `x` is a dictionary mapping keys to values, in this case
       the integer 1 to the string "one", and the string "two" to
       the list ``[1, 2]``.  The values may be accessed using their
       corresponding keys::

         >>> x[1]
         'one'

         >>> x['two']
         [1, 2]

       Note that dictionaries are not stored in any specific order.  Also,
       most mutable (see *immutable* below) objects, such as lists, may not
       be used as keys.

       For more information on dictionaries, read the
       `Python tutorial <https://docs.python.org/tutorial/>`_.


   dimension

       See :term:`axis`.


   dtype

       The datatype describing the (identically typed) elements in an ndarray.
       It can be changed to reinterpret the array contents. For details, see
       :doc:`Data type objects (dtype). <reference/arrays.dtypes>`


   fancy indexing

       Another term for :term:`advanced indexing`.


   field
       In a :term:`structured data type`, each sub-type is called a `field`.
       The `field` has a name (a string), a type (any valid dtype), and
       an optional `title`. See :ref:`arrays.dtypes`


   Fortran order
       See `column-major`


   flattened
       Collapsed to a one-dimensional array. See `numpy.ndarray.flatten`
       for details.


   homogeneous
       Describes a block of memory comprised of blocks, each block comprised of
       items and of the same size, and blocks are interpreted in exactly the
       same way. In the simplest case each block contains a single item, for
       instance int32 or float64.


   immutable
       An object that cannot be modified after execution is called
       immutable.  Two common examples are strings and tuples.


   itemsize
       The size of the dtype element in bytes.


   list
       A Python container that can hold any number of objects or items.
       The items do not have to be of the same type, and can even be
       lists themselves::

         >>> x = [2, 2.0, "two", [2, 2.0]]

       The list `x` contains 4 items, each which can be accessed individually::

         >>> x[2] # the string 'two'
         'two'

         >>> x[3] # a list, containing an integer 2 and a float 2.0
         [2, 2.0]

       It is also possible to select more than one item at a time,
       using *slicing*::

         >>> x[0:2] # or, equivalently, x[:2]
         [2, 2.0]

       In code, arrays are often conveniently expressed as nested lists::


         >>> np.array([[1, 2], [3, 4]])
         array([[1, 2],
                [3, 4]])

       For more information, read the section on lists in the `Python
       tutorial <https://docs.python.org/tutorial/>`_.  For a mapping
       type (key-value), see *dictionary*.


   little-endian
       When storing a multi-byte value in memory as a sequence of bytes, the
       sequence addresses/sends/stores the least significant byte first (lowest
       address) and the most significant byte last (highest address). Common in
       x86 processors.


   mask
       A boolean array, used to select only certain elements for an operation::

         >>> x = np.arange(5)
         >>> x
         array([0, 1, 2, 3, 4])

         >>> mask = (x > 2)
         >>> mask
         array([False, False, False, True,  True])

         >>> x[mask] = -1
         >>> x
         array([ 0,  1,  2,  -1, -1])


   masked array
       Array that suppressed values indicated by a mask::

         >>> x = np.ma.masked_array([np.nan, 2, np.nan], [True, False, True])
         >>> x
         masked_array(data=[--, 2.0, --],
                      mask=[ True, False,  True],
                fill_value=1e+20)

         >>> x + [1, 2, 3]
         masked_array(data=[--, 4.0, --],
                      mask=[ True, False,  True],
                fill_value=1e+20)


       Masked arrays are often used when operating on arrays containing
       missing or invalid entries.


   matrix
       A 2-dimensional ndarray that preserves its two-dimensional nature
       throughout operations.  It has certain special operations, such as ``*``
       (matrix multiplication) and ``**`` (matrix power), defined::

         >>> x = np.mat([[1, 2], [3, 4]])
         >>> x
         matrix([[1, 2],
                 [3, 4]])

         >>> x**2
         matrix([[ 7, 10],
               [15, 22]])


   ndarray
       See *array*.


   object array

       An array whose dtype is ``object``; that is, it contains references to
       Python objects. Indexing the array dereferences the Python objects, so
       unlike other ndarrays, an object array has the ability to hold
       heterogeneous objects.


   ravel

       `numpy.ravel` and `numpy.ndarray.flatten` both flatten an ndarray. ``ravel``
       will return a view if possible; ``flatten`` always returns a copy.

       Flattening collapses a multi-dimensional array to a single dimension;
       details of how this is done (for instance, whether ``a[n+1]`` should be
       the next row or next column) are parameters.


   record array
       An :term:`ndarray` with :term:`structured data type` which has been
       subclassed as ``np.recarray`` and whose dtype is of type ``np.record``,
       making the fields of its data type to be accessible by attribute.


   reference
       If ``a`` is a reference to ``b``, then ``(a is b) == True``.  Therefore,
       ``a`` and ``b`` are different names for the same Python object.


   row-major
       A way to represent items in a N-dimensional array in the 1-dimensional
       computer memory. In row-major order, the rightmost index "varies
       the fastest": for example the array::

            [[1, 2, 3],
             [4, 5, 6]]

       is represented in the row-major order as::

           [1, 2, 3, 4, 5, 6]

       Row-major order is also known as the C order, as the C programming
       language uses it. New NumPy arrays are by default in row-major order.


   slice
       Used to select only certain elements from a sequence:

       >>> x = range(5)
       >>> x
       [0, 1, 2, 3, 4]

       >>> x[1:3] # slice from 1 to 3 (excluding 3 itself)
       [1, 2]

       >>> x[1:5:2] # slice from 1 to 5, but skipping every second element
       [1, 3]

       >>> x[::-1] # slice a sequence in reverse
       [4, 3, 2, 1, 0]

       Arrays may have more than one dimension, each which can be sliced
       individually:

       >>> x = np.array([[1, 2], [3, 4]])
       >>> x
       array([[1, 2],
              [3, 4]])

       >>> x[:, 1]
       array([2, 4])


   stride

       Physical memory is one-dimensional;  strides provide a mechanism to map
       a given index to an address in memory. For an N-dimensional array, its
       ``strides`` attribute is an N-element tuple; advancing from index
       ``i`` to index ``i+1`` on axis ``n`` means adding ``a.strides[n]`` bytes
       to the address.

       Strides are computed automatically from an array's dtype and
       shape, but can be directly specified using
       :doc:`as_strided. <reference/generated/numpy.lib.stride_tricks.as_strided>`

       For details, see
       :doc:`numpy.ndarray.strides <reference/generated/numpy.ndarray.strides>`.

       To see how striding underlies the power of NumPy views, see
       `The NumPy array: a structure for efficient numerical computation. \
       <https://arxiv.org/pdf/1102.1523.pdf>`_


   structure
       See :term:`structured data type`


   structured array

       Array whose :term:`dtype` is a :term:`structured data type`.


   structured data type
       A data type composed of other datatypes


   subarray data type
       A :term:`structured data type` may contain a :term:`ndarray` with its
       own dtype and shape:

       >>> dt = np.dtype([('a', np.int32), ('b', np.float32, (3,))])
       >>> np.zeros(3, dtype=dt)
       array([(0, [0., 0., 0.]), (0, [0., 0., 0.]), (0, [0., 0., 0.])],
             dtype=[('a', '<i4'), ('b', '<f4', (3,))])


   title
       In addition to field names, structured array fields may have an
       associated :ref:`title <titles>` which is an alias to the name and is
       commonly used for plotting.


   ufunc
       Universal function.  A fast element-wise, :term:`vectorized
       <vectorization>` array operation.  Examples include ``add``, ``sin`` and
       ``logical_or``.


   vectorization
       Optimizing a looping block by specialized code. In a traditional sense,
       vectorization performs the same operation on multiple elements with
       fixed strides between them via specialized hardware. Compilers know how
       to take advantage of well-constructed loops to implement such
       optimizations. NumPy uses :ref:`vectorization <whatis-vectorization>`
       to mean any optimization via specialized code performing the same
       operations on multiple elements, typically achieving speedups by
       avoiding some of the overhead in looking up and converting the elements.


   view
       An array that does not own its data, but refers to another array's
       data instead.  For example, we may create a view that only shows
       every second element of another array::

         >>> x = np.arange(5)
         >>> x
         array([0, 1, 2, 3, 4])

         >>> y = x[::2]
         >>> y
         array([0, 2, 4])

         >>> x[0] = 3 # changing x changes y as well, since y is a view on x
         >>> y
         array([3, 2, 4])


   wrapper
       Python is a high-level (highly abstracted, or English-like) language.
       This abstraction comes at a price in execution speed, and sometimes
       it becomes necessary to use lower level languages to do fast
       computations.  A wrapper is code that provides a bridge between
       high and the low level languages, allowing, e.g., Python to execute
       code written in C or Fortran.

       Examples include ctypes, SWIG and Cython (which wraps C and C++)
       and f2py (which wraps Fortran).

