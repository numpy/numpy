********
Glossary
********

.. glossary::


   (`n`,)
       A parenthesized number followed by a comma denotes a tuple with one
       element. The trailing comma distinguishes a one-element tuple from a
       parenthesized ``n``.


   -1
       - **In a dimension entry**, instructs NumPy to choose the length
         that will keep the total number of array elements the same.

           >>> np.arange(12).reshape(4, -1).shape
           (4, 3)

       - **In an index**, any negative value
         `denotes <https://docs.python.org/dev/faq/programming.html#what-s-a-negative-index>`_
         indexing from the right.

   . . .
       An :py:data:`Ellipsis`.

       - **When indexing an array**, shorthand that the missing axes, if they
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

       - **In printouts**, NumPy substitutes ``...`` for the middle elements of
         large arrays. To see the entire array, use `numpy.printoptions`


   :
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


   <
       In a dtype declaration, indicates that the data is
       :term:`little-endian` (the bracket is big on the right). ::

           >>> dt = np.dtype('<f')  # little-endian single-precision float


   >
       In a dtype declaration, indicates that the data is
       :term:`big-endian` (the bracket is big on the left). ::

           >>> dt = np.dtype('>H')  # big-endian unsigned short


   advanced indexing
       Rather than using a :doc:`scalar <reference/arrays.scalars>` or slice as
       an index, an axis can be indexed with an array, providing fine-grained
       selection. This is known as :ref:`advanced indexing<advanced-indexing>`
       or "fancy indexing".


   along an axis
       An operation `along axis n` of array ``a`` behaves as if its argument
       were an array of slices of ``a`` where each slice has a successive
       index of axis `n`.

       For example, if ``a`` is a 3 x `N` array, an operation along axis 0
       behaves as if its argument were an array containing slices of each row:

           >>> np.array((a[0,:], a[1,:], a[2,:])) #doctest: +SKIP

       To make it concrete, we can pick the operation to be the array-reversal
       function :func:`numpy.flip`, which accepts an ``axis`` argument. We
       construct a 3 x 4 array ``a``:

           >>> a = np.arange(12).reshape(3,4)
           >>> a
           array([[ 0,  1,  2,  3],
                  [ 4,  5,  6,  7],
                  [ 8,  9, 10, 11]])

       Reversing along axis 0 (the row axis) yields

           >>> np.flip(a,axis=0)
           array([[ 8,  9, 10, 11],
                  [ 4,  5,  6,  7],
                  [ 0,  1,  2,  3]])

       Recalling the definition of `along an axis`,  ``flip`` along axis 0 is
       treating its argument as if it were

           >>> np.array((a[0,:], a[1,:], a[2,:]))
           array([[ 0,  1,  2,  3],
                  [ 4,  5,  6,  7],
                  [ 8,  9, 10, 11]])

       and the result of ``np.flip(a,axis=0)`` is to reverse the slices:

           >>> np.array((a[2,:],a[1,:],a[0,:]))
           array([[ 8,  9, 10, 11],
                  [ 4,  5,  6,  7],
                  [ 0,  1,  2,  3]])


   array
       Used synonymously in the NumPy docs with :term:`ndarray`.


   array_like
       Any :doc:`scalar <reference/arrays.scalars>` or
       :term:`python:sequence`
       that can be interpreted as an ndarray.  In addition to ndarrays
       and scalars this category includes lists (possibly nested and with
       different element types) and tuples. Any argument accepted by
       :doc:`numpy.array <reference/generated/numpy.array>`
       is array_like. ::

           >>> a = np.array([[1, 2.0], [0, 0], (1+1j, 3.)])

           >>> a
           array([[1.+0.j, 2.+0.j],
                  [0.+0.j, 0.+0.j],
                  [1.+1.j, 3.+0.j]])


   array scalar
       An :doc:`array scalar <reference/arrays.scalars>` is an instance of the types/classes float32, float64,
       etc.. For uniformity in handling operands, NumPy treats a scalar as
       an array of zero dimension. In contrast, a 0-dimensional array is an :doc:`ndarray <reference/arrays.ndarray>` instance
       containing precisely one value.


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
       :doc:`base <reference/generated/numpy.ndarray.base>` attribute returns
       the object whose memory the array is referencing. That object may be
       referencing the memory from still another object, so the owning object
       may be ``a.base.base.base...``. Some writers erroneously claim that
       testing ``base`` determines if arrays are :term:`view`\ s. For the
       correct way, see :func:`numpy.shares_memory`.


   big-endian
       See `Endianness <https://en.wikipedia.org/wiki/Endianness>`_.


   BLAS
       `Basic Linear Algebra Subprograms <https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms>`_


   broadcast
       *broadcasting* is NumPy's ability to process ndarrays of
       different sizes as if all were the same size.

       It permits an elegant do-what-I-mean behavior where, for instance,
       adding a scalar to a vector adds the scalar value to every element.

           >>> a = np.arange(3)
           >>> a
           array([0, 1, 2])

           >>> a + [3, 3, 3]
           array([3, 4, 5])

           >>> a + 3
           array([3, 4, 5])

       Ordinarly, vector operands must all be the same size, because NumPy
       works element by element -- for instance, ``c = a * b`` is ::

           c[0,0,0] = a[0,0,0] * b[0,0,0]
           c[0,0,1] = a[0,0,1] * b[0,0,1]
          ...

       But in certain useful cases, NumPy can duplicate data along "missing"
       axes or "too-short" dimensions so shapes will match. The duplication
       costs no memory or time. For details, see
       :doc:`Broadcasting. <user/basics.broadcasting>`


   C order
       Same as :term:`row-major`.


   column-major
       See `Row- and column-major order <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`_.


   contiguous

       An array is contiguous if:

       - it occupies an unbroken block of memory, and
       - array elements with higher indexes occupy higher addresses (that
         is, no :term:`stride` is negative).

       There are two types of proper-contiguous NumPy arrays:

       - Fortran-contiguous arrays refer to data that is stored column-wise,
         i.e. the indexing of data as stored in memory starts from the
         lowest dimension;
       - C-contiguous, or simply contiguous arrays, refer to data that is
         stored row-wise, i.e. the indexing of data as stored in memory
         starts from the highest dimension.

       For one-dimensional arrays these notions coincide.

       For example, a 2x2 array ``A`` is Fortran-contiguous if its elements are
       stored in memory in the following order::

           A[0,0] A[1,0] A[0,1] A[1,1]

       and C-contiguous if the order is as follows::

           A[0,0] A[0,1] A[1,0] A[1,1]

       To test whether an array is C-contiguous, use the ``.flags.c_contiguous``
       attribute of NumPy arrays.  To test for Fortran contiguity, use the
       ``.flags.f_contiguous`` attribute.


   copy
       See :term:`view`.


   dimension
       See :term:`axis`.


   dtype
       The datatype describing the (identically typed) elements in an ndarray.
       It can be changed to reinterpret the array contents. For details, see
       :doc:`Data type objects (dtype). <reference/arrays.dtypes>`


   fancy indexing
       Another term for :term:`advanced indexing`.


   field
       In a :term:`structured data type`, each subtype is called a `field`.
       The `field` has a name (a string), a type (any valid dtype), and
       an optional `title`. See :ref:`arrays.dtypes`.


   Fortran order
       Same as :term:`column-major`.


   flattened
       See :term:`ravel`.


   homogeneous
       All elements of a homogeneous array have the same type. ndarrays, in
       contrast to Python lists, are homogeneous. The type can be complicated,
       as in a :term:`structured array`, but all elements have that type.

       NumPy `object arrays <#term-object-array>`_, which contain references to
       Python objects, fill the role of heterogeneous arrays.


   itemsize
       The size of the dtype element in bytes.


   little-endian
       See `Endianness <https://en.wikipedia.org/wiki/Endianness>`_.


   mask
       A boolean array used to select only certain elements for an operation:

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
       Bad or missing data can be cleanly ignored by putting it in a masked
       array, which has an internal boolean array indicating invalid
       entries. Operations with masked arrays ignore these entries. ::

         >>> a = np.ma.masked_array([np.nan, 2, np.nan], [True, False, True])
         >>> a
         masked_array(data=[--, 2.0, --],
                      mask=[ True, False,  True],
                fill_value=1e+20)

         >>> a + [1, 2, 3]
         masked_array(data=[--, 4.0, --],
                      mask=[ True, False,  True],
                fill_value=1e+20)

       For details, see :doc:`Masked arrays. <reference/maskedarray>`


   matrix
       NumPy's two-dimensional
       :doc:`matrix class <reference/generated/numpy.matrix>`
       should no longer be used; use regular ndarrays.


   ndarray
      :doc:`NumPy's basic structure <reference/arrays>`.


   object array
       An array whose dtype is ``object``; that is, it contains references to
       Python objects. Indexing the array dereferences the Python objects, so
       unlike other ndarrays, an object array has the ability to hold
       heterogeneous objects.


   ravel
       :doc:`numpy.ravel \
       <reference/generated/numpy.ravel>`
       and :doc:`numpy.flatten \
       <reference/generated/numpy.ndarray.flatten>`
       both flatten an ndarray. ``ravel`` will return a view if possible;
       ``flatten`` always returns a copy.

       Flattening collapses a multidimensional array to a single dimension;
       details of how this is done (for instance, whether ``a[n+1]`` should be
       the next row or next column) are parameters.


   record array
       A :term:`structured array` with allowing access in an attribute style
       (``a.field``) in addition to ``a['field']``. For details, see
       :doc:`numpy.recarray. <reference/generated/numpy.recarray>`


   row-major
       See `Row- and column-major order <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`_.
       NumPy creates arrays in row-major order by default.


   scalar
       In NumPy, usually a synonym for :term:`array scalar`.


   shape
       A tuple showing the length of each dimension of an ndarray. The
       length of the tuple itself is the number of dimensions
       (:doc:`numpy.ndim <reference/generated/numpy.ndarray.ndim>`).
       The product of the tuple elements is the number of elements in the
       array. For details, see
       :doc:`numpy.ndarray.shape <reference/generated/numpy.ndarray.shape>`.


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


   structured array
       Array whose :term:`dtype` is a :term:`structured data type`.


   structured data type
       Users can create arbitrarily complex :term:`dtypes <dtype>`
       that can include other arrays and dtypes. These composite dtypes are called
       :doc:`structured data types. <user/basics.rec>`


   subarray
      An array nested in a :term:`structured data type`, as ``b`` is here:

        >>> dt = np.dtype([('a', np.int32), ('b', np.float32, (3,))])
        >>> np.zeros(3, dtype=dt)
        array([(0, [0., 0., 0.]), (0, [0., 0., 0.]), (0, [0., 0., 0.])],
              dtype=[('a', '<i4'), ('b', '<f4', (3,))])


   subarray data type
       An element of a structured datatype that behaves like an ndarray.


   title
       An alias for a field name in a structured datatype.


   type
       In NumPy, usually a synonym for :term:`dtype`. For the more general
       Python meaning, :term:`see here. <python:type>`


   ufunc
       NumPy's fast element-by-element computation (:term:`vectorization`)
       gives a choice which function gets applied. The general term for the
       function is ``ufunc``, short for ``universal function``. NumPy routines
       have built-in ufuncs, but users can also
       :doc:`write their own. <reference/ufuncs>`


   vectorization
       NumPy hands off array processing to C, where looping and computation are
       much faster than in Python. To exploit this, programmers using NumPy
       eliminate Python loops in favor of array-to-array operations.
       :term:`vectorization` can refer both to the C offloading and to
       structuring NumPy code to leverage it.

   view
       Without touching underlying data, NumPy can make one array appear
       to change its datatype and shape.

       An array created this way is a `view`, and NumPy often exploits the
       performance gain of using a view versus making a new array.

       A potential drawback is that writing to a view can alter the original
       as well. If this is a problem, NumPy instead needs to create a
       physically distinct array -- a `copy`.

       Some NumPy routines always return views, some always return copies, some
       may return one or the other, and for some the choice can be specified.
       Responsibility for managing views and copies falls to the programmer.
       :func:`numpy.shares_memory` will check whether ``b`` is a view of
       ``a``, but an exact answer isn't always feasible, as the documentation
       page explains.

         >>> x = np.arange(5)
         >>> x
         array([0, 1, 2, 3, 4])

         >>> y = x[::2]
         >>> y
         array([0, 2, 4])

         >>> x[0] = 3 # changing x changes y as well, since y is a view on x
         >>> y
         array([3, 2, 4])

