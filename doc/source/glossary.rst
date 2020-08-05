********
Glossary
********

..  glossary::

    (`n`,)

      A tuple with one element. The trailing comma distinguishes a one-element
      tuple from a parenthesized ``n``.


    -1

      Used as a dimension entry, ``-1`` instructs NumPy to choose the length
      that will keep the total number of elements the same.


    ``...``

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

      It can be used at most once; ``a[...,0,...]`` raises an ``IndexError``.

      **In printouts**, NumPy substitutes ``...`` for the middle elements of
      large arrays. To see the entire array, use
      :doc:`numpy.printoptions. <reference/generated/numpy.printoptions>`


    ``:``

      The Python
      `slice <https://docs.python.org/3/glossary.html#term-slice>`_
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

      For details, see :ref:`Combining advanced and basic indexing <combining-advanced-and-basic-indexing>`.


    ``<``

      In a dtype declaration, indicates that the data is
      `little-endian <https://en.wikipedia.org/wiki/Endianness>`_
      (the bracket is big on the right). ::

        >>> dt = np.dtype('<f') # little-endian single-precision float


    ``>``

      In a dtype declaration, indicates that the data is
      `big-endian <https://en.wikipedia.org/wiki/Endianness>`_
      (the bracket is big on the left). ::

        >>> dt = np.dtype('>H') # big-endian unsigned short


    advanced indexing

      Rather than using a :doc:`scalar <reference/arrays.scalars>` or slice as
      an index, an axis can be indexed with an array, providing fine-grained
      selection. This is known as :ref:`advanced indexing<advanced-indexing>`
      or ``fancy indexing``.


    along an axis

      The result of an operation along an :term:`axis` X is an array in which X
      disappears. This can surprise new users expecting the opposite.

      The operation can be visualized this way:

      Imagine a slice of array ``a`` where axis X has a fixed index
      and the other dimensions are left full (``:``).

          >>> a = np.arange(24).reshape(2,3,4)

          >>> a.shape
          (2,3,4)

          >>> a[:,0,:].shape
          (2,4)

      The slice has ``a``'s shape with the X dimension deleted. Saying an
      operation ``op`` is ``performed along X`` means that ``op`` takes as its
      operands slices having every value of X:

          >>> np.sum(a,axis=1) == a[:,0,:] + a[:,1,:] + a[:,2,:]
          array([[ True,  True,  True,  True],
                 [ True,  True,  True,  True]])


    array

      Used synonymously in the NumPy docs with
      :doc:`ndarray <reference/arrays>`, NumPy's basic structure.


    array_like

      Any :doc:`scalar <reference/arrays.scalars>` or
      `sequence <https://docs.python.org/3/glossary.html#term-sequence>`_
      that can be interpreted as an ndarray.  In addition to ndarrays
      and scalars this category includes lists (possibly nested and with
      different element types) and tuples. Any argument accepted by
      :doc:`numpy.array <reference/generated/numpy.array>`
      is array_like. ::

          >>> a = np.array([[1,2.0],[0,0],(1+1j,3.)])

          >>> a
          array([[1.+0.j, 2.+0.j],
                 [0.+0.j, 0.+0.j],
                 [1.+1.j, 3.+0.j]])


    array scalar

      For uniformity in handling operands, NumPy treats
      a :doc:`scalar <reference/arrays.scalars>` as an array of zero
      dimension.


    `attribute <https://docs.python.org/3/glossary.html#term-attribute>`_
      \


    axis

      Another term for an array dimension. Axes are numbered left to right;
      axis 0 is the first element in the shape tuple.

      In a two-dimensional vector, the elements of axis 0 are rows and the
      elements of axis 1 are columns.

      In higher dimensions the picture changes. NumPy prints
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
        >>> a.shape
        (2, 3)
        >>> a
        array([[0, 1, 2],
               [3, 4, 5]])

        >>> a = np.arange(6).reshape(1,2,3)
        >>> a.shape
        (1, 2, 3)
        >>> a
        array([[[0, 1, 2],
                [3, 4, 5]]])


    .base

      If an array does not own its memory, then its
      :doc:`base <reference/generated/numpy.ndarray.base>` attribute
      returns the object whose memory the array is referencing. That object
      may may be borrowing the memory from still another object, so the
      owning object may be ``a.base.base.base...``. Despite advice to the
      contrary, testing ``base`` is not a surefire way to determine if two
      arrays are :term:`view`\ s.


    `big-endian <https://en.wikipedia.org/wiki/Endianness>`_
       \


    `BLAS <https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms>`_
       \


    broadcast

      ``broadcasting`` is NumPy's ability to process ndarrays of
      different sizes as if all were the same size.

      When NumPy operates on two arrays, it works element by
      element -- for instance, ``c = a * b`` is ::

          c[0,0,0] = a[0,0,0] * b[0,0,0]
          c[0,0,1] = a[0,0,1] * b[0,0,1]
          ...

      Ordinarily this means the shapes of a and b must be identical. But in
      some cases, NumPy can fill "missing" axes or "too-short" dimensions
      with duplicate data so shapes will match. The duplication costs
      no memory or time. For details, see :doc:`Broadcasting. <user/basics.broadcasting>`


    C order

      Same as `row-major. <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`_


    `column-major <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`_
      \


    copy

      See :term:`view`.


    `decorator <https://docs.python.org/3/glossary.html#term-decorator>`_
       \


    `dictionary <https://docs.python.org/3/glossary.html#term-dictionary>`_
       \


    dimension

      See :term:`axis`.


    dtype

      The datatype describing the (identically typed) elements in an ndarray.
      It can be changed to reinterpret the array contents. For details, see
      :doc:`Data type objects (dtype). <reference/arrays.dtypes>`


    fancy indexing

       Another term for :term:`advanced indexing`.


    field

       In a :term:`structured data type`, each subtype is called a
       :doc:`field <reference/generated/numpy.dtype.fields>`.
       A field has a name (a string), a type (any valid dtype), and
       an optional :term:`title`. For details, see :ref:`arrays.dtypes`.


    Fortran order

       Same as `column-major <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`_


    flattened

       See :term:`ravel`.


    homogeneous

      All elements of a homogeneous array have the same type. ndarrays, in
      contrast to Python lists, are homogeneous. The type can be complicated,
      as in a :term:`structured array`, but all elements have that type.

      NumPy `object arrays <#term-object-array>`_, which contain references to
      Python objects, fill the role of heterogeneous arrays.


    `immutable <https://docs.python.org/3/glossary.html#term-immutable>`_
       \


    `iterable <https://docs.python.org/3/glossary.html#term-iterable>`_
      \


    itemsize

       The size of the dtype element in bytes.


    `list <https://docs.python.org/3/glossary.html#term-list>`_
       \


    `little-endian <https://en.wikipedia.org/wiki/Endianness>`_
       \


    mask

       The boolean array used to select elements in a :term:`masked array`.


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

       See :term:`array`.


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

      Flattening collapses a multimdimensional array to a single dimension;
      details of how this is done (for instance, whether ``a[n+1]`` should be
      the next row or next column) are parameters.


    record array

       A :term:`structured array` with an additional way to access
       fields -- ``a.field`` in addition to ``a['field']``. For details, see
       :doc:`numpy.recarray. <reference/generated/numpy.recarray>`


    `row-major <https://en.wikipedia.org/wiki/Row-_and_column-major_order>`_
      \

    :doc:`scalar <reference/arrays.scalars>`
      \

    shape

      A tuple showing the length of each dimension of an ndarray. The
      length of the tuple itself is the number of dimensions
      (:doc:`numpy.ndim <reference/generated/numpy.ndarray.ndim>`).
      The product of the tuple elements is the number of elements in the
      array. For details, see
      :doc:`numpy.ndarray.shape <reference/generated/numpy.ndarray.shape>`.


    :term:`slice <:>`
      \


    stride

      Physical memory is one-dimensional; ``stride`` maps an index in an
      N-dimensional ndarray to an address in memory. For an N-dimensional
      array, stride is an N-element tuple; advancing from index ``i`` to index
      ``i+1`` on axis ``n`` means adding ``a.strides[n]`` bytes to the
      address.

      Stride is computed automatically from an array's dtype and
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

      An array nested in a :term:`structured data type`: ::

        >>> dt = np.dtype([('a', np.int32), ('b', np.float32, (3,))])
        >>> np.zeros(3, dtype=dt)
        array([(0, [0., 0., 0.]), (0, [0., 0., 0.]), (0, [0., 0., 0.])],
              dtype=[('a', '<i4'), ('b', '<f4', (3,))])


    subarray data type

      An element of a strctured datatype that behaves like an ndarray.

      ..


    title

      An alias for a field name in a structured datatype.


    `tuple <https://docs.python.org/3/glossary.html#term-tuple>`_
      \


    type

      In NumPy, a synonym for :term:`dtype`. For the more general Python
      meaning,
      `see here. <https://docs.python.org/3/glossary.html#term-type>`_


    ufunc

      NumPy's fast element-by-element computation (:term:`vectorization`) is
      structured so as to leave the choice of function open. A function used
      in vectorization is called a ``ufunc``, short for ``universal
      function``. NumPy routines have built-in ufuncs, but users can also
      :doc:`write their own. <reference/ufuncs>`


    vectorization

      NumPy hands off array processing to C, where looping and computation are
      much faster than in Python. To exploit this, programmers using NumPy
      eliminate Python loops in favor of array-to-array operations.
      :term:`vectorization` can refer both to the C offloading and to
      structuring NumPy code to leverage it.


    view

      Without changing underlying data, NumPy can make one array masquerade as
      any number of other arrays with different types, shapes, and even
      content. This is much faster than creating those arrays.

      An array created this way is a ``view``, and the performance gain often
      makes an array created as a view preferable to one created as a new
      array.

      But because a view shares data with the original array, a write in one
      array can affect the other, even though they appear to be different
      arrays. If this is an problem, a view can't be used; the second array
      needs to be physically distinct -- a ``copy``.

      Some NumPy routines always return views, some always return copies, some
      may return one or the other, and for some the choice can be specified.
      Responsiblity for managing views and copies falls to the programmer.
      NumPy reports whether arrays share memory
      :doc:`numpy.shares_memory <reference/generated/numpy.shares_memory>`,
      but an exact answer isn't always feasible; see the link.
