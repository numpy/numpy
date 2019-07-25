Array methods and functions
===========================


Array method alternative API
----------------------------


Conversion
^^^^^^^^^^

.. c:function:: PyObject* PyArray_GetField( \
        PyArrayObject* self, PyArray_Descr* dtype, int offset)

    Equivalent to :meth:`ndarray.getfield<numpy.ndarray.getfield>`
    (*self*, *dtype*, *offset*). This function `steals a reference
    <https://docs.python.org/3/c-api/intro.html?reference-count-details>`_
    to `PyArray_Descr` and returns a new array of the given `dtype` using
    the data in the current array at a specified `offset` in bytes. The
    `offset` plus the itemsize of the new array type must be less than ``self
    ->descr->elsize`` or an error is raised. The same shape and strides
    as the original array are used. Therefore, this function has the
    effect of returning a field from a structured array. But, it can also
    be used to select specific bytes or groups of bytes from any array
    type.

.. c:function:: int PyArray_SetField( \
        PyArrayObject* self, PyArray_Descr* dtype, int offset, PyObject* val)

    Equivalent to :meth:`ndarray.setfield<numpy.ndarray.setfield>` (*self*, *val*, *dtype*, *offset*
    ). Set the field starting at *offset* in bytes and of the given
    *dtype* to *val*. The *offset* plus *dtype* ->elsize must be less
    than *self* ->descr->elsize or an error is raised. Otherwise, the
    *val* argument is converted to an array and copied into the field
    pointed to. If necessary, the elements of *val* are repeated to
    fill the destination array, But, the number of elements in the
    destination must be an integer multiple of the number of elements
    in *val*.

.. c:function:: PyObject* PyArray_Byteswap(PyArrayObject* self, Bool inplace)

    Equivalent to :meth:`ndarray.byteswap<numpy.ndarray.byteswap>` (*self*, *inplace*). Return an array
    whose data area is byteswapped. If *inplace* is non-zero, then do
    the byteswap inplace and return a reference to self. Otherwise,
    create a byteswapped copy and leave self unchanged.

.. c:function:: PyObject* PyArray_NewCopy(PyArrayObject* old, NPY_ORDER order)

    Equivalent to :meth:`ndarray.copy<numpy.ndarray.copy>` (*self*, *fortran*). Make a copy of the
    *old* array. The returned array is always aligned and writeable
    with data interpreted the same as the old array. If *order* is
    :c:data:`NPY_CORDER`, then a C-style contiguous array is returned. If
    *order* is :c:data:`NPY_FORTRANORDER`, then a Fortran-style contiguous
    array is returned. If *order is* :c:data:`NPY_ANYORDER`, then the array
    returned is Fortran-style contiguous only if the old one is;
    otherwise, it is C-style contiguous.

.. c:function:: PyObject* PyArray_ToList(PyArrayObject* self)

    Equivalent to :meth:`ndarray.tolist<numpy.ndarray.tolist>` (*self*). Return a nested Python list
    from *self*.

.. c:function:: PyObject* PyArray_ToString(PyArrayObject* self, NPY_ORDER order)

    Equivalent to :meth:`ndarray.tobytes<numpy.ndarray.tobytes>` (*self*, *order*). Return the bytes
    of this array in a Python string.

.. c:function:: PyObject* PyArray_ToFile( \
        PyArrayObject* self, FILE* fp, char* sep, char* format)

    Write the contents of *self* to the file pointer *fp* in C-style
    contiguous fashion. Write the data as binary bytes if *sep* is the
    string ""or ``NULL``. Otherwise, write the contents of *self* as
    text using the *sep* string as the item separator. Each item will
    be printed to the file.  If the *format* string is not ``NULL`` or
    "", then it is a Python print statement format string showing how
    the items are to be written.

.. c:function:: int PyArray_Dump(PyObject* self, PyObject* file, int protocol)

    Pickle the object in *self* to the given *file* (either a string
    or a Python file object). If *file* is a Python string it is
    considered to be the name of a file which is then opened in binary
    mode. The given *protocol* is used (if *protocol* is negative, or
    the highest available is used). This is a simple wrapper around
    cPickle.dump(*self*, *file*, *protocol*).

.. c:function:: PyObject* PyArray_Dumps(PyObject* self, int protocol)

    Pickle the object in *self* to a Python string and return it. Use
    the Pickle *protocol* provided (or the highest available if
    *protocol* is negative).

.. c:function:: int PyArray_FillWithScalar(PyArrayObject* arr, PyObject* obj)

    Fill the array, *arr*, with the given scalar object, *obj*. The
    object is first converted to the data type of *arr*, and then
    copied into every location. A -1 is returned if an error occurs,
    otherwise 0 is returned.

.. c:function:: PyObject* PyArray_View( \
        PyArrayObject* self, PyArray_Descr* dtype, PyTypeObject *ptype)

    Equivalent to :meth:`ndarray.view<numpy.ndarray.view>` (*self*, *dtype*). Return a new
    view of the array *self* as possibly a different data-type, *dtype*,
    and different array subclass *ptype*.

    If *dtype* is ``NULL``, then the returned array will have the same
    data type as *self*. The new data-type must be consistent with the
    size of *self*. Either the itemsizes must be identical, or *self* must
    be single-segment and the total number of bytes must be the same.
    In the latter case the dimensions of the returned array will be
    altered in the last (or first for Fortran-style contiguous arrays)
    dimension. The data area of the returned array and self is exactly
    the same.


Shape Manipulation
^^^^^^^^^^^^^^^^^^

.. c:function:: PyObject* PyArray_Newshape( \
        PyArrayObject* self, PyArray_Dims* newshape, NPY_ORDER order)

    Result will be a new array (pointing to the same memory location
    as *self* if possible), but having a shape given by *newshape*.
    If the new shape is not compatible with the strides of *self*,
    then a copy of the array with the new specified shape will be
    returned.

.. c:function:: PyObject* PyArray_Reshape(PyArrayObject* self, PyObject* shape)

    Equivalent to :meth:`ndarray.reshape<numpy.ndarray.reshape>` (*self*, *shape*) where *shape* is a
    sequence. Converts *shape* to a :c:type:`PyArray_Dims` structure and
    calls :c:func:`PyArray_Newshape` internally.
    For back-ward compatibility -- Not recommended

.. c:function:: PyObject* PyArray_Squeeze(PyArrayObject* self)

    Equivalent to :meth:`ndarray.squeeze<numpy.ndarray.squeeze>` (*self*). Return a new view of *self*
    with all of the dimensions of length 1 removed from the shape.

.. warning::

    matrix objects are always 2-dimensional. Therefore,
    :c:func:`PyArray_Squeeze` has no effect on arrays of matrix sub-class.

.. c:function:: PyObject* PyArray_SwapAxes(PyArrayObject* self, int a1, int a2)

    Equivalent to :meth:`ndarray.swapaxes<numpy.ndarray.swapaxes>` (*self*, *a1*, *a2*). The returned
    array is a new view of the data in *self* with the given axes,
    *a1* and *a2*, swapped.

.. c:function:: PyObject* PyArray_Resize( \
        PyArrayObject* self, PyArray_Dims* newshape, int refcheck, \
        NPY_ORDER fortran)

    Equivalent to :meth:`ndarray.resize<numpy.ndarray.resize>` (*self*, *newshape*, refcheck
    ``=`` *refcheck*, order= fortran ). This function only works on
    single-segment arrays. It changes the shape of *self* inplace and
    will reallocate the memory for *self* if *newshape* has a
    different total number of elements then the old shape. If
    reallocation is necessary, then *self* must own its data, have
    *self* - ``>base==NULL``, have *self* - ``>weakrefs==NULL``, and
    (unless refcheck is 0) not be referenced by any other array.
    The fortran argument can be :c:data:`NPY_ANYORDER`, :c:data:`NPY_CORDER`,
    or :c:data:`NPY_FORTRANORDER`. It currently has no effect. Eventually
    it could be used to determine how the resize operation should view
    the data when constructing a differently-dimensioned array.
    Returns None on success and NULL on error.

.. c:function:: PyObject* PyArray_Transpose( \
        PyArrayObject* self, PyArray_Dims* permute)

    Equivalent to :meth:`ndarray.transpose<numpy.ndarray.transpose>` (*self*, *permute*). Permute the
    axes of the ndarray object *self* according to the data structure
    *permute* and return the result. If *permute* is ``NULL``, then
    the resulting array has its axes reversed. For example if *self*
    has shape :math:`10\times20\times30`, and *permute* ``.ptr`` is
    (0,2,1) the shape of the result is :math:`10\times30\times20.` If
    *permute* is ``NULL``, the shape of the result is
    :math:`30\times20\times10.`

.. c:function:: PyObject* PyArray_Flatten(PyArrayObject* self, NPY_ORDER order)

    Equivalent to :meth:`ndarray.flatten<numpy.ndarray.flatten>` (*self*, *order*). Return a 1-d copy
    of the array. If *order* is :c:data:`NPY_FORTRANORDER` the elements are
    scanned out in Fortran order (first-dimension varies the
    fastest). If *order* is :c:data:`NPY_CORDER`, the elements of ``self``
    are scanned in C-order (last dimension varies the fastest). If
    *order* :c:data:`NPY_ANYORDER`, then the result of
    :c:func:`PyArray_ISFORTRAN` (*self*) is used to determine which order
    to flatten.

.. c:function:: PyObject* PyArray_Ravel(PyArrayObject* self, NPY_ORDER order)

    Equivalent to *self*.ravel(*order*). Same basic functionality
    as :c:func:`PyArray_Flatten` (*self*, *order*) except if *order* is 0
    and *self* is C-style contiguous, the shape is altered but no copy
    is performed.


Item selection and manipulation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. c:function:: PyObject* PyArray_TakeFrom( \
        PyArrayObject* self, PyObject* indices, int axis, PyArrayObject* ret, \
        NPY_CLIPMODE clipmode)

    Equivalent to :meth:`ndarray.take<numpy.ndarray.take>` (*self*, *indices*, *axis*, *ret*,
    *clipmode*) except *axis* =None in Python is obtained by setting
    *axis* = :c:data:`NPY_MAXDIMS` in C. Extract the items from self
    indicated by the integer-valued *indices* along the given *axis.*
    The clipmode argument can be :c:data:`NPY_RAISE`, :c:data:`NPY_WRAP`, or
    :c:data:`NPY_CLIP` to indicate what to do with out-of-bound indices. The
    *ret* argument can specify an output array rather than having one
    created internally.

.. c:function:: PyObject* PyArray_PutTo( \
        PyArrayObject* self, PyObject* values, PyObject* indices, \
        NPY_CLIPMODE clipmode)

    Equivalent to *self*.put(*values*, *indices*, *clipmode*
    ). Put *values* into *self* at the corresponding (flattened)
    *indices*. If *values* is too small it will be repeated as
    necessary.

.. c:function:: PyObject* PyArray_PutMask( \
        PyArrayObject* self, PyObject* values, PyObject* mask)

    Place the *values* in *self* wherever corresponding positions
    (using a flattened context) in *mask* are true. The *mask* and
    *self* arrays must have the same total number of elements. If
    *values* is too small, it will be repeated as necessary.

.. c:function:: PyObject* PyArray_Repeat( \
        PyArrayObject* self, PyObject* op, int axis)

    Equivalent to :meth:`ndarray.repeat<numpy.ndarray.repeat>` (*self*, *op*, *axis*). Copy the
    elements of *self*, *op* times along the given *axis*. Either
    *op* is a scalar integer or a sequence of length *self*
    ->dimensions[ *axis* ] indicating how many times to repeat each
    item along the axis.

.. c:function:: PyObject* PyArray_Choose( \
        PyArrayObject* self, PyObject* op, PyArrayObject* ret, \
        NPY_CLIPMODE clipmode)

    Equivalent to :meth:`ndarray.choose<numpy.ndarray.choose>` (*self*, *op*, *ret*, *clipmode*).
    Create a new array by selecting elements from the sequence of
    arrays in *op* based on the integer values in *self*. The arrays
    must all be broadcastable to the same shape and the entries in
    *self* should be between 0 and len(*op*). The output is placed
    in *ret* unless it is ``NULL`` in which case a new output is
    created. The :c:data:`NPY_CLIPMODE` argument determines behavior for when
    entries in *self* are not between 0 and len(*op*).

.. c:function:: PyObject* PyArray_Sort(PyArrayObject* self, int axis, NPY_SORTKIND kind)

    Equivalent to :meth:`ndarray.sort<numpy.ndarray.sort>` (*self*, *axis*, *kind*).
    Return an array with the items of *self* sorted along *axis*. The array
    is sorted using the algorithm denoted by *kind* , which is an integer/enum pointing
    to the type of sorting algorithms used.

.. c:function:: PyObject* PyArray_ArgSort(PyArrayObject* self, int axis)

    Equivalent to :meth:`ndarray.argsort<numpy.ndarray.argsort>` (*self*, *axis*).
    Return an array of indices such that selection of these indices
    along the given ``axis`` would return a sorted version of *self*. If *self* ->descr
    is a data-type with fields defined, then self->descr->names is used
    to determine the sort order. A comparison where the first field is equal
    will use the second field and so on. To alter the sort order of a
    structured array, create a new data-type with a different order of names
    and construct a view of the array with that new data-type.

.. c:function:: PyObject* PyArray_LexSort(PyObject* sort_keys, int axis)

    Given a sequence of arrays (*sort_keys*) of the same shape,
    return an array of indices (similar to :c:func:`PyArray_ArgSort` (...))
    that would sort the arrays lexicographically. A lexicographic sort
    specifies that when two keys are found to be equal, the order is
    based on comparison of subsequent keys. A merge sort (which leaves
    equal entries unmoved) is required to be defined for the
    types. The sort is accomplished by sorting the indices first using
    the first *sort_key* and then using the second *sort_key* and so
    forth. This is equivalent to the lexsort(*sort_keys*, *axis*)
    Python command. Because of the way the merge-sort works, be sure
    to understand the order the *sort_keys* must be in (reversed from
    the order you would use when comparing two elements).

    If these arrays are all collected in a structured array, then
    :c:func:`PyArray_Sort` (...) can also be used to sort the array
    directly.

.. c:function:: PyObject* PyArray_SearchSorted( \
        PyArrayObject* self, PyObject* values, NPY_SEARCHSIDE side, \
        PyObject* perm)

    Equivalent to :meth:`ndarray.searchsorted<numpy.ndarray.searchsorted>` (*self*, *values*, *side*,
    *perm*). Assuming *self* is a 1-d array in ascending order, then the
    output is an array of indices the same shape as *values* such that, if
    the elements in *values* were inserted before the indices, the order of
    *self* would be preserved. No checking is done on whether or not self is
    in ascending order.

    The *side* argument indicates whether the index returned should be that of
    the first suitable location (if :c:data:`NPY_SEARCHLEFT`) or of the last
    (if :c:data:`NPY_SEARCHRIGHT`).

    The *sorter* argument, if not ``NULL``, must be a 1D array of integer
    indices the same length as *self*, that sorts it into ascending order.
    This is typically the result of a call to :c:func:`PyArray_ArgSort` (...)
    Binary search is used to find the required insertion points.

.. c:function:: int PyArray_Partition( \
        PyArrayObject *self, PyArrayObject * ktharray, int axis, \
        NPY_SELECTKIND which)

    Equivalent to :meth:`ndarray.partition<numpy.ndarray.partition>` (*self*, *ktharray*, *axis*,
    *kind*). Partitions the array so that the values of the element indexed by
    *ktharray* are in the positions they would be if the array is fully sorted
    and places all elements smaller than the kth before and all elements equal
    or greater after the kth element. The ordering of all elements within the
    partitions is undefined.
    If *self*->descr is a data-type with fields defined, then
    self->descr->names is used to determine the sort order. A comparison where
    the first field is equal will use the second field and so on. To alter the
    sort order of a structured array, create a new data-type with a different
    order of names and construct a view of the array with that new data-type.
    Returns zero on success and -1 on failure.

.. c:function:: PyObject* PyArray_ArgPartition( \
        PyArrayObject *op, PyArrayObject * ktharray, int axis, \
        NPY_SELECTKIND which)

    Equivalent to :meth:`ndarray.argpartition<numpy.ndarray.argpartition>` (*self*, *ktharray*, *axis*,
    *kind*). Return an array of indices such that selection of these indices
    along the given ``axis`` would return a partitioned version of *self*.

.. c:function:: PyObject* PyArray_Diagonal( \
        PyArrayObject* self, int offset, int axis1, int axis2)

    Equivalent to :meth:`ndarray.diagonal<numpy.ndarray.diagonal>` (*self*, *offset*, *axis1*, *axis2*
    ). Return the *offset* diagonals of the 2-d arrays defined by
    *axis1* and *axis2*.

.. c:function:: npy_intp PyArray_CountNonzero(PyArrayObject* self)

    .. versionadded:: 1.6

    Counts the number of non-zero elements in the array object *self*.

.. c:function:: PyObject* PyArray_Nonzero(PyArrayObject* self)

    Equivalent to :meth:`ndarray.nonzero<numpy.ndarray.nonzero>` (*self*). Returns a tuple of index
    arrays that select elements of *self* that are nonzero. If (nd=
    :c:func:`PyArray_NDIM` ( ``self`` ))==1, then a single index array is
    returned. The index arrays have data type :c:data:`NPY_INTP`. If a
    tuple is returned (nd :math:`\neq` 1), then its length is nd.

.. c:function:: PyObject* PyArray_Compress( \
        PyArrayObject* self, PyObject* condition, int axis, PyArrayObject* out)

    Equivalent to :meth:`ndarray.compress<numpy.ndarray.compress>` (*self*, *condition*, *axis*
    ). Return the elements along *axis* corresponding to elements of
    *condition* that are true.


Calculation
^^^^^^^^^^^

.. tip::

    Pass in :c:data:`NPY_MAXDIMS` for axis in order to achieve the same
    effect that is obtained by passing in *axis* = :const:`None` in Python
    (treating the array as a 1-d array).


.. note::

    The out argument specifies where to place the result. If out is
    NULL, then the output array is created, otherwise the output is
    placed in out which must be the correct size and type. A new
    reference to the output array is always returned even when out
    is not NULL. The caller of the routine has the responsibility
    to ``Py_DECREF`` out if not NULL or a memory-leak will occur.


.. c:function:: PyObject* PyArray_ArgMax( \
        PyArrayObject* self, int axis, PyArrayObject* out)

    Equivalent to :meth:`ndarray.argmax<numpy.ndarray.argmax>` (*self*, *axis*). Return the index of
    the largest element of *self* along *axis*.

.. c:function:: PyObject* PyArray_ArgMin( \
        PyArrayObject* self, int axis, PyArrayObject* out)

    Equivalent to :meth:`ndarray.argmin<numpy.ndarray.argmin>` (*self*, *axis*). Return the index of
    the smallest element of *self* along *axis*.

.. c:function:: PyObject* PyArray_Max( \
        PyArrayObject* self, int axis, PyArrayObject* out)

    Equivalent to :meth:`ndarray.max<numpy.ndarray.max>` (*self*, *axis*). Returns the largest
    element of *self* along the given *axis*. When the result is a single
    element, returns a numpy scalar instead of an ndarray.

.. c:function:: PyObject* PyArray_Min( \
        PyArrayObject* self, int axis, PyArrayObject* out)

    Equivalent to :meth:`ndarray.min<numpy.ndarray.min>` (*self*, *axis*). Return the smallest
    element of *self* along the given *axis*. When the result is a single
    element, returns a numpy scalar instead of an ndarray.


.. c:function:: PyObject* PyArray_Ptp( \
        PyArrayObject* self, int axis, PyArrayObject* out)

    Equivalent to :meth:`ndarray.ptp<numpy.ndarray.ptp>` (*self*, *axis*). Return the difference
    between the largest element of *self* along *axis* and the
    smallest element of *self* along *axis*. When the result is a single
    element, returns a numpy scalar instead of an ndarray.




.. note::

    The rtype argument specifies the data-type the reduction should
    take place over. This is important if the data-type of the array
    is not "large" enough to handle the output. By default, all
    integer data-types are made at least as large as :c:data:`NPY_LONG`
    for the "add" and "multiply" ufuncs (which form the basis for
    mean, sum, cumsum, prod, and cumprod functions).

.. c:function:: PyObject* PyArray_Mean( \
        PyArrayObject* self, int axis, int rtype, PyArrayObject* out)

    Equivalent to :meth:`ndarray.mean<numpy.ndarray.mean>` (*self*, *axis*, *rtype*). Returns the
    mean of the elements along the given *axis*, using the enumerated
    type *rtype* as the data type to sum in. Default sum behavior is
    obtained using :c:data:`NPY_NOTYPE` for *rtype*.

.. c:function:: PyObject* PyArray_Trace( \
        PyArrayObject* self, int offset, int axis1, int axis2, int rtype, \
        PyArrayObject* out)

    Equivalent to :meth:`ndarray.trace<numpy.ndarray.trace>` (*self*, *offset*, *axis1*, *axis2*,
    *rtype*). Return the sum (using *rtype* as the data type of
    summation) over the *offset* diagonal elements of the 2-d arrays
    defined by *axis1* and *axis2* variables. A positive offset
    chooses diagonals above the main diagonal. A negative offset
    selects diagonals below the main diagonal.

.. c:function:: PyObject* PyArray_Clip( \
        PyArrayObject* self, PyObject* min, PyObject* max)

    Equivalent to :meth:`ndarray.clip<numpy.ndarray.clip>` (*self*, *min*, *max*). Clip an array,
    *self*, so that values larger than *max* are fixed to *max* and
    values less than *min* are fixed to *min*.

.. c:function:: PyObject* PyArray_Conjugate(PyArrayObject* self)

    Equivalent to :meth:`ndarray.conjugate<numpy.ndarray.conjugate>` (*self*).
    Return the complex conjugate of *self*. If *self* is not of
    complex data type, then return *self* with a reference.

.. c:function:: PyObject* PyArray_Round( \
        PyArrayObject* self, int decimals, PyArrayObject* out)

    Equivalent to :meth:`ndarray.round<numpy.ndarray.round>` (*self*, *decimals*, *out*). Returns
    the array with elements rounded to the nearest decimal place. The
    decimal place is defined as the :math:`10^{-\textrm{decimals}}`
    digit so that negative *decimals* cause rounding to the nearest 10's, 100's, etc. If out is ``NULL``, then the output array is created, otherwise the output is placed in *out* which must be the correct size and type.

.. c:function:: PyObject* PyArray_Std( \
        PyArrayObject* self, int axis, int rtype, PyArrayObject* out)

    Equivalent to :meth:`ndarray.std<numpy.ndarray.std>` (*self*, *axis*, *rtype*). Return the
    standard deviation using data along *axis* converted to data type
    *rtype*.

.. c:function:: PyObject* PyArray_Sum( \
        PyArrayObject* self, int axis, int rtype, PyArrayObject* out)

    Equivalent to :meth:`ndarray.sum<numpy.ndarray.sum>` (*self*, *axis*, *rtype*). Return 1-d
    vector sums of elements in *self* along *axis*. Perform the sum
    after converting data to data type *rtype*.

.. c:function:: PyObject* PyArray_CumSum( \
        PyArrayObject* self, int axis, int rtype, PyArrayObject* out)

    Equivalent to :meth:`ndarray.cumsum<numpy.ndarray.cumsum>` (*self*, *axis*, *rtype*). Return
    cumulative 1-d sums of elements in *self* along *axis*. Perform
    the sum after converting data to data type *rtype*.

.. c:function:: PyObject* PyArray_Prod( \
        PyArrayObject* self, int axis, int rtype, PyArrayObject* out)

    Equivalent to :meth:`ndarray.prod<numpy.ndarray.prod>` (*self*, *axis*, *rtype*). Return 1-d
    products of elements in *self* along *axis*. Perform the product
    after converting data to data type *rtype*.

.. c:function:: PyObject* PyArray_CumProd( \
        PyArrayObject* self, int axis, int rtype, PyArrayObject* out)

    Equivalent to :meth:`ndarray.cumprod<numpy.ndarray.cumprod>` (*self*, *axis*, *rtype*). Return
    1-d cumulative products of elements in ``self`` along ``axis``.
    Perform the product after converting data to data type ``rtype``.

.. c:function:: PyObject* PyArray_All( \
        PyArrayObject* self, int axis, PyArrayObject* out)

    Equivalent to :meth:`ndarray.all<numpy.ndarray.all>` (*self*, *axis*). Return an array with
    True elements for every 1-d sub-array of ``self`` defined by
    ``axis`` in which all the elements are True.

.. c:function:: PyObject* PyArray_Any( \
        PyArrayObject* self, int axis, PyArrayObject* out)

    Equivalent to :meth:`ndarray.any<numpy.ndarray.any>` (*self*, *axis*). Return an array with
    True elements for every 1-d sub-array of *self* defined by *axis*
    in which any of the elements are True.

Functions
---------


Array Functions
^^^^^^^^^^^^^^^

.. c:function:: int PyArray_AsCArray( \
        PyObject** op, void* ptr, npy_intp* dims, int nd, int typenum, \
        int itemsize)

    Sometimes it is useful to access a multidimensional array as a
    C-style multi-dimensional array so that algorithms can be
    implemented using C's a[i][j][k] syntax. This routine returns a
    pointer, *ptr*, that simulates this kind of C-style array, for
    1-, 2-, and 3-d ndarrays.

    :param op:

        The address to any Python object. This Python object will be replaced
        with an equivalent well-behaved, C-style contiguous, ndarray of the
        given data type specified by the last two arguments. Be sure that
        stealing a reference in this way to the input object is justified.

    :param ptr:

        The address to a (ctype* for 1-d, ctype** for 2-d or ctype*** for 3-d)
        variable where ctype is the equivalent C-type for the data type. On
        return, *ptr* will be addressable as a 1-d, 2-d, or 3-d array.

    :param dims:

        An output array that contains the shape of the array object. This
        array gives boundaries on any looping that will take place.

    :param nd:

        The dimensionality of the array (1, 2, or 3).

    :param typenum:

        The expected data type of the array.

    :param itemsize:

        This argument is only needed when *typenum* represents a
        flexible array. Otherwise it should be 0.

.. note::

    The simulation of a C-style array is not complete for 2-d and 3-d
    arrays. For example, the simulated arrays of pointers cannot be passed
    to subroutines expecting specific, statically-defined 2-d and 3-d
    arrays. To pass to functions requiring those kind of inputs, you must
    statically define the required array and copy data.

.. c:function:: int PyArray_Free(PyObject* op, void* ptr)

    Must be called with the same objects and memory locations returned
    from :c:func:`PyArray_AsCArray` (...). This function cleans up memory
    that otherwise would get leaked.

.. c:function:: PyObject* PyArray_Concatenate(PyObject* obj, int axis)

    Join the sequence of objects in *obj* together along *axis* into a
    single array. If the dimensions or types are not compatible an
    error is raised.

.. c:function:: PyObject* PyArray_InnerProduct(PyObject* obj1, PyObject* obj2)

    Compute a product-sum over the last dimensions of *obj1* and
    *obj2*. Neither array is conjugated.

.. c:function:: PyObject* PyArray_MatrixProduct(PyObject* obj1, PyObject* obj)

    Compute a product-sum over the last dimension of *obj1* and the
    second-to-last dimension of *obj2*. For 2-d arrays this is a
    matrix-product. Neither array is conjugated.

.. c:function:: PyObject* PyArray_MatrixProduct2( \
        PyObject* obj1, PyObject* obj, PyArrayObject* out)

    .. versionadded:: 1.6

    Same as PyArray_MatrixProduct, but store the result in *out*.  The
    output array must have the correct shape, type, and be
    C-contiguous, or an exception is raised.

.. c:function:: PyObject* PyArray_EinsteinSum( \
        char* subscripts, npy_intp nop, PyArrayObject** op_in, \
        PyArray_Descr* dtype, NPY_ORDER order, NPY_CASTING casting, \
        PyArrayObject* out)

    .. versionadded:: 1.6

    Applies the Einstein summation convention to the array operands
    provided, returning a new array or placing the result in *out*.
    The string in *subscripts* is a comma separated list of index
    letters. The number of operands is in *nop*, and *op_in* is an
    array containing those operands. The data type of the output can
    be forced with *dtype*, the output order can be forced with *order*
    (:c:data:`NPY_KEEPORDER` is recommended), and when *dtype* is specified,
    *casting* indicates how permissive the data conversion should be.

    See the :func:`~numpy.einsum` function for more details.

.. c:function:: PyObject* PyArray_CopyAndTranspose(PyObject \* op)

    A specialized copy and transpose function that works only for 2-d
    arrays. The returned array is a transposed copy of *op*.

.. c:function:: PyObject* PyArray_Correlate( \
        PyObject* op1, PyObject* op2, int mode)

    Compute the 1-d correlation of the 1-d arrays *op1* and *op2*
    . The correlation is computed at each output point by multiplying
    *op1* by a shifted version of *op2* and summing the result. As a
    result of the shift, needed values outside of the defined range of
    *op1* and *op2* are interpreted as zero. The mode determines how
    many shifts to return: 0 - return only shifts that did not need to
    assume zero- values; 1 - return an object that is the same size as
    *op1*, 2 - return all possible shifts (any overlap at all is
    accepted).

    .. rubric:: Notes

    This does not compute the usual correlation: if op2 is larger than op1, the
    arguments are swapped, and the conjugate is never taken for complex arrays.
    See PyArray_Correlate2 for the usual signal processing correlation.

.. c:function:: PyObject* PyArray_Correlate2( \
        PyObject* op1, PyObject* op2, int mode)

    Updated version of PyArray_Correlate, which uses the usual definition of
    correlation for 1d arrays. The correlation is computed at each output point
    by multiplying *op1* by a shifted version of *op2* and summing the result.
    As a result of the shift, needed values outside of the defined range of
    *op1* and *op2* are interpreted as zero. The mode determines how many
    shifts to return: 0 - return only shifts that did not need to assume zero-
    values; 1 - return an object that is the same size as *op1*, 2 - return all
    possible shifts (any overlap at all is accepted).

    .. rubric:: Notes

    Compute z as follows::

      z[k] = sum_n op1[n] * conj(op2[n+k])

.. c:function:: PyObject* PyArray_Where( \
        PyObject* condition, PyObject* x, PyObject* y)

    If both ``x`` and ``y`` are ``NULL``, then return
    :c:func:`PyArray_Nonzero` (*condition*). Otherwise, both *x* and *y*
    must be given and the object returned is shaped like *condition*
    and has elements of *x* and *y* where *condition* is respectively
    True or False.


Other functions
^^^^^^^^^^^^^^^

.. c:function:: Bool PyArray_CheckStrides( \
        int elsize, int nd, npy_intp numbytes, npy_intp const* dims, \
        npy_intp const* newstrides)

    Determine if *newstrides* is a strides array consistent with the
    memory of an *nd* -dimensional array with shape ``dims`` and
    element-size, *elsize*. The *newstrides* array is checked to see
    if jumping by the provided number of bytes in each direction will
    ever mean jumping more than *numbytes* which is the assumed size
    of the available memory segment. If *numbytes* is 0, then an
    equivalent *numbytes* is computed assuming *nd*, *dims*, and
    *elsize* refer to a single-segment array. Return :c:data:`NPY_TRUE` if
    *newstrides* is acceptable, otherwise return :c:data:`NPY_FALSE`.

.. c:function:: npy_intp PyArray_MultiplyList(npy_intp const* seq, int n)

.. c:function:: int PyArray_MultiplyIntList(int const* seq, int n)

    Both of these routines multiply an *n* -length array, *seq*, of
    integers and return the result. No overflow checking is performed.

.. c:function:: int PyArray_CompareLists(npy_intp const* l1, npy_intp const* l2, int n)

    Given two *n* -length arrays of integers, *l1*, and *l2*, return
    1 if the lists are identical; otherwise, return 0.