Iterators
=========

Array Iterators
---------------

As of NumPy 1.6.0, these array iterators are superceded by
the new array iterator, :c:type:`NpyIter`.

An array iterator is a simple way to access the elements of an
N-dimensional array quickly and efficiently. Section `2
<#sec-array-iterator>`__ provides more description and examples of
this useful approach to looping over an array.

.. c:function:: PyObject* PyArray_IterNew(PyObject* arr)

    Return an array iterator object from the array, *arr*. This is
    equivalent to *arr*. **flat**. The array iterator object makes
    it easy to loop over an N-dimensional non-contiguous array in
    C-style contiguous fashion.

.. c:function:: PyObject* PyArray_IterAllButAxis(PyObject* arr, int \*axis)

    Return an array iterator that will iterate over all axes but the
    one provided in *\*axis*. The returned iterator cannot be used
    with :c:func:`PyArray_ITER_GOTO1D`. This iterator could be used to
    write something similar to what ufuncs do wherein the loop over
    the largest axis is done by a separate sub-routine. If *\*axis* is
    negative then *\*axis* will be set to the axis having the smallest
    stride and that axis will be used.

.. c:function:: PyObject *PyArray_BroadcastToShape( \
        PyObject* arr, npy_intp *dimensions, int nd)

    Return an array iterator that is broadcast to iterate as an array
    of the shape provided by *dimensions* and *nd*.

.. c:function:: int PyArrayIter_Check(PyObject* op)

    Evaluates true if *op* is an array iterator (or instance of a
    subclass of the array iterator type).

.. c:function:: void PyArray_ITER_RESET(PyObject* iterator)

    Reset an *iterator* to the beginning of the array.

.. c:function:: void PyArray_ITER_NEXT(PyObject* iterator)

    Incremement the index and the dataptr members of the *iterator* to
    point to the next element of the array. If the array is not
    (C-style) contiguous, also increment the N-dimensional coordinates
    array.

.. c:function:: void *PyArray_ITER_DATA(PyObject* iterator)

    A pointer to the current element of the array.

.. c:function:: void PyArray_ITER_GOTO( \
        PyObject* iterator, npy_intp* destination)

    Set the *iterator* index, dataptr, and coordinates members to the
    location in the array indicated by the N-dimensional c-array,
    *destination*, which must have size at least *iterator*
    ->nd_m1+1.

.. c:function:: PyArray_ITER_GOTO1D(PyObject* iterator, npy_intp index)

    Set the *iterator* index and dataptr to the location in the array
    indicated by the integer *index* which points to an element in the
    C-styled flattened array.

.. c:function:: int PyArray_ITER_NOTDONE(PyObject* iterator)

    Evaluates TRUE as long as the iterator has not looped through all of
    the elements, otherwise it evaluates FALSE.


Broadcasting (multi-iterators)
------------------------------

.. c:function:: PyObject* PyArray_MultiIterNew(int num, ...)

    A simplified interface to broadcasting. This function takes the
    number of arrays to broadcast and then *num* extra ( :c:type:`PyObject *<PyObject>`
    ) arguments. These arguments are converted to arrays and iterators
    are created. :c:func:`PyArray_Broadcast` is then called on the resulting
    multi-iterator object. The resulting, broadcasted mult-iterator
    object is then returned. A broadcasted operation can then be
    performed using a single loop and using :c:func:`PyArray_MultiIter_NEXT`
    (..)

.. c:function:: void PyArray_MultiIter_RESET(PyObject* multi)

    Reset all the iterators to the beginning in a multi-iterator
    object, *multi*.

.. c:function:: void PyArray_MultiIter_NEXT(PyObject* multi)

    Advance each iterator in a multi-iterator object, *multi*, to its
    next (broadcasted) element.

.. c:function:: void *PyArray_MultiIter_DATA(PyObject* multi, int i)

    Return the data-pointer of the *i* :math:`^{\textrm{th}}` iterator
    in a multi-iterator object.

.. c:function:: void PyArray_MultiIter_NEXTi(PyObject* multi, int i)

    Advance the pointer of only the *i* :math:`^{\textrm{th}}` iterator.

.. c:function:: void PyArray_MultiIter_GOTO( \
        PyObject* multi, npy_intp* destination)

    Advance each iterator in a multi-iterator object, *multi*, to the
    given :math:`N` -dimensional *destination* where :math:`N` is the
    number of dimensions in the broadcasted array.

.. c:function:: void PyArray_MultiIter_GOTO1D(PyObject* multi, npy_intp index)

    Advance each iterator in a multi-iterator object, *multi*, to the
    corresponding location of the *index* into the flattened
    broadcasted array.

.. c:function:: int PyArray_MultiIter_NOTDONE(PyObject* multi)

    Evaluates TRUE as long as the multi-iterator has not looped
    through all of the elements (of the broadcasted result), otherwise
    it evaluates FALSE.

.. c:function:: int PyArray_Broadcast(PyArrayMultiIterObject* mit)

    This function encapsulates the broadcasting rules. The *mit*
    container should already contain iterators for all the arrays that
    need to be broadcast. On return, these iterators will be adjusted
    so that iteration over each simultaneously will accomplish the
    broadcasting. A negative number is returned if an error occurs.

.. c:function:: int PyArray_RemoveSmallest(PyArrayMultiIterObject* mit)

    This function takes a multi-iterator object that has been
    previously "broadcasted," finds the dimension with the smallest
    "sum of strides" in the broadcasted result and adapts all the
    iterators so as not to iterate over that dimension (by effectively
    making them of length-1 in that dimension). The corresponding
    dimension is returned unless *mit* ->nd is 0, then -1 is
    returned. This function is useful for constructing ufunc-like
    routines that broadcast their inputs correctly and then call a
    strided 1-d version of the routine as the inner-loop.  This 1-d
    version is usually optimized for speed and for this reason the
    loop should be performed over the axis that won't require large
    stride jumps.

Neighborhood iterator
---------------------

.. versionadded:: 1.4.0

Neighborhood iterators are subclasses of the iterator object, and can be used
to iter over a neighborhood of a point. For example, you may want to iterate
over every voxel of a 3d image, and for every such voxel, iterate over an
hypercube. Neighborhood iterator automatically handle boundaries, thus making
this kind of code much easier to write than manual boundaries handling, at the
cost of a slight overhead.

.. c:function:: PyObject* PyArray_NeighborhoodIterNew( \
        PyArrayIterObject* iter, npy_intp bounds, int mode, \
        PyArrayObject* fill_value)

    This function creates a new neighborhood iterator from an existing
    iterator.  The neighborhood will be computed relatively to the position
    currently pointed by *iter*, the bounds define the shape of the
    neighborhood iterator, and the mode argument the boundaries handling mode.

    The *bounds* argument is expected to be a (2 * iter->ao->nd) arrays, such
    as the range bound[2*i]->bounds[2*i+1] defines the range where to walk for
    dimension i (both bounds are included in the walked coordinates). The
    bounds should be ordered for each dimension (bounds[2*i] <= bounds[2*i+1]).

    The mode should be one of:

    .. c:macro:: NPY_NEIGHBORHOOD_ITER_ZERO_PADDING
            Zero padding. Outside bounds values will be 0.
    .. c:macro:: NPY_NEIGHBORHOOD_ITER_ONE_PADDING
            One padding, Outside bounds values will be 1.
    .. c:macro:: NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING
            Constant padding. Outside bounds values will be the
            same as the first item in fill_value.
    .. c:macro:: NPY_NEIGHBORHOOD_ITER_MIRROR_PADDING
            Mirror padding. Outside bounds values will be as if the
            array items were mirrored. For example, for the array [1, 2, 3, 4],
            x[-2] will be 2, x[-2] will be 1, x[4] will be 4, x[5] will be 1,
            etc...
    .. c:macro:: NPY_NEIGHBORHOOD_ITER_CIRCULAR_PADDING
            Circular padding. Outside bounds values will be as if the array
            was repeated. For example, for the array [1, 2, 3, 4], x[-2] will
            be 3, x[-2] will be 4, x[4] will be 1, x[5] will be 2, etc...

    If the mode is constant filling (`NPY_NEIGHBORHOOD_ITER_CONSTANT_PADDING`),
    fill_value should point to an array object which holds the filling value
    (the first item will be the filling value if the array contains more than
    one item). For other cases, fill_value may be NULL.

    - The iterator holds a reference to iter
    - Return NULL on failure (in which case the reference count of iter is not
      changed)
    - iter itself can be a Neighborhood iterator: this can be useful for .e.g
      automatic boundaries handling
    - the object returned by this function should be safe to use as a normal
      iterator
    - If the position of iter is changed, any subsequent call to
      PyArrayNeighborhoodIter_Next is undefined behavior, and
      PyArrayNeighborhoodIter_Reset must be called.

    .. code-block:: c

       PyArrayIterObject *iter;
       PyArrayNeighborhoodIterObject *neigh_iter;
       iter = PyArray_IterNew(x);

       /*For a 3x3 kernel */
       bounds = {-1, 1, -1, 1};
       neigh_iter = (PyArrayNeighborhoodIterObject*)PyArrayNeighborhoodIter_New(
            iter, bounds, NPY_NEIGHBORHOOD_ITER_ZERO_PADDING, NULL);

       for(i = 0; i < iter->size; ++i) {
            for (j = 0; j < neigh_iter->size; ++j) {
                    /* Walk around the item currently pointed by iter->dataptr */
                    PyArrayNeighborhoodIter_Next(neigh_iter);
            }

            /* Move to the next point of iter */
            PyArrayIter_Next(iter);
            PyArrayNeighborhoodIter_Reset(neigh_iter);
       }

.. c:function:: int PyArrayNeighborhoodIter_Reset( \
        PyArrayNeighborhoodIterObject* iter)

    Reset the iterator position to the first point of the neighborhood. This
    should be called whenever the iter argument given at
    PyArray_NeighborhoodIterObject is changed (see example)

.. c:function:: int PyArrayNeighborhoodIter_Next( \
        PyArrayNeighborhoodIterObject* iter)

    After this call, iter->dataptr points to the next point of the
    neighborhood. Calling this function after every point of the
    neighborhood has been visited is undefined.