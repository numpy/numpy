#ifndef _NPY_PRIVATE__ARRAY_ASSIGN_H_
#define _NPY_PRIVATE__ARRAY_ASSIGN_H_

/*
 * An array assignment function for copying arrays, treating the
 * arrays as flat according to their respective ordering rules.
 * This function makes a temporary copy of 'src' if 'src' and
 * 'dst' overlap, to be able to handle views of the same data with
 * different strides.
 *
 * dst: The destination array.
 * dst_order: The rule for how 'dst' is to be made flat.
 * src: The source array.
 * src_order: The rule for how 'src' is to be made flat.
 * casting: An exception is raised if the copy violates this
 *          casting rule.
 *
 * Returns 0 on success, -1 on failure.
 */
/* Not yet implemented
NPY_NO_EXPORT int
PyArray_AssignArrayAsFlat(PyArrayObject *dst, NPY_ORDER dst_order,
                  PyArrayObject *src, NPY_ORDER src_order,
                  NPY_CASTING casting,
                  npy_bool preservena, npy_bool *preservewhichna);
*/

NPY_NO_EXPORT int
PyArray_AssignArray(PyArrayObject *dst, PyArrayObject *src,
                    PyArrayObject *wheremask,
                    NPY_CASTING casting);

NPY_NO_EXPORT int
PyArray_AssignRawScalar(PyArrayObject *dst,
                        PyArray_Descr *src_dtype, char *src_data,
                        PyArrayObject *wheremask,
                        NPY_CASTING casting);

/******** LOW-LEVEL SCALAR TO ARRAY ASSIGNMENT ********/

/*
 * Assigns the scalar value to every element of the destination raw array.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_assign_scalar(int ndim, npy_intp *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp *dst_strides,
        PyArray_Descr *src_dtype, char *src_data);

/*
 * Assigns the scalar value to every element of the destination raw array
 * where the 'wheremask' value is True.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
raw_array_wheremasked_assign_scalar(int ndim, npy_intp *shape,
        PyArray_Descr *dst_dtype, char *dst_data, npy_intp *dst_strides,
        PyArray_Descr *src_dtype, char *src_data,
        PyArray_Descr *wheremask_dtype, char *wheremask_data,
        npy_intp *wheremask_strides);

/******** LOW-LEVEL ARRAY MANIPULATION HELPERS ********/

/*
 * Internal detail of how much to buffer during array assignments which
 * need it. This is for more complex NA masking operations where masks
 * need to be inverted or combined together.
 */
#define NPY_ARRAY_ASSIGN_BUFFERSIZE 8192

/*
 * Broadcasts strides to match the given dimensions. Can be used,
 * for instance, to set up a raw iteration.
 *
 * 'strides_name' is used to produce an error message if the strides
 * cannot be broadcast.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
broadcast_strides(int ndim, npy_intp *shape,
                int strides_ndim, npy_intp *strides_shape, npy_intp *strides,
                char *strides_name,
                npy_intp *out_strides);

/*
 * Checks whether a data pointer + set of strides refers to a raw
 * array whose elements are all aligned to a given alignment. Returns
 * 1 if data is aligned to alignment or 0 if not.
 * alignment should be a power of two, or may be the sentinel value 0 to mean
 * cannot-be-aligned, in which case 0 (false) is always returned.
 */
NPY_NO_EXPORT int
raw_array_is_aligned(int ndim, npy_intp *shape,
                     char *data, npy_intp *strides, int alignment);

/*
 * Checks if an array is aligned to its "true alignment"
 * given by dtype->alignment.
 */
NPY_NO_EXPORT int
IsAligned(PyArrayObject *ap);

/*
 * Checks if an array is aligned to its "uint alignment"
 * given by npy_uint_alignment(dtype->elsize).
 */
NPY_NO_EXPORT int
IsUintAligned(PyArrayObject *ap);

/* Returns 1 if the arrays have overlapping data, 0 otherwise */
NPY_NO_EXPORT int
arrays_overlap(PyArrayObject *arr1, PyArrayObject *arr2);


#endif
