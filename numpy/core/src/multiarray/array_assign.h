#ifndef _NPY_PRIVATE__ARRAY_ASSIGN_H_
#define _NPY_PRIVATE__ARRAY_ASSIGN_H_

/*
 * Assigns a scalar value specified by 'src_dtype' and 'src_data'
 * to elements of 'dst'.
 *
 * dst: The destination array.
 * src_dtype: The data type of the source scalar.
 * src_data: The memory element of the source scalar.
 * wheremask: If non-NULL, a boolean mask specifying where to copy.
 * casting: An exception is raised if the assignment violates this
 *          casting rule.
 * preservena: If 0, overwrites everything in 'dst', if 1, it
 *              preserves elements in 'dst' which are NA.
 * preservewhichna: Must be NULL. When multi-NA support is implemented,
 *                   this will be an array of flags for 'preservena=True',
 *                   indicating which NA payload values to preserve.
 *
 * This function is implemented in array_assign_scalar.c.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
array_assign_scalar(PyArrayObject *dst,
                    PyArray_Descr *src_dtype, char *src_data,
                    PyArrayObject *wheremask,
                    NPY_CASTING casting,
                    npy_bool preservena, npy_bool *preservewhichna);

/*
 * An array assignment function for copying arrays, broadcasting 'src' into
 * 'dst'. This function makes a temporary copy of 'src' if 'src' and
 * 'dst' overlap, to be able to handle views of the same data with
 * different strides.
 *
 * dst: The destination array.
 * src: The source array.
 * wheremask: If non-NULL, a boolean mask specifying where to copy.
 * casting: An exception is raised if the copy violates this
 *          casting rule.
 * preservena: If 0, overwrites everything in 'dst', if 1, it
 *              preserves elements in 'dst' which are NA.
 * preservewhichna: Must be NULL. When multi-NA support is implemented,
 *                   this will be an array of flags for 'preservena=True',
 *                   indicating which NA payload values to preserve.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
array_assign_array(PyArrayObject *dst, PyArrayObject *src,
                    PyArrayObject *wheremask, 
                    NPY_CASTING casting,
                    npy_bool preservena, npy_bool *preservewhichna);

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
 * wheremask: If non-NULL, a boolean mask specifying where to copy.
 * casting: An exception is raised if the copy violates this
 *          casting rule.
 * preservena: If 0, overwrites everything in 'dst', if 1, it
 *              preserves elements in 'dst' which are NA.
 * preservewhichna: Must be NULL. When multi-NA support is implemented,
 *                   this will be an array of flags for 'preservena=True',
 *                   indicating which NA payload values to preserve.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
array_assign_flat(PyArrayObject *dst, NPY_ORDER dst_order,
                  PyArrayObject *src, NPY_ORDER src_order,
                  PyArrayObject *wheremask,
                  NPY_CASTING casting,
                  npy_bool preservena, npy_bool *preservewhichna);



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
 * array which is fully aligned data.
 */
NPY_NO_EXPORT int
raw_array_is_aligned(int ndim, char *data, npy_intp *strides, int alignment);

#endif
