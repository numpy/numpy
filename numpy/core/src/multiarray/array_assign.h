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
 * overwritena: If 1, overwrites everything in 'dst', if 0, it
 *              does not overwrite elements in 'dst' which are NA.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
array_assign_scalar(PyArrayObject *dst,
                    PyArray_Descr *src_dtype, char *src_data,
                    PyArrayObject *wheremask,
                    NPY_CASTING casting, npy_bool overwritena);

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
 * overwritena: If 1, overwrites everything in 'dst', if 0, it
 *              does not overwrite elements in 'dst' which are NA.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
array_assign_broadcast(PyArrayObject *dst, PyArrayObject *src,
                    PyArrayObject *wheremask, 
                    NPY_CASTING casting, npy_bool overwritena);

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
 * overwritena: If 1, overwrites everything in 'dst', if 0, it
 *              does not overwrite elements in 'dst' which are NA.
 *
 * Returns 0 on success, -1 on failure.
 */
NPY_NO_EXPORT int
array_assign_flat(PyArrayObject *dst, NPY_ORDER dst_order,
                  PyArrayObject *src, NPY_ORDER src_order,
                  PyArrayObject *wheremask,
                  NPY_CASTING casting, npy_bool overwritena);

#endif
