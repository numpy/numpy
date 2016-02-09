#ifndef _NPY_ARRAYITERATORS_H_
#define _NPY_ARRAYITERATORS_H_

/*
 * Parses an index that has no fancy indexing. Populates
 * out_dimensions, out_strides, and out_offset.
 */
NPY_NO_EXPORT int
parse_index(PyArrayObject *self, PyObject *op,
            npy_intp *out_dimensions,
            npy_intp *out_strides,
            npy_intp *out_offset,
            int check_index);

NPY_NO_EXPORT PyObject
*iter_subscript(PyArrayIterObject *, PyObject *);

NPY_NO_EXPORT int
iter_ass_subscript(PyArrayIterObject *, PyObject *, PyObject *);

#endif
