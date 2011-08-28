#ifndef _NPY_ARRAYITERATORS_H_
#define _NPY_ARRAYITERATORS_H_

/*
 * Parses an index that has no fancy indexing. Populates
 * out_dimensions, out_strides, and out_offset. If out_maskstrides
 * and out_maskoffset aren't NULL, then 'self' must have an NA mask
 * which is used to populate those variables as well.
 */
NPY_NO_EXPORT int
parse_index(PyArrayObject *self, PyObject *op,
            npy_intp *out_dimensions,
            npy_intp *out_strides,
            npy_intp *out_offset,
            npy_intp *out_maskna_strides,
            npy_intp *out_maskna_offset);

NPY_NO_EXPORT PyObject
*iter_subscript(PyArrayIterObject *, PyObject *);

NPY_NO_EXPORT int
iter_ass_subscript(PyArrayIterObject *, PyObject *, PyObject *);

NPY_NO_EXPORT int
slice_GetIndices(PySliceObject *r, intp length,
                 intp *start, intp *stop, intp *step,
                 intp *slicelength);

#endif
