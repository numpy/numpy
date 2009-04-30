#ifndef _NPY_ARRAYITERATORS_H_
#define _NPY_ARRAYITERATORS_H_

NPY_NO_EXPORT intp
parse_subindex(PyObject *op, intp *step_size, intp *n_steps, intp max);

NPY_NO_EXPORT int
parse_index(PyArrayObject *self, PyObject *op,
            intp *dimensions, intp *strides, intp *offset_ptr);

NPY_NO_EXPORT PyObject
*iter_subscript(PyArrayIterObject *, PyObject *);

NPY_NO_EXPORT int
iter_ass_subscript(PyArrayIterObject *, PyObject *, PyObject *);

NPY_NO_EXPORT int
slice_GetIndices(PySliceObject *r, intp length,
                 intp *start, intp *stop, intp *step,
                 intp *slicelength);

/*
 * Prototypes for Mapping calls --- not part of the C-API
 * because only useful as part of a getitem call.
 */
NPY_NO_EXPORT void
PyArray_MapIterReset(PyArrayMapIterObject *mit);

NPY_NO_EXPORT void
PyArray_MapIterNext(PyArrayMapIterObject *mit);

NPY_NO_EXPORT void
PyArray_MapIterBind(PyArrayMapIterObject *, PyArrayObject *);

NPY_NO_EXPORT PyObject*
PyArray_MapIterNew(PyObject *, int, int);

#endif
