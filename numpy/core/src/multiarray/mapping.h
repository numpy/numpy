#ifndef _NPY_ARRAYMAPPING_H_
#define _NPY_ARRAYMAPPING_H_

#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT PyMappingMethods array_as_mapping;
#else
NPY_NO_EXPORT PyMappingMethods array_as_mapping;
#endif

NPY_NO_EXPORT PyObject *
array_big_item(PyArrayObject *self, intp i);

NPY_NO_EXPORT Py_ssize_t
array_length(PyArrayObject *self);

NPY_NO_EXPORT PyObject *
array_item_nice(PyArrayObject *self, Py_ssize_t i);

NPY_NO_EXPORT PyObject *
array_subscript(PyArrayObject *self, PyObject *op);

NPY_NO_EXPORT int
array_ass_big_item(PyArrayObject *self, intp i, PyObject *v);

#if PY_VERSION_HEX < 0x02050000
        #if SIZEOF_INT == SIZEOF_INTP
                #define array_ass_item array_ass_big_item
        #endif
#else
        #if SIZEOF_SIZE_T == SIZEOF_INTP
                #define array_ass_item array_ass_big_item
        #endif
#endif
#ifndef array_ass_item
NPY_NO_EXPORT int
_array_ass_item(PyArrayObject *self, Py_ssize_t i, PyObject *v);
#define array_ass_item _array_ass_item
#endif

NPY_NO_EXPORT PyObject *
add_new_axes_0d(PyArrayObject *,  int);

NPY_NO_EXPORT int
count_new_axes_0d(PyObject *tuple);

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
