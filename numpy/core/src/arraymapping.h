#ifndef _NPY_ARRAYMAPPING_H_
#define _NPY_ARRAYMAPPING_H_

extern NPY_NO_EXPORT PyMappingMethods array_as_mapping;

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
array_ass_item(PyArrayObject *self, Py_ssize_t i, PyObject *v)
{
    return array_ass_big_item(self, (intp) i, v);
}
#endif

#endif
