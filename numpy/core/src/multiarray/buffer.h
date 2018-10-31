#ifndef _NPY_PRIVATE_BUFFER_H_
#define _NPY_PRIVATE_BUFFER_H_

extern NPY_NO_EXPORT PyBufferProcs array_as_buffer;

NPY_NO_EXPORT void
_dealloc_cached_buffer_info(PyObject *self);

NPY_NO_EXPORT PyArray_Descr*
_descriptor_from_pep3118_format(char *s);

NPY_NO_EXPORT int
gentype_getbuffer(PyObject *obj, Py_buffer *view, int flags);

#endif
