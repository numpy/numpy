#ifndef _NPY_PRIVATE_BUFFER_H_
#define _NPY_PRIVATE_BUFFER_H_

extern NPY_NO_EXPORT PyBufferProcs array_as_buffer;

NPY_NO_EXPORT int
_buffer_info_free(void *buffer_info, PyObject *obj);

NPY_NO_EXPORT PyArray_Descr*
_descriptor_from_pep3118_format(char const *s);

NPY_NO_EXPORT int
void_getbuffer(PyObject *obj, Py_buffer *view, int flags);

#endif
