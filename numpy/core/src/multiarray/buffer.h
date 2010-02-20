#ifndef _NPY_PRIVATE_BUFFER_H_
#define _NPY_PRIVATE_BUFFER_H_

#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT PyBufferProcs array_as_buffer;
#else
NPY_NO_EXPORT PyBufferProcs array_as_buffer;
#endif

NPY_NO_EXPORT void
_array_dealloc_buffer_info(PyArrayObject *self);

NPY_NO_EXPORT PyArray_Descr*
_descriptor_from_pep3118_format(char *s);

#endif
