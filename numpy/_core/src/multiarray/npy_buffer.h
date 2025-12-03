#ifndef NUMPY_CORE_SRC_MULTIARRAY_NPY_BUFFER_H_
#define NUMPY_CORE_SRC_MULTIARRAY_NPY_BUFFER_H_

#ifdef __cplusplus
extern "C" {
#endif

extern NPY_NO_EXPORT PyBufferProcs array_as_buffer;

NPY_NO_EXPORT int
_buffer_info_free(void *buffer_info, PyObject *obj);

NPY_NO_EXPORT PyArray_Descr*
_descriptor_from_pep3118_format(char const *s);

NPY_NO_EXPORT int
void_getbuffer(PyObject *obj, Py_buffer *view, int flags);

#ifdef __cplusplus
}
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_NPY_BUFFER_H_ */
