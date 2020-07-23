#ifndef _NPY_PRIVATE_BUFFER_H_
#define _NPY_PRIVATE_BUFFER_H_

extern NPY_NO_EXPORT PyBufferProcs array_as_buffer;

NPY_NO_EXPORT int
_buffer_info_free(void *buffer_info, PyObject *obj);

/*
 * Tag the buffer info pointer. This was appended to the array struct
 * in NumPy 1.20, tagging the pointer gives us a chance to raise/print
 * a useful error message when a user modifies fields that should belong to
 * us.
 */
static NPY_INLINE void *
buffer_info_tag(void *buffer_info)
{
    return (void *)((uintptr_t)buffer_info + 2);
}

NPY_NO_EXPORT PyArray_Descr*
_descriptor_from_pep3118_format(char const *s);

NPY_NO_EXPORT int
void_getbuffer(PyObject *obj, Py_buffer *view, int flags);

#endif
