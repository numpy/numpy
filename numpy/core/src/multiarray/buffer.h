#ifndef _NPY_PRIVATE_BUFFER_H_
#define _NPY_PRIVATE_BUFFER_H_

#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT PyBufferProcs array_as_buffer;
#else
NPY_NO_EXPORT PyBufferProcs array_as_buffer;
#endif

#endif
