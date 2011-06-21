#ifndef _NPY_ARRAY_METHODS_H_
#define _NPY_ARRAY_METHODS_H_

#ifdef NPY_ENABLE_SEPARATE_COMPILATION
extern NPY_NO_EXPORT PyMethodDef array_methods[];
#endif

NPY_NO_EXPORT const char *
npy_casting_to_string(NPY_CASTING casting);

#endif
