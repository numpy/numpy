#ifndef _NPY_ARRAY_METHODS_H_
#define _NPY_ARRAY_METHODS_H_

extern NPY_NO_EXPORT PyMethodDef array_methods[];

NPY_NO_EXPORT const char *
npy_casting_to_string(NPY_CASTING casting);

#endif
