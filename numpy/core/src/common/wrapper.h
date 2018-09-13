#ifndef _COMMON_WRAPPER_H
#define _COMMON_WRAPPER_H


/* Convert a string to an int in an arbitrary base */
NPY_NO_EXPORT npy_longlong
npy_strtoll(const char *str, char **endptr, int base);

/* Convert a string to an int in an arbitrary base */
NPY_NO_EXPORT npy_ulonglong
npy_strtoull(const char *str, char **endptr, int base);

#endif
