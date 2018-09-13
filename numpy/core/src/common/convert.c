#include <numpy/ndarraytypes.h>
#include "convert.h"

NPY_NO_EXPORT npy_longlong
npy_strtoll(const char *str, char **endptr, int base)
{
#if defined HAVE_STRTOLL
    return strtoll(str, endptr, base);
#elif defined _MSC_VER
    return _strtoi64(str, endptr, base);
#else
    /* ok on 64 bit posix */
    return PyOS_strtol(str, endptr, base);
#endif
}

NPY_NO_EXPORT npy_ulonglong
npy_strtoull(const char *str, char **endptr, int base)
{
#if defined HAVE_STRTOULL
    return strtoull(str, endptr, base);
#elif defined _MSC_VER
    return _strtoui64(str, endptr, base);
#else
    /* ok on 64 bit posix */
    return PyOS_strtoul(str, endptr, base);
#endif
}


