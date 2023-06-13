#ifndef _NPY_PRIVATE__EXTOBJ_H_
#define _NPY_PRIVATE__EXTOBJ_H_

#include <numpy/ndarraytypes.h>  /* for NPY_NO_EXPORT */


NPY_NO_EXPORT int
_check_ufunc_fperr(int errmask, const char *ufunc_name);

NPY_NO_EXPORT int
_get_bufsize_errmask(const char *ufunc_name, int *buffersize, int *errormask);


#endif
