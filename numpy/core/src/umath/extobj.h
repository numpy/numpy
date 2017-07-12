#ifndef _NPY_PRIVATE__EXTOBJ_H_
#define _NPY_PRIVATE__EXTOBJ_H_

#include <numpy/ndarraytypes.h>  /* for NPY_NO_EXPORT */

NPY_NO_EXPORT int
_error_handler(int method, PyObject *errobj, char *errtype, int retstatus, int *first);

NPY_NO_EXPORT PyObject *
get_global_ext_obj(void);

NPY_NO_EXPORT int
_extract_pyvals(PyObject *ref, const char *name, int *bufsize,
                int *errmask, PyObject **errobj);

NPY_NO_EXPORT int
_check_ufunc_fperr(int errmask, PyObject *extobj, const char *ufunc_name);

NPY_NO_EXPORT int
_get_bufsize_errmask(PyObject * extobj, const char *ufunc_name,
                     int *buffersize, int *errormask);

/********************/
#define USE_USE_DEFAULTS 1
/********************/

#if USE_USE_DEFAULTS==1
NPY_NO_EXPORT int
ufunc_update_use_defaults(void);
#endif

#endif
