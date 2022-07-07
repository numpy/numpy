#ifndef NUMPY_CORE_SRC_COMMON_UMATHMODULE_H_
#define NUMPY_CORE_SRC_COMMON_UMATHMODULE_H_

#include "ufunc_object.h"
#include "ufunc_type_resolution.h"

NPY_NO_EXPORT PyObject *
get_sfloat_dtype(PyObject *NPY_UNUSED(mod), PyObject *NPY_UNUSED(args));

/* Defined in umath/extobj.c */
NPY_NO_EXPORT int
PyUFunc_GiveFloatingpointErrors(const char *name, int fpe_errors);

PyObject * add_newdoc_ufunc(PyObject *NPY_UNUSED(dummy), PyObject *args);
PyObject * ufunc_frompyfunc(PyObject *NPY_UNUSED(dummy), PyObject *args, PyObject *NPY_UNUSED(kwds));


int initumath(PyObject *m);

#endif  /* NUMPY_CORE_SRC_COMMON_UMATHMODULE_H_ */
