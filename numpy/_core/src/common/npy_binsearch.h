#ifndef __NPY_BINSEARCH_H__
#define __NPY_BINSEARCH_H__

#include "npy_sort.h"
#include <numpy/npy_common.h>
#include <numpy/ndarraytypes.h>


#ifdef __cplusplus
extern "C" {
#endif

typedef void (PyArray_BinSearchFunc)(const char*, const char*, char*,
                                     npy_intp, npy_intp,
                                     npy_intp, npy_intp, npy_intp,
                                     PyArrayObject*);

typedef int (PyArray_ArgBinSearchFunc)(const char*, const char*,
                                       const char*, char*,
                                       npy_intp, npy_intp, npy_intp,
                                       npy_intp, npy_intp, npy_intp,
                                       PyArrayObject*);

NPY_NO_EXPORT PyArray_BinSearchFunc* get_binsearch_func(PyArray_Descr *dtype, NPY_SEARCHSIDE side);
NPY_NO_EXPORT PyArray_ArgBinSearchFunc* get_argbinsearch_func(PyArray_Descr *dtype, NPY_SEARCHSIDE side);

#ifdef __cplusplus
}
#endif

#endif
