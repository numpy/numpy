#ifndef NUMPY_CORE_SRC_COMMON_PARTITION_H_
#define NUMPY_CORE_SRC_COMMON_PARTITION_H_

#include "npy_sort.h"

/* Python include is for future object sorts */
#include <Python.h>

#include <numpy/ndarraytypes.h>
#include <numpy/npy_common.h>

#define NPY_MAX_PIVOT_STACK 50

typedef int (PyArray_PartitionFunc)(void *, npy_intp, npy_intp,
                                    npy_intp *, npy_intp *, npy_intp,
                                    void *);
typedef int (PyArray_ArgPartitionFunc)(void *, npy_intp *, npy_intp, npy_intp,
                                       npy_intp *, npy_intp *, npy_intp,
                                       void *);
#ifdef __cplusplus
extern "C" {
#endif

NPY_NO_EXPORT PyArray_PartitionFunc *
get_partition_func(int type, NPY_SELECTKIND which);

NPY_NO_EXPORT PyArray_ArgPartitionFunc *
get_argpartition_func(int type, NPY_SELECTKIND which);

#ifdef __cplusplus
}
#endif

#endif
