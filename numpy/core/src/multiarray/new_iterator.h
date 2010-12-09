#ifndef __NPY_NEW_ITERATOR__
#define __NPY_NEW_ITERATOR__

#include <Python.h>
#include <numpy/ndarraytypes.h>

/* The actual structure of the iterator is an internal detail */
typedef struct {
    npy_intp internal;
} PyArray_NpyIter;

/* Iterator function pointers that may be specialized */
typedef int (*NpyIter_IterNext_Fn )(PyArray_NpyIter *iter);
typedef void (*NpyIter_GetCoords_Fn )(PyArray_NpyIter *iter,
                                      npy_intp *outcoords);


/* Allocate a new iterator over one object */
PyArray_NpyIter*
NpyIter_New(PyObject* op, npy_uint32 flags, PyArray_Descr* dtype,
                  int min_depth, int max_depth);
/* Allocate a new iterator over multiple objects */
PyArray_NpyIter*
NpyIter_MultiNew(npy_intp niter, PyObject **op_in, npy_uint32 flags,
                 npy_uint32 *op_flags, PyArray_Descr **op_request_dtypes,
                 int min_depth, int max_depth);
/* Deallocate an iterator */
int NpyIter_Deallocate(PyArray_NpyIter* iter);

/* Compute a specialized iteration function for an iterator */
NpyIter_IterNext_Fn NpyIter_GetIterNext(PyArray_NpyIter *iter);
/* Compute a specialized getcoords function for an iterator */
NpyIter_GetCoords_Fn NpyIter_GetGetCoords(PyArray_NpyIter *iter);

/* Gets the number of dimension being iterated */
npy_intp NpyIter_GetNDim(PyArray_NpyIter *iter);
/* Get the array of data pointers (1 per object being iterated) */
char **NpyIter_GetDataPtrArray(PyArray_NpyIter *iter);
/* Get a pointer to the index, if it is being tracked */
npy_intp *NpyIter_GetIndexPtr(PyArray_NpyIter *iter);
/* Get the array of item sizes (1 per object being iterated) */
npy_intp *NpyIter_GetItemSizeArray(PyArray_NpyIter *iter);

/* Get the array of strides for the inner loop */
npy_intp *NpyIter_GetInnerStrideArray(PyArray_NpyIter *iter);
/* Get a pointer to the size of the inner loop */
npy_intp* NpyIter_GetInnerLoopSizePtr(PyArray_NpyIter *iter);

/* Global flags that may be passed to the iterator constructors */
#define NPY_ITER_C_ORDER_INDEX              0x0001
#define NPY_ITER_F_ORDER_INDEX              0x0002
#define NPY_ITER_COORDS                     0x0004
#define NPY_ITER_FORCE_C_ORDER              0x0008
#define NPY_ITER_FORCE_F_ORDER              0x0010
#define NPY_ITER_FORCE_ANY_CONTIGUOUS       0x0020
#define NPY_ITER_NO_INNER_ITERATION         0x0040
/* Per-operand flags that may be passed to the iterator constructors */
#define NPY_ITER_READONLY                   0x0080
#define NPY_ITER_WRITEONLY                  0x0100
#define NPY_ITER_ALLOW_WRITEABLE_REFERENCES 0x0200

#define NPY_ITER_GLOBAL_FLAGS  (NPY_ITER_C_ORDER_INDEX | \
                                NPY_ITER_F_ORDER_INDEX | \
                                NPY_ITER_COORDS | \
                                NPY_ITER_FORCE_C_ORDER | \
                                NPY_ITER_FORCE_F_ORDER | \
                                NPY_ITER_FORCE_ANY_CONTIGUOUS | \
                                NPY_ITER_NO_INNER_ITERATION)

#define NPY_ITER_PER_OP_FLAGS  (NPY_ITER_READONLY | \
                                NPY_ITER_WRITEONLY | \
                                NPY_ITER_ALLOW_WRITEABLE_REFERENCES)
#endif
