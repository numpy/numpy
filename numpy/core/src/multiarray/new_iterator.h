#ifndef __NPY_NEW_ITERATOR__
#define __NPY_NEW_ITERATOR__

#include <Python.h>
#include <numpy/ndarraytypes.h>

/* The actual structure of the iterator is an internal detail */
typedef struct {
    npy_intp internal_only;
} PyArray_NpyIter;

/* Iterator function pointers that may be specialized */
typedef int (*NpyIter_IterNext_Fn )(PyArray_NpyIter *iter);
typedef void (*NpyIter_GetCoords_Fn )(PyArray_NpyIter *iter, npy_intp *outcoords);


/* Allocate a new iterator */
PyArray_NpyIter*
NpyIter_New(PyObject* op, npy_uint32 flags, PyArray_Descr* dtype,
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

/* Flags that may be passed to the iterator constructors */
#define NPY_ITER_C_ORDER_INDEX        0x0001
#define NPY_ITER_F_ORDER_INDEX        0x0002
#define NPY_ITER_COORDS               0x0004
#define NPY_ITER_FORCE_C_ORDER        0x0008
#define NPY_ITER_FORCE_F_ORDER        0x0010
#define NPY_ITER_FORCE_ANY_CONTIGUOUS 0x0020

#endif
