#ifndef __NPY_NEW_ITERATOR__
#define __NPY_NEW_ITERATOR__

#include <Python.h>
#include <numpy/ndarraytypes.h>

/* Iterator function pointers that may be specialized */
typedef int (*NpyIter_IterNext_Fn )(void *iter);


/* Allocate a new iterator */
void* NpyIter_New(PyObject* op, npy_uint32 flags, PyArray_Descr* dtype,
                  int min_depth, int max_depth);
/* Deallocate an iterator */
int NpyIter_Deallocate(void* iter);
/* Compute a specialized iteration function for an iterator */
NpyIter_IterNext_Fn NpyIter_GetIterNext(void *iter);
/* Get the array of data pointers (1 per object being iterated) */
char **NpyIter_GetDataPtrArray(void *iter);
/* Get a pointer to the index, if it is being tracked */
npy_intp *NpyIter_GetIndexPtr(void *iter);
/* Get the array of item sizes (1 per object being iterated) */
npy_intp *NpyIter_GetItemSizeArray(void *iter);

/* Flags that may be passed to the iterator constructors */
#define NPY_ITER_CORDER_INDEX       0x0001
#define NPY_ITER_FORTRANORDER_INDEX 0x0002

#endif
