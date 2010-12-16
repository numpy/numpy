#ifndef __NPY_NEW_ITERATOR__
#define __NPY_NEW_ITERATOR__

#include <Python.h>
#include <numpy/ndarraytypes.h>

/* The actual structure of the iterator is an internal detail */
typedef struct PyArray_NpyIter_InternalOnly PyArray_NpyIter;

/* Iterator function pointers that may be specialized */
typedef int (*NpyIter_IterNext_Fn )(PyArray_NpyIter *iter);
typedef void (*NpyIter_GetCoords_Fn )(PyArray_NpyIter *iter,
                                      npy_intp *outcoords);


/* Allocate a new iterator over one object */
PyArray_NpyIter*
NpyIter_New(PyObject* op, npy_uint32 flags, PyArray_Descr* dtype,
                  int min_depth, int max_depth,
                  npy_intp a_ndim, npy_intp *axes);

/* Allocate a new iterator over multiple objects */
PyArray_NpyIter*
NpyIter_MultiNew(npy_intp niter, PyObject **op_in, npy_uint32 flags,
                 npy_uint32 *op_flags, PyArray_Descr **op_request_dtypes,
                 int min_depth, int max_depth,
                 npy_intp oa_ndim, npy_intp **op_axes);

/* Deallocate an iterator */
int NpyIter_Deallocate(PyArray_NpyIter* iter);
/* Resets the iterator back to its initial state */
void NpyIter_Reset(PyArray_NpyIter *iter);
/* Sets the iterator to point at the coordinates in 'coords' */
int NpyIter_GotoCoords(PyArray_NpyIter *iter, npy_intp *coords);
/* Sets the iterator to point at the given index */
int NpyIter_GotoIndex(PyArray_NpyIter *iter, npy_intp index);

/* Whether the iterator handles the inner loop */
int NpyIter_HasInnerLoop(PyArray_NpyIter *iter);
/* Whether the iterator is tracking coordinates */
int NpyIter_HasCoords(PyArray_NpyIter *iter);
/* Whether the iterator is tracking an index */
int NpyIter_HasIndex(PyArray_NpyIter *iter);
/* Whether the iterator gives back offsets instead of pointers */
int NpyIter_HasOffsets(PyArray_NpyIter *iter);

/* Compute a specialized iteration function for an iterator */
NpyIter_IterNext_Fn NpyIter_GetIterNext(PyArray_NpyIter *iter);
/* Compute a specialized getcoords function for an iterator */
NpyIter_GetCoords_Fn NpyIter_GetGetCoords(PyArray_NpyIter *iter);

/* Gets the number of dimensions being iterated */
npy_intp NpyIter_GetNDim(PyArray_NpyIter *iter);
/* Gets the number of objects being iterated */
npy_intp NpyIter_GetNIter(PyArray_NpyIter *iter);
/* Gets the number of times the iterator iterates */
npy_intp NpyIter_GetIterSize(PyArray_NpyIter *iter);
/* Gets the broadcast shape (if coords are enabled) */
int NpyIter_GetShape(PyArray_NpyIter *iter, npy_intp *outshape);
/* Get the array of data pointers (1 per object being iterated) */
char **NpyIter_GetDataPtrArray(PyArray_NpyIter *iter);
/* Get the array of data type pointers (1 per object being iterated) */
PyArray_Descr **NpyIter_GetDescrArray(PyArray_NpyIter *iter);
/* Get the array of objects being iterated */
PyObject **NpyIter_GetObjectArray(PyArray_NpyIter *iter);
/* Get a pointer to the index, if it is being tracked */
npy_intp *NpyIter_GetIndexPtr(PyArray_NpyIter *iter);
/* Gets an array of read flags (1 per object being iterated) */
void NpyIter_GetReadFlags(PyArray_NpyIter *iter, char *outreadflags);
/* Gets an array of write flags (1 per object being iterated) */
void NpyIter_GetWriteFlags(PyArray_NpyIter *iter, char *outwriteflags);

/* Get the array of strides for the inner loop */
npy_intp *NpyIter_GetInnerStrideArray(PyArray_NpyIter *iter);
/* Get a pointer to the size of the inner loop */
npy_intp* NpyIter_GetInnerLoopSizePtr(PyArray_NpyIter *iter);

/* For debugging */
NPY_NO_EXPORT void NpyIter_DebugPrint(PyArray_NpyIter *iter);


/* Global flags that may be passed to the iterator constructors */
#define NPY_ITER_C_ORDER_INDEX              0x00000001
#define NPY_ITER_F_ORDER_INDEX              0x00000002
#define NPY_ITER_COORDS                     0x00000004
#define NPY_ITER_FORCE_C_ORDER              0x00000008
#define NPY_ITER_FORCE_F_ORDER              0x00000010
#define NPY_ITER_FORCE_ANY_CONTIGUOUS       0x00000020
#define NPY_ITER_NO_INNER_ITERATION         0x00000040
#define NPY_ITER_OFFSETS                    0x00000080
/* Per-operand flags that may be passed to the iterator constructors */
#define NPY_ITER_READWRITE                  0x00010000
#define NPY_ITER_READONLY                   0x00020000
#define NPY_ITER_WRITEONLY                  0x00040000
#define NPY_ITER_NBO_ALIGNED                0x00080000
#define NPY_ITER_ALLOW_COPY                 0x00100000
#define NPY_ITER_ALLOW_UPDATEIFCOPY         0x00200000
#define NPY_ITER_ALLOW_SAFE_CASTS           0x00400000
#define NPY_ITER_ALLOW_SAME_KIND_CASTS      0x00800000
#define NPY_ITER_ALLOW_UNSAFE_CASTS         0x01000000
#define NPY_ITER_ALLOW_WRITEABLE_REFERENCES 0x02000000
#define NPY_ITER_ALLOCATE                   0x04000000
#define NPY_ITER_NO_SUBTYPE                 0x08000000

#define NPY_ITER_GLOBAL_FLAGS               0x0000ffff
#define NPY_ITER_PER_OP_FLAGS               0xffff0000

#endif
