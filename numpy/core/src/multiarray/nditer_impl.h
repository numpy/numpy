/*
 * This is a PRIVATE INTERNAL NumPy header, intended to be used *ONLY*
 * by the iterator implementation code. All other internal NumPy code
 * should use the exposed iterator API.
 */
#ifndef NPY_ITERATOR_IMPLEMENTATION_CODE
#error This header is intended for use ONLY by iterator implementation code.
#endif

#ifndef NUMPY_CORE_SRC_MULTIARRAY_NDITER_IMPL_H_
#define NUMPY_CORE_SRC_MULTIARRAY_NDITER_IMPL_H_

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define _MULTIARRAYMODULE

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#include "numpy/arrayobject.h"
#include "npy_pycompat.h"
#include "convert_datatype.h"

#include "lowlevel_strided_loops.h"
#include "dtype_transfer.h"
#include "dtype_traversal.h"


/********** ITERATOR CONSTRUCTION TIMING **************/
#define NPY_IT_CONSTRUCTION_TIMING 0

#if NPY_IT_CONSTRUCTION_TIMING
#define NPY_IT_TIME_POINT(var) { \
            unsigned int hi, lo; \
            __asm__ __volatile__ ( \
                "rdtsc" \
                : "=d" (hi), "=a" (lo)); \
            var = (((unsigned long long)hi) << 32) | lo; \
        }
#define NPY_IT_PRINT_TIME_START(var) { \
            printf("%30s: start\n", #var); \
            c_temp = var; \
        }
#define NPY_IT_PRINT_TIME_VAR(var) { \
            printf("%30s: %6.0f clocks\n", #var, \
                    ((double)(var-c_temp))); \
            c_temp = var; \
        }
#else
#define NPY_IT_TIME_POINT(var)
#endif

/******************************************************/

/********** PRINTF DEBUG TRACING **************/
#define NPY_IT_DBG_TRACING 0

#if NPY_IT_DBG_TRACING
#define NPY_IT_DBG_PRINT(s) printf("%s", s)
#define NPY_IT_DBG_PRINT1(s, p1) printf(s, p1)
#define NPY_IT_DBG_PRINT2(s, p1, p2) printf(s, p1, p2)
#define NPY_IT_DBG_PRINT3(s, p1, p2, p3) printf(s, p1, p2, p3)
#else
#define NPY_IT_DBG_PRINT(s)
#define NPY_IT_DBG_PRINT1(s, p1)
#define NPY_IT_DBG_PRINT2(s, p1, p2)
#define NPY_IT_DBG_PRINT3(s, p1, p2, p3)
#endif
/**********************************************/

/* Rounds up a number of bytes to be divisible by sizeof intp */
#if NPY_SIZEOF_INTP == 4
#define NPY_INTP_ALIGNED(size) ((size + 0x3)&(-0x4))
#else
#define NPY_INTP_ALIGNED(size) ((size + 0x7)&(-0x8))
#endif

/* Internal iterator flags */

/* The perm is the identity */
#define NPY_ITFLAG_IDENTPERM    (1 << 0)
/* The perm has negative entries (indicating flipped axes) */
#define NPY_ITFLAG_NEGPERM      (1 << 1)
/* The iterator is tracking an index */
#define NPY_ITFLAG_HASINDEX     (1 << 2)
/* The iterator is tracking a multi-index */
#define NPY_ITFLAG_HASMULTIINDEX    (1 << 3)
/* The iteration order was forced on construction */
#define NPY_ITFLAG_FORCEDORDER  (1 << 4)
/* The inner loop is handled outside the iterator */
#define NPY_ITFLAG_EXLOOP      (1 << 5)
/* The iterator is ranged */
#define NPY_ITFLAG_RANGE        (1 << 6)
/* The iterator is buffered */
#define NPY_ITFLAG_BUFFER       (1 << 7)
/* The iterator should grow the buffered inner loop when possible */
#define NPY_ITFLAG_GROWINNER    (1 << 8)
/* There is just one iteration, can specialize iternext for that */
#define NPY_ITFLAG_ONEITERATION (1 << 9)
/* Delay buffer allocation until first Reset* call */
#define NPY_ITFLAG_DELAYBUF     (1 << 10)
/* Iteration needs API access during iternext */
#define NPY_ITFLAG_NEEDSAPI     (1 << 11)
/* Iteration includes one or more operands being reduced */
#define NPY_ITFLAG_REDUCE       (1 << 12)
/* Reduce iteration doesn't need to recalculate reduce loops next time */
#define NPY_ITFLAG_REUSE_REDUCE_LOOPS (1 << 13)
/*
 * Offset of (combined) ArrayMethod flags for all transfer functions.
 * For now, we use the top 8 bits.
 */
#define NPY_ITFLAG_TRANSFERFLAGS_SHIFT 24

/* Internal iterator per-operand iterator flags */

/* The operand will be written to */
#define NPY_OP_ITFLAG_WRITE        0x0001
/* The operand will be read from */
#define NPY_OP_ITFLAG_READ         0x0002
/* The operand needs type conversion/byte swapping/alignment */
#define NPY_OP_ITFLAG_CAST         0x0004
/* The operand never needs buffering */
#define NPY_OP_ITFLAG_BUFNEVER     0x0008
/* The operand is aligned */
#define NPY_OP_ITFLAG_ALIGNED      0x0010
/* The operand is being reduced */
#define NPY_OP_ITFLAG_REDUCE       0x0020
/* The operand is for temporary use, does not have a backing array */
#define NPY_OP_ITFLAG_VIRTUAL      0x0040
/* The operand requires masking when copying buffer -> array */
#define NPY_OP_ITFLAG_WRITEMASKED  0x0080
/* The operand's data pointer is pointing into its buffer */
#define NPY_OP_ITFLAG_USINGBUFFER  0x0100
/* The operand must be copied (with UPDATEIFCOPY if also ITFLAG_WRITE) */
#define NPY_OP_ITFLAG_FORCECOPY    0x0200
/* The operand has temporary data, write it back at dealloc */
#define NPY_OP_ITFLAG_HAS_WRITEBACK 0x0400

/*
 * The data layout of the iterator is fully specified by
 * a triple (itflags, ndim, nop).  These three variables
 * are expected to exist in all functions calling these macros,
 * either as true variables initialized to the correct values
 * from the iterator, or as constants in the case of specialized
 * functions such as the various iternext functions.
 */

struct NpyIter_InternalOnly {
    /* Initial fixed position data */
    npy_uint32 itflags;
    npy_uint8 ndim, nop;
    npy_int8 maskop;
    npy_intp itersize, iterstart, iterend;
    /* iterindex is only used if RANGED or BUFFERED is set */
    npy_intp iterindex;
    /* The rest is variable */
    char iter_flexdata[];
};

typedef struct NpyIter_AxisData_tag NpyIter_AxisData;
typedef struct NpyIter_TransferInfo_tag NpyIter_TransferInfo;
typedef struct NpyIter_BufferData_tag NpyIter_BufferData;

typedef npy_int16 npyiter_opitflags;

/* Byte sizes of the iterator members */
#define NIT_PERM_SIZEOF(itflags, ndim, nop) \
        NPY_INTP_ALIGNED(NPY_MAXDIMS)
#define NIT_DTYPES_SIZEOF(itflags, ndim, nop) \
        ((NPY_SIZEOF_INTP)*(nop))
#define NIT_RESETDATAPTR_SIZEOF(itflags, ndim, nop) \
        ((NPY_SIZEOF_INTP)*(nop+1))
#define NIT_BASEOFFSETS_SIZEOF(itflags, ndim, nop) \
        ((NPY_SIZEOF_INTP)*(nop+1))
#define NIT_OPERANDS_SIZEOF(itflags, ndim, nop) \
        ((NPY_SIZEOF_INTP)*(nop))
#define NIT_OPITFLAGS_SIZEOF(itflags, ndim, nop) \
        (NPY_INTP_ALIGNED(sizeof(npyiter_opitflags) * nop))
#define NIT_BUFFERDATA_SIZEOF(itflags, ndim, nop) \
        ((itflags&NPY_ITFLAG_BUFFER) ? ( \
            (NPY_SIZEOF_INTP)*(6 + 5*nop) + sizeof(NpyIter_TransferInfo) * nop) : 0)

/* Byte offsets of the iterator members starting from iter->iter_flexdata */
#define NIT_PERM_OFFSET() \
        (0)
#define NIT_DTYPES_OFFSET(itflags, ndim, nop) \
        (NIT_PERM_OFFSET() + \
         NIT_PERM_SIZEOF(itflags, ndim, nop))
#define NIT_RESETDATAPTR_OFFSET(itflags, ndim, nop) \
        (NIT_DTYPES_OFFSET(itflags, ndim, nop) + \
         NIT_DTYPES_SIZEOF(itflags, ndim, nop))
#define NIT_BASEOFFSETS_OFFSET(itflags, ndim, nop) \
        (NIT_RESETDATAPTR_OFFSET(itflags, ndim, nop) + \
         NIT_RESETDATAPTR_SIZEOF(itflags, ndim, nop))
#define NIT_OPERANDS_OFFSET(itflags, ndim, nop) \
        (NIT_BASEOFFSETS_OFFSET(itflags, ndim, nop) + \
         NIT_BASEOFFSETS_SIZEOF(itflags, ndim, nop))
#define NIT_OPITFLAGS_OFFSET(itflags, ndim, nop) \
        (NIT_OPERANDS_OFFSET(itflags, ndim, nop) + \
         NIT_OPERANDS_SIZEOF(itflags, ndim, nop))
#define NIT_BUFFERDATA_OFFSET(itflags, ndim, nop) \
        (NIT_OPITFLAGS_OFFSET(itflags, ndim, nop) + \
         NIT_OPITFLAGS_SIZEOF(itflags, ndim, nop))
#define NIT_AXISDATA_OFFSET(itflags, ndim, nop) \
        (NIT_BUFFERDATA_OFFSET(itflags, ndim, nop) + \
         NIT_BUFFERDATA_SIZEOF(itflags, ndim, nop))

/* Internal-only ITERATOR DATA MEMBER ACCESS */
#define NIT_ITFLAGS(iter) \
        ((iter)->itflags)
#define NIT_NDIM(iter) \
        ((iter)->ndim)
#define NIT_NOP(iter) \
        ((iter)->nop)
#define NIT_MASKOP(iter) \
        ((iter)->maskop)
#define NIT_ITERSIZE(iter) \
        (iter->itersize)
#define NIT_ITERSTART(iter) \
        (iter->iterstart)
#define NIT_ITEREND(iter) \
        (iter->iterend)
#define NIT_ITERINDEX(iter) \
        (iter->iterindex)
#define NIT_PERM(iter)  ((npy_int8 *)( \
        iter->iter_flexdata + NIT_PERM_OFFSET()))
#define NIT_DTYPES(iter) ((PyArray_Descr **)( \
        iter->iter_flexdata + NIT_DTYPES_OFFSET(itflags, ndim, nop)))
#define NIT_RESETDATAPTR(iter) ((char **)( \
        iter->iter_flexdata + NIT_RESETDATAPTR_OFFSET(itflags, ndim, nop)))
#define NIT_BASEOFFSETS(iter) ((npy_intp *)( \
        iter->iter_flexdata + NIT_BASEOFFSETS_OFFSET(itflags, ndim, nop)))
#define NIT_OPERANDS(iter) ((PyArrayObject **)( \
        iter->iter_flexdata + NIT_OPERANDS_OFFSET(itflags, ndim, nop)))
#define NIT_OPITFLAGS(iter) ((npyiter_opitflags *)( \
        iter->iter_flexdata + NIT_OPITFLAGS_OFFSET(itflags, ndim, nop)))
#define NIT_BUFFERDATA(iter) ((NpyIter_BufferData *)( \
        iter->iter_flexdata + NIT_BUFFERDATA_OFFSET(itflags, ndim, nop)))
#define NIT_AXISDATA(iter) ((NpyIter_AxisData *)( \
        iter->iter_flexdata + NIT_AXISDATA_OFFSET(itflags, ndim, nop)))

/* Internal-only BUFFERDATA MEMBER ACCESS */

struct NpyIter_TransferInfo_tag {
    NPY_cast_info read;
    NPY_cast_info write;
    NPY_traverse_info clear;
    /* Probably unnecessary, but make sure what follows is intp aligned: */
    npy_intp _unused_ensure_alignment[];
};

struct NpyIter_BufferData_tag {
    npy_intp buffersize, size, bufiterend,
             reduce_pos, reduce_outersize, reduce_outerdim;
    npy_intp bd_flexdata;
};

#define NBF_BUFFERSIZE(bufferdata) ((bufferdata)->buffersize)
#define NBF_SIZE(bufferdata) ((bufferdata)->size)
#define NBF_BUFITEREND(bufferdata) ((bufferdata)->bufiterend)
#define NBF_REDUCE_POS(bufferdata) ((bufferdata)->reduce_pos)
#define NBF_REDUCE_OUTERSIZE(bufferdata) ((bufferdata)->reduce_outersize)
#define NBF_REDUCE_OUTERDIM(bufferdata) ((bufferdata)->reduce_outerdim)
#define NBF_STRIDES(bufferdata) ( \
        &(bufferdata)->bd_flexdata + 0)
#define NBF_PTRS(bufferdata) ((char **) \
        (&(bufferdata)->bd_flexdata + 1*(nop)))
#define NBF_REDUCE_OUTERSTRIDES(bufferdata) ( \
        (&(bufferdata)->bd_flexdata + 2*(nop)))
#define NBF_REDUCE_OUTERPTRS(bufferdata) ((char **) \
        (&(bufferdata)->bd_flexdata + 3*(nop)))
#define NBF_BUFFERS(bufferdata) ((char **) \
        (&(bufferdata)->bd_flexdata + 4*(nop)))
#define NBF_TRANSFERINFO(bufferdata) ((NpyIter_TransferInfo *) \
        (&(bufferdata)->bd_flexdata + 5*(nop)))

/* Internal-only AXISDATA MEMBER ACCESS. */
struct NpyIter_AxisData_tag {
    npy_intp shape, index;
    npy_intp ad_flexdata;
};
#define NAD_SHAPE(axisdata) ((axisdata)->shape)
#define NAD_INDEX(axisdata) ((axisdata)->index)
#define NAD_STRIDES(axisdata) ( \
        &(axisdata)->ad_flexdata + 0)
#define NAD_PTRS(axisdata) ((char **) \
        (&(axisdata)->ad_flexdata + 1*(nop+1)))

#define NAD_NSTRIDES() \
        ((nop) + ((itflags&NPY_ITFLAG_HASINDEX) ? 1 : 0))

/* Size of one AXISDATA struct within the iterator */
#define NIT_AXISDATA_SIZEOF(itflags, ndim, nop) (( \
        /* intp shape */ \
        1 + \
        /* intp index */ \
        1 + \
        /* intp stride[nop+1] AND char* ptr[nop+1] */ \
        2*((nop)+1) \
        )*(size_t)NPY_SIZEOF_INTP)

/*
 * Macro to advance an AXISDATA pointer by a specified count.
 * Requires that sizeof_axisdata be previously initialized
 * to NIT_AXISDATA_SIZEOF(itflags, ndim, nop).
 */
#define NIT_INDEX_AXISDATA(axisdata, index) ((NpyIter_AxisData *) \
        (((char *)(axisdata)) + (index)*sizeof_axisdata))
#define NIT_ADVANCE_AXISDATA(axisdata, count) \
        axisdata = NIT_INDEX_AXISDATA(axisdata, count)

/* Size of the whole iterator */
#define NIT_SIZEOF_ITERATOR(itflags, ndim, nop) ( \
        sizeof(struct NpyIter_InternalOnly) + \
        NIT_AXISDATA_OFFSET(itflags, ndim, nop) + \
        NIT_AXISDATA_SIZEOF(itflags, ndim, nop)*(ndim ? ndim : 1))

/* Internal helper functions shared between implementation files */

/**
 * Undo the axis permutation of the iterator. When the operand has fewer
 * dimensions then the iterator, this can return negative values for
 * inserted (broadcast) dimensions.
 *
 * @param axis Axis for which to undo the iterator axis permutation.
 * @param ndim If `op_axes` is being used, this is the iterator dimension,
 *             otherwise this is the operand dimension.
 * @param perm The iterator axis permutation NIT_PERM(iter)
 * @param axis_flipped Will be set to true if this is a flipped axis
 *        (i.e. is iterated in reversed order) and otherwise false.
 *        Can be NULL if the information is not needed.
 * @return The unpermuted axis. Without `op_axes` this is correct, with
 *         `op_axes` this indexes into `op_axes` (unpermuted iterator axis)
 */
static inline int
npyiter_undo_iter_axis_perm(
        int axis, int ndim, const npy_int8 *perm, npy_bool *axis_flipped)
{
    npy_int8 p = perm[axis];
    /* The iterator treats axis reversed, thus adjust by ndim */
    npy_bool flipped = p < 0;
    if (axis_flipped != NULL) {
        *axis_flipped = flipped;
    }
    if (flipped) {
        axis = ndim + p;
    }
    else {
        axis = ndim - p - 1;
    }
    return axis;
}

NPY_NO_EXPORT void
npyiter_coalesce_axes(NpyIter *iter);
NPY_NO_EXPORT int
npyiter_allocate_buffers(NpyIter *iter, char **errmsg);
NPY_NO_EXPORT void
npyiter_goto_iterindex(NpyIter *iter, npy_intp iterindex);
NPY_NO_EXPORT int
npyiter_copy_from_buffers(NpyIter *iter);
NPY_NO_EXPORT int
npyiter_copy_to_buffers(NpyIter *iter, char **prev_dataptrs);
NPY_NO_EXPORT void
npyiter_clear_buffers(NpyIter *iter);

/*
 * Function to get the ArrayMethod flags of the transfer functions.
 * TODO: This function should be public and removed from `nditer_impl.h`, but
 *       this requires making the ArrayMethod flags public API first.
 */
NPY_NO_EXPORT int
NpyIter_GetTransferFlags(NpyIter *iter);

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_NDITER_IMPL_H_ */
