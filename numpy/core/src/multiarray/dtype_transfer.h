#ifndef _NPY_DTYPE_TRANSFER_H
#define _NPY_DTYPE_TRANSFER_H

#include "lowlevel_strided_loops.h"
#include "array_method.h"


NPY_NO_EXPORT int
any_to_object_get_loop(
        PyArrayMethod_Context *context,
        int aligned, int move_references,
        npy_intp *strides,
        PyArray_StridedUnaryOp **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags);

NPY_NO_EXPORT int
object_to_any_get_loop(
        PyArrayMethod_Context *context,
        int NPY_UNUSED(aligned), int move_references,
        npy_intp *NPY_UNUSED(strides),
        PyArray_StridedUnaryOp **out_loop,
        NpyAuxData **out_transferdata,
        NPY_ARRAYMETHOD_FLAGS *flags);


#endif  /* _NPY_DTYPE_TRANSFER_H */
