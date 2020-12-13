#ifndef _NPY_UMATH_LOOPS_UTILS_H_
#define _NPY_UMATH_LOOPS_UTILS_H_

#include "numpy/npy_common.h" // NPY_FINLINE
/*
 * nomemoverlap - returns false if two strided arrays have an overlapping
 * region in memory. ip_size/op_size = size of the arrays which can be negative
 * indicating negative steps.
 */
NPY_FINLINE npy_bool
nomemoverlap(char *ip, npy_intp ip_size, char *op, npy_intp op_size)
{
    char *ip_start, *ip_end, *op_start, *op_end;
    if (ip_size < 0) {
        ip_start = ip + ip_size;
        ip_end = ip;
    }
    else {
        ip_start = ip;
        ip_end = ip + ip_size;
    }
    if (op_size < 0) {
        op_start = op + op_size;
        op_end = op;
    }
    else {
        op_start = op;
        op_end = op + op_size;
    }
    return (ip_start == op_start && op_end == ip_end) ||
           (ip_start > op_end) || (op_start > ip_end);
}

// returns true if two strided arrays have an overlapping region in memory
// same as `nomemoverlap()` but requires array length and step sizes
NPY_FINLINE npy_bool
is_mem_overlap(const void *src, npy_intp src_step, const void *dst, npy_intp dst_step, npy_intp len)
{
    return !(nomemoverlap((char*)src, src_step*len, (char*)dst, dst_step*len));
}

#endif // _NPY_UMATH_LOOPS_UTILS_H_
