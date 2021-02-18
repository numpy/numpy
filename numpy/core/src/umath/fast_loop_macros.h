/**
 * Macros to help build fast ufunc inner loops.
 *
 * These expect to have access to the arguments of a typical ufunc loop,
 *
 *     char **args
 *     npy_intp const *dimensions
 *     npy_intp const *steps
 */
#ifndef _NPY_UMATH_FAST_LOOP_MACROS_H_
#define _NPY_UMATH_FAST_LOOP_MACROS_H_

/*
 * MAX_STEP_SIZE is used to determine if we need to use SIMD version of the ufunc.
 * Very large step size can be as slow as processing it using scalar. The
 * value of 2097152 ( = 2MB) was chosen using 2 considerations:
 * 1) Typical linux kernel page size is 4Kb, but sometimes it could also be 2MB
 *    which is == 2097152 Bytes. For a step size as large as this, surely all
 *    the loads/stores of gather/scatter instructions falls on 16 different pages
 *    which one would think would slow down gather/scatter instructions.
 * 2) It additionally satisfies MAX_STEP_SIZE*16/esize < NPY_MAX_INT32 which
 *    allows us to use i32 version of gather/scatter (as opposed to the i64 version)
 *    without problems (step larger than NPY_MAX_INT32*esize/16 would require use of
 *    i64gather/scatter). esize = element size = 4/8 bytes for float/double.
 */
#define MAX_STEP_SIZE 2097152

static NPY_INLINE npy_uintp
abs_ptrdiff(char *a, char *b)
{
    return (a > b) ? (a - b) : (b - a);
}

/**
 * Simple unoptimized loop macros that iterate over the ufunc arguments in
 * parallel.
 * @{
 */

/** (<ignored>) -> (op1) */
#define OUTPUT_LOOP\
    char *op1 = args[1];\
    npy_intp os1 = steps[1];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, op1 += os1)

/** (ip1) -> (op1) */
#define UNARY_LOOP\
    char *ip1 = args[0], *op1 = args[1];\
    npy_intp is1 = steps[0], os1 = steps[1];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, op1 += os1)

/** (ip1) -> (op1, op2) */
#define UNARY_LOOP_TWO_OUT\
    char *ip1 = args[0], *op1 = args[1], *op2 = args[2];\
    npy_intp is1 = steps[0], os1 = steps[1], os2 = steps[2];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, op1 += os1, op2 += os2)

#define BINARY_DEFS\
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2];\
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2];\
    npy_intp n = dimensions[0];\
    npy_intp i;\

#define BINARY_LOOP_SLIDING\
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1)

/** (ip1, ip2) -> (op1) */
#define BINARY_LOOP\
    BINARY_DEFS\
    BINARY_LOOP_SLIDING

/** (ip1, ip2) -> (op1, op2) */
#define BINARY_LOOP_TWO_OUT\
    char *ip1 = args[0], *ip2 = args[1], *op1 = args[2], *op2 = args[3];\
    npy_intp is1 = steps[0], is2 = steps[1], os1 = steps[2], os2 = steps[3];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1, op2 += os2)

/** (ip1, ip2, ip3) -> (op1) */
#define TERNARY_LOOP\
    char *ip1 = args[0], *ip2 = args[1], *ip3 = args[2], *op1 = args[3];\
    npy_intp is1 = steps[0], is2 = steps[1], is3 = steps[2], os1 = steps[3];\
    npy_intp n = dimensions[0];\
    npy_intp i;\
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, ip3 += is3, op1 += os1)

/** @} */

/* unary loop input and output contiguous */
#define IS_UNARY_CONT(tin, tout) (steps[0] == sizeof(tin) && \
                                  steps[1] == sizeof(tout))

#define IS_OUTPUT_CONT(tout) (steps[1] == sizeof(tout))

#define IS_BINARY_REDUCE ((args[0] == args[2])\
        && (steps[0] == steps[2])\
        && (steps[0] == 0))

/* binary loop input and output contiguous */
#define IS_BINARY_CONT(tin, tout) (steps[0] == sizeof(tin) && \
                                   steps[1] == sizeof(tin) && \
                                   steps[2] == sizeof(tout))

/* binary loop input and output contiguous with first scalar */
#define IS_BINARY_CONT_S1(tin, tout) (steps[0] == 0 && \
                                   steps[1] == sizeof(tin) && \
                                   steps[2] == sizeof(tout))

/* binary loop input and output contiguous with second scalar */
#define IS_BINARY_CONT_S2(tin, tout) (steps[0] == sizeof(tin) && \
                                   steps[1] == 0 && \
                                   steps[2] == sizeof(tout))

/*
 * loop with contiguous specialization
 * op should be the code working on `tin in` and
 * storing the result in `tout *out`
 * combine with NPY_GCC_OPT_3 to allow autovectorization
 * should only be used where its worthwhile to avoid code bloat
 */
#define BASE_UNARY_LOOP(tin, tout, op) \
    UNARY_LOOP { \
        const tin in = *(tin *)ip1; \
        tout *out = (tout *)op1; \
        op; \
    }

#define UNARY_LOOP_FAST(tin, tout, op)          \
    do { \
        /* condition allows compiler to optimize the generic macro */ \
        if (IS_UNARY_CONT(tin, tout)) { \
            if (args[0] == args[1]) { \
                BASE_UNARY_LOOP(tin, tout, op) \
            } \
            else { \
                BASE_UNARY_LOOP(tin, tout, op) \
            } \
        } \
        else { \
            BASE_UNARY_LOOP(tin, tout, op) \
        } \
    } \
    while (0)

/*
 * loop with contiguous specialization
 * op should be the code working on `tin in1`, `tin in2` and
 * storing the result in `tout *out`
 * combine with NPY_GCC_OPT_3 to allow autovectorization
 * should only be used where its worthwhile to avoid code bloat
 */
#define BASE_BINARY_LOOP(tin, tout, op) \
    BINARY_LOOP { \
        const tin in1 = *(tin *)ip1; \
        const tin in2 = *(tin *)ip2; \
        tout *out = (tout *)op1; \
        op; \
    }

/*
 * unfortunately gcc 6/7 regressed and we need to give it additional hints to
 * vectorize inplace operations (PR80198)
 * must only be used after op1 == ip1 or ip2 has been checked
 * TODO: using ivdep might allow other compilers to vectorize too
 */
#if __GNUC__ >= 6
#define IVDEP_LOOP _Pragma("GCC ivdep")
#else
#define IVDEP_LOOP
#endif
#define BASE_BINARY_LOOP_INP(tin, tout, op) \
    BINARY_DEFS\
    IVDEP_LOOP \
    for(i = 0; i < n; i++, ip1 += is1, ip2 += is2, op1 += os1) { \
        const tin in1 = *(tin *)ip1; \
        const tin in2 = *(tin *)ip2; \
        tout *out = (tout *)op1; \
        op; \
    }

#define BASE_BINARY_LOOP_S(tin, tout, cin, cinp, vin, vinp, op) \
    const tin cin = *(tin *)cinp; \
    BINARY_LOOP { \
        const tin vin = *(tin *)vinp; \
        tout *out = (tout *)op1; \
        op; \
    }

/* PR80198 again, scalar works without the pragma */
#define BASE_BINARY_LOOP_S_INP(tin, tout, cin, cinp, vin, vinp, op) \
    const tin cin = *(tin *)cinp; \
    BINARY_LOOP { \
        const tin vin = *(tin *)vinp; \
        tout *out = (tout *)vinp; \
        op; \
    }

#define BINARY_LOOP_FAST(tin, tout, op)         \
    do { \
        /* condition allows compiler to optimize the generic macro */ \
        if (IS_BINARY_CONT(tin, tout)) { \
            if (abs_ptrdiff(args[2], args[0]) == 0 && \
                    abs_ptrdiff(args[2], args[1]) >= NPY_MAX_SIMD_SIZE) { \
                BASE_BINARY_LOOP_INP(tin, tout, op) \
            } \
            else if (abs_ptrdiff(args[2], args[1]) == 0 && \
                         abs_ptrdiff(args[2], args[0]) >= NPY_MAX_SIMD_SIZE) { \
                BASE_BINARY_LOOP_INP(tin, tout, op) \
            } \
            else { \
                BASE_BINARY_LOOP(tin, tout, op) \
            } \
        } \
        else if (IS_BINARY_CONT_S1(tin, tout)) { \
            if (abs_ptrdiff(args[2], args[1]) == 0) { \
                BASE_BINARY_LOOP_S_INP(tin, tout, in1, args[0], in2, ip2, op) \
            } \
            else { \
                BASE_BINARY_LOOP_S(tin, tout, in1, args[0], in2, ip2, op) \
            } \
        } \
        else if (IS_BINARY_CONT_S2(tin, tout)) { \
            if (abs_ptrdiff(args[2], args[0]) == 0) { \
                BASE_BINARY_LOOP_S_INP(tin, tout, in2, args[1], in1, ip1, op) \
            } \
            else { \
                BASE_BINARY_LOOP_S(tin, tout, in2, args[1], in1, ip1, op) \
            }\
        } \
        else { \
            BASE_BINARY_LOOP(tin, tout, op) \
        } \
    } \
    while (0)

#define BINARY_REDUCE_LOOP_INNER\
    char *ip2 = args[1]; \
    npy_intp is2 = steps[1]; \
    npy_intp n = dimensions[0]; \
    npy_intp i; \
    for(i = 0; i < n; i++, ip2 += is2)

#define BINARY_REDUCE_LOOP(TYPE)\
    char *iop1 = args[0]; \
    TYPE io1 = *(TYPE *)iop1; \
    BINARY_REDUCE_LOOP_INNER

#define IS_BINARY_STRIDE_ONE(esize, vsize) \
    ((steps[0] == esize) && \
     (steps[1] == esize) && \
     (steps[2] == esize) && \
     (abs_ptrdiff(args[2], args[0]) >= vsize) && \
     (abs_ptrdiff(args[2], args[1]) >= vsize))

/*
 * stride is equal to element size and input and destination are equal or
 * don't overlap within one register. The check of the steps against
 * esize also quarantees that steps are >= 0.
 */
#define IS_BLOCKABLE_UNARY(esize, vsize) \
    (steps[0] == (esize) && steps[0] == steps[1] && \
     (npy_is_aligned(args[0], esize) && npy_is_aligned(args[1], esize)) && \
     ((abs_ptrdiff(args[1], args[0]) >= (vsize)) || \
      ((abs_ptrdiff(args[1], args[0]) == 0))))

/*
 * Avoid using SIMD for very large step sizes for several reasons:
 * 1) Supporting large step sizes requires use of i64gather/scatter_ps instructions,
 *    in which case we need two i64gather instructions and an additional vinsertf32x8
 *    instruction to load a single zmm register (since one i64gather instruction
 *    loads into a ymm register). This is not ideal for performance.
 * 2) Gather and scatter instructions can be slow when the loads/stores
 *    cross page boundaries.
 *
 * We instead rely on i32gather/scatter_ps instructions which use a 32-bit index
 * element. The index needs to be < INT_MAX to avoid overflow. MAX_STEP_SIZE
 * ensures this. The condition also requires that the input and output arrays
 * should have no overlap in memory.
 */
#define IS_BINARY_SMALL_STEPS_AND_NOMEMOVERLAP \
    ((labs(steps[0]) < MAX_STEP_SIZE)  && \
     (labs(steps[1]) < MAX_STEP_SIZE)  && \
     (labs(steps[2]) < MAX_STEP_SIZE)  && \
     (nomemoverlap(args[0], steps[0] * dimensions[0], args[2], steps[2] * dimensions[0])) && \
     (nomemoverlap(args[1], steps[1] * dimensions[0], args[2], steps[2] * dimensions[0])))

#define IS_UNARY_TWO_OUT_SMALL_STEPS_AND_NOMEMOVERLAP \
    ((labs(steps[0]) < MAX_STEP_SIZE)  && \
     (labs(steps[1]) < MAX_STEP_SIZE)  && \
     (labs(steps[2]) < MAX_STEP_SIZE)  && \
     (nomemoverlap(args[0], steps[0] * dimensions[0], args[2], steps[2] * dimensions[0])) && \
     (nomemoverlap(args[0], steps[0] * dimensions[0], args[1], steps[1] * dimensions[0])))

/*
 * 1) Output should be contiguous, can handle strided input data
 * 2) Input step should be smaller than MAX_STEP_SIZE for performance
 * 3) Input and output arrays should have no overlap in memory
 */
#define IS_OUTPUT_BLOCKABLE_UNARY(esizein, esizeout, vsize) \
    ((steps[0] & (esizein-1)) == 0 && \
     steps[1] == (esizeout) && labs(steps[0]) < MAX_STEP_SIZE && \
     (nomemoverlap(args[1], steps[1] * dimensions[0], args[0], steps[0] * dimensions[0])))

#define IS_BLOCKABLE_REDUCE(esize, vsize) \
    (steps[1] == (esize) && abs_ptrdiff(args[1], args[0]) >= (vsize) && \
     npy_is_aligned(args[1], (esize)) && \
     npy_is_aligned(args[0], (esize)))

#define IS_BLOCKABLE_BINARY(esize, vsize) \
    (steps[0] == steps[1] && steps[1] == steps[2] && steps[2] == (esize) && \
     npy_is_aligned(args[2], (esize)) && npy_is_aligned(args[1], (esize)) && \
     npy_is_aligned(args[0], (esize)) && \
     (abs_ptrdiff(args[2], args[0]) >= (vsize) || \
      abs_ptrdiff(args[2], args[0]) == 0) && \
     (abs_ptrdiff(args[2], args[1]) >= (vsize) || \
      abs_ptrdiff(args[2], args[1]) >= 0))

#define IS_BLOCKABLE_BINARY_SCALAR1(esize, vsize) \
    (steps[0] == 0 && steps[1] == steps[2] && steps[2] == (esize) && \
     npy_is_aligned(args[2], (esize)) && npy_is_aligned(args[1], (esize)) && \
     ((abs_ptrdiff(args[2], args[1]) >= (vsize)) || \
      (abs_ptrdiff(args[2], args[1]) == 0)) && \
     abs_ptrdiff(args[2], args[0]) >= (esize))

#define IS_BLOCKABLE_BINARY_SCALAR2(esize, vsize) \
    (steps[1] == 0 && steps[0] == steps[2] && steps[2] == (esize) && \
     npy_is_aligned(args[2], (esize)) && npy_is_aligned(args[0], (esize)) && \
     ((abs_ptrdiff(args[2], args[0]) >= (vsize)) || \
      (abs_ptrdiff(args[2], args[0]) == 0)) && \
     abs_ptrdiff(args[2], args[1]) >= (esize))

#undef abs_ptrdiff

#define IS_BLOCKABLE_BINARY_BOOL(esize, vsize) \
    (steps[0] == (esize) && steps[0] == steps[1] && steps[2] == (1) && \
     npy_is_aligned(args[1], (esize)) && \
     npy_is_aligned(args[0], (esize)))

#define IS_BLOCKABLE_BINARY_SCALAR1_BOOL(esize, vsize) \
    (steps[0] == 0 && steps[1] == (esize) && steps[2] == (1) && \
     npy_is_aligned(args[1], (esize)))

#define IS_BLOCKABLE_BINARY_SCALAR2_BOOL(esize, vsize) \
    (steps[0] == (esize) && steps[1] == 0 && steps[2] == (1) && \
     npy_is_aligned(args[0], (esize)))

/* align var to alignment */
#define LOOP_BLOCK_ALIGN_VAR(var, type, alignment)\
    npy_intp i, peel = npy_aligned_block_offset(var, sizeof(type),\
                                                alignment, n);\
    for(i = 0; i < peel; i++)

#define LOOP_BLOCKED(type, vsize)\
    for(; i < npy_blocked_end(peel, sizeof(type), vsize, n);\
            i += (vsize / sizeof(type)))

#define LOOP_BLOCKED_END\
    for (; i < n; i++)


#endif /* _NPY_UMATH_FAST_LOOP_MACROS_H_ */
