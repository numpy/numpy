/*
 * This file provides debug macros used by the other einsum files.
 *
 * Copyright (c) 2011 by Mark Wiebe (mwwiebe@gmail.com)
 * The University of British Columbia
 *
 * See LICENSE.txt for the license.
 */
#ifndef NUMPY_CORE_SRC_MULTIARRAY_EINSUM_DEBUG_H_
#define NUMPY_CORE_SRC_MULTIARRAY_EINSUM_DEBUG_H_

/********** PRINTF DEBUG TRACING **************/
#define NPY_EINSUM_DBG_TRACING 0

#if NPY_EINSUM_DBG_TRACING
#include <cstdio>
#define NPY_EINSUM_DBG_PRINT(s) printf("%s", s);
#define NPY_EINSUM_DBG_PRINT1(s, p1) printf(s, p1);
#define NPY_EINSUM_DBG_PRINT2(s, p1, p2) printf(s, p1, p2);
#define NPY_EINSUM_DBG_PRINT3(s, p1, p2, p3) printf(s);
#else
#define NPY_EINSUM_DBG_PRINT(s)
#define NPY_EINSUM_DBG_PRINT1(s, p1)
#define NPY_EINSUM_DBG_PRINT2(s, p1, p2)
#define NPY_EINSUM_DBG_PRINT3(s, p1, p2, p3)
#endif

#endif  /* NUMPY_CORE_SRC_MULTIARRAY_EINSUM_DEBUG_H_ */
