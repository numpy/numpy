/*
 * This file is part of tela the Tensor Language.
 * Copyright (c) 1994-1995 Pekka Janhunen
 */

#ifdef __cplusplus
extern "C" {
#endif

#define DOUBLE

#ifdef DOUBLE
#define Treal double
#else
#define Treal float
#endif

extern NPY_VISIBILITY_HIDDEN void npy_cfftf(int N, Treal data[], const Treal wrk[]);
extern NPY_VISIBILITY_HIDDEN void npy_cfftb(int N, Treal data[], const Treal wrk[]);
extern NPY_VISIBILITY_HIDDEN void npy_cffti(int N, Treal wrk[]);

extern NPY_VISIBILITY_HIDDEN void npy_rfftf(int N, Treal data[], const Treal wrk[]);
extern NPY_VISIBILITY_HIDDEN void npy_rfftb(int N, Treal data[], const Treal wrk[]);
extern NPY_VISIBILITY_HIDDEN void npy_rffti(int N, Treal wrk[]);

#ifdef __cplusplus
}
#endif
