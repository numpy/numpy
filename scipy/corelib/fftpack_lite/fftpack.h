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

extern void cfftf(int N, Treal data[], const Treal wrk[]);
extern void cfftb(int N, Treal data[], const Treal wrk[]);
extern void cffti(int N, Treal wrk[]);

extern void rfftf(int N, Treal data[], const Treal wrk[]);
extern void rfftb(int N, Treal data[], const Treal wrk[]);
extern void rffti(int N, Treal wrk[]);

#ifdef __cplusplus
}
#endif
