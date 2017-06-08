/* Random kit 1.3 */

/*
 * Copyright (c) 2003-2005, Jean-Sebastien Roy (js@jeannot.org)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/* @(#) $Jeannot: randomkit.h,v 1.24 2005/07/21 22:14:09 js Exp $ */


/*
 * Useful macro:
 *  RK_DEV_RANDOM: the device used for random seeding.
 *                 defaults to "/dev/urandom"
 */

#include <stddef.h>
#include "mkl_vsl.h"
#include "numpy/npy_common.h"

#ifndef _I_RANDOMKIT_
#define _I_RANDOMKIT_

typedef struct vrk_state_
{
    VSLStreamStatePtr stream;
} vrk_state;

typedef enum {
    RK_NOERR = 0, /* no error */
    RK_ENODEV = 1, /* no RK_DEV_RANDOM device */
    RK_ERR_MAX = 2
} vrk_error;

/* if changing this, also adjust brng_list[BRNG_KINDS] in randomkit.c */
#define BRNG_KINDS 9

typedef enum {
    MT19937       = 0,
    SFMT19937     = 1,
    WH            = 2,
    MT2203        = 3,
    MCG31         = 4,
    R250          = 5,
    MRG32K3A      = 6,
    MCG59         = 7,
    PHILOX4X32X10 = 8
} vrk_brng_t;


/* error strings */
extern char *vrk_strerror[RK_ERR_MAX];

/* Maximum generated random value */
#define RK_MAX 0xFFFFFFFFUL

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize the RNG state using the given seed.
 */


/*
 * Initialize the RNG state using a random seed.
 * Uses /dev/random or, when unavailable, the clock (see randomkit.c).
 * Returns RK_NOERR when no errors occurs.
 * Returns RK_ENODEV when the use of RK_DEV_RANDOM failed (for example because
 * there is no such device). In this case, the RNG was initialized using the
 * clock.
 */

/*
 * Initialize the RNG state using the given seed.
 */
extern void vrk_dealloc_stream(vrk_state *state);
extern void vrk_seed_mkl(vrk_state *state, const unsigned int seed, const vrk_brng_t brng, const unsigned int stream_id);
extern void vrk_seed_mkl_array(vrk_state *state, const unsigned int *seed_vec,
    const int seed_len, const vrk_brng_t brng, const unsigned int stream_id);
extern vrk_error vrk_randomseed_mkl(vrk_state *state, const vrk_brng_t brng, const unsigned int stream_id);
extern int vrk_get_stream_size(vrk_state *state);
extern void vrk_get_state_mkl(vrk_state *state, char * buf);
extern int vrk_set_state_mkl(vrk_state *state, char * buf);
extern int vrk_get_brng_mkl(vrk_state *state);

extern int vrk_leapfrog_stream_mkl(vrk_state *state, const int k, const int nstreams);
extern int vrk_skipahead_stream_mkl(vrk_state *state, const long long int nskip);

/*
 * fill the buffer with size random bytes
 */
extern void vrk_fill(void *buffer, size_t size, vrk_state *state);

/*
 * fill the buffer with randombytes from the random device
 * Returns RK_ENODEV if the device is unavailable, or RK_NOERR if it is
 * On Unix, if strong is defined, RK_DEV_RANDOM is used. If not, RK_DEV_URANDOM
 * is used instead. This parameter has no effect on Windows.
 * Warning: on most unixes RK_DEV_RANDOM will wait for enough entropy to answer
 * which can take a very long time on quiet systems.
 */
extern vrk_error vrk_devfill(void *buffer, size_t size, int strong);

/*
 * fill the buffer using vrk_devfill if the random device is available and using
 * vrk_fill if is is not
 * parameters have the same meaning as vrk_fill and vrk_devfill
 * Returns RK_ENODEV if the device is unavailable, or RK_NOERR if it is
 */
extern vrk_error vrk_altfill(void *buffer, size_t size, int strong,
                            vrk_state *state);

#ifdef __cplusplus
}
#endif

#endif /* _I_RANDOMKIT_ */
