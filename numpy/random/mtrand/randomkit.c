/* Random kit 1.3 */

/*
 * Copyright (c) 2003-2005, Jean-Sebastien Roy (js@jeannot.org)
 *
 * The rk_random and rk_seed functions algorithms and the original design of
 * the Mersenne Twister RNG:
 *
 *   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
 *   All rights reserved.
 *
 *   Redistribution and use in source and binary forms, with or without
 *   modification, are permitted provided that the following conditions
 *   are met:
 *
 *   1. Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 *   2. Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 *   3. The names of its contributors may not be used to endorse or promote
 *   products derived from this software without specific prior written
 *   permission.
 *
 *   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 *   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 *   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 *   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * Original algorithm for the implementation of rk_interval function from
 * Richard J. Wagner's implementation of the Mersenne Twister RNG, optimised by
 * Magnus Jonsson.
 *
 * Constants used in the rk_double implementation by Isaku Wada.
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

/* static char const rcsid[] =
  "@(#) $Jeannot: randomkit.c,v 1.28 2005/07/21 22:14:09 js Exp $"; */

#ifdef _WIN32
/*
 * Windows
 * XXX: we have to use this ugly defined(__GNUC__) because it is not easy to
 * detect the compiler used in distutils itself
 */
#if (defined(__GNUC__) && defined(NPY_NEEDS_MINGW_TIME_WORKAROUND))

/*
 * FIXME: ideally, we should set this to the real version of MSVCRT. We need
 * something higher than 0x601 to enable _ftime64 and co
 */
#define __MSVCRT_VERSION__ 0x0700
#include <time.h>
#include <sys/timeb.h>

/*
 * mingw msvcr lib import wrongly export _ftime, which does not exist in the
 * actual msvc runtime for version >= 8; we make it an alias to _ftime64, which
 * is available in those versions of the runtime
 */
#define _FTIME(x) _ftime64((x))
#else
#include <time.h>
#include <sys/timeb.h>
#define _FTIME(x) _ftime((x))
#endif

#ifndef RK_NO_WINCRYPT
/* Windows crypto */
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0400
#endif
#include <windows.h>
#include <wincrypt.h>
#endif

/*
 * Do not move this include. randomkit.h must be included
 * after windows timeb.h is included.
 */
#include "randomkit.h"

#else
/* Unix */
#include "randomkit.h"
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#endif

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <assert.h>

#ifndef RK_DEV_URANDOM
#define RK_DEV_URANDOM "/dev/urandom"
#endif

#ifndef RK_DEV_RANDOM
#define RK_DEV_RANDOM "/dev/random"
#endif

char *rk_strerror[RK_ERR_MAX] =
{
    "no error",
    "random device unvavailable"
};

/* static functions */
static unsigned long rk_hash(unsigned long key);

void
rk_seed(unsigned long seed, rk_state *state)
{
    int pos;
    seed &= 0xffffffffUL;

    /* Knuth's PRNG as used in the Mersenne Twister reference implementation */
    for (pos = 0; pos < RK_STATE_LEN; pos++) {
        state->key[pos] = seed;
        seed = (1812433253UL * (seed ^ (seed >> 30)) + pos + 1) & 0xffffffffUL;
    }
    state->pos = RK_STATE_LEN;
    state->gauss = 0;
    state->has_gauss = 0;
    state->has_binomial = 0;
}

/* Thomas Wang 32 bits integer hash function */
unsigned long
rk_hash(unsigned long key)
{
    key += ~(key << 15);
    key ^=  (key >> 10);
    key +=  (key << 3);
    key ^=  (key >> 6);
    key += ~(key << 11);
    key ^=  (key >> 16);
    return key;
}

rk_error
rk_randomseed(rk_state *state)
{
#ifndef _WIN32
    struct timeval tv;
#else
    struct _timeb  tv;
#endif
    int i;

    if (rk_devfill(state->key, sizeof(state->key), 0) == RK_NOERR) {
        /* ensures non-zero key */
        state->key[0] |= 0x80000000UL;
        state->pos = RK_STATE_LEN;
        state->gauss = 0;
        state->has_gauss = 0;
        state->has_binomial = 0;

        for (i = 0; i < 624; i++) {
            state->key[i] &= 0xffffffffUL;
        }
        return RK_NOERR;
    }

#ifndef _WIN32
    gettimeofday(&tv, NULL);
    rk_seed(rk_hash(getpid()) ^ rk_hash(tv.tv_sec) ^ rk_hash(tv.tv_usec)
            ^ rk_hash(clock()), state);
#else
    _FTIME(&tv);
    rk_seed(rk_hash(tv.time) ^ rk_hash(tv.millitm) ^ rk_hash(clock()), state);
#endif

    return RK_ENODEV;
}

/* Magic Mersenne Twister constants */
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL
#define UPPER_MASK 0x80000000UL
#define LOWER_MASK 0x7fffffffUL

/*
 * Slightly optimised reference implementation of the Mersenne Twister
 * Note that regardless of the precision of long, only 32 bit random
 * integers are produced
 */
unsigned long
rk_random(rk_state *state)
{
    unsigned long y;

    if (state->pos == RK_STATE_LEN) {
        int i;

        for (i = 0; i < N - M; i++) {
            y = (state->key[i] & UPPER_MASK) | (state->key[i+1] & LOWER_MASK);
            state->key[i] = state->key[i+M] ^ (y>>1) ^ (-(y & 1) & MATRIX_A);
        }
        for (; i < N - 1; i++) {
            y = (state->key[i] & UPPER_MASK) | (state->key[i+1] & LOWER_MASK);
            state->key[i] = state->key[i+(M-N)] ^ (y>>1) ^ (-(y & 1) & MATRIX_A);
        }
        y = (state->key[N - 1] & UPPER_MASK) | (state->key[0] & LOWER_MASK);
        state->key[N - 1] = state->key[M - 1] ^ (y >> 1) ^ (-(y & 1) & MATRIX_A);

        state->pos = 0;
    }
    y = state->key[state->pos++];

    /* Tempering */
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680UL;
    y ^= (y << 15) & 0xefc60000UL;
    y ^= (y >> 18);

    return y;
}


/*
 * Returns an unsigned 64 bit random integer.
 */
NPY_INLINE static npy_uint64
rk_uint64(rk_state *state)
{
    npy_uint64 upper = (npy_uint64)rk_random(state) << 32;
    npy_uint64 lower = (npy_uint64)rk_random(state);
    return upper | lower;
}


/*
 * Returns an unsigned 32 bit random integer.
 */
NPY_INLINE static npy_uint32
rk_uint32(rk_state *state)
{
    return (npy_uint32)rk_random(state);
}


/*
 * Fills an array with cnt random npy_uint64 between off and off + rng
 * inclusive. The numbers wrap if rng is sufficiently large.
 */
void
rk_random_uint64(npy_uint64 off, npy_uint64 rng, npy_intp cnt,
                 npy_uint64 *out, rk_state *state)
{
    npy_uint64 val, mask = rng;
    npy_intp i;

    if (rng == 0) {
        for (i = 0; i < cnt; i++) {
            out[i] = off;
        }
        return;
    }

    /* Smallest bit mask >= max */
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
    mask |= mask >> 32;

    for (i = 0; i < cnt; i++) {
        if (rng <= 0xffffffffUL) {
            while ((val = (rk_uint32(state) & mask)) > rng);
        }
        else {
            while ((val = (rk_uint64(state) & mask)) > rng);
        }
        out[i] =  off + val;
    }
}


/*
 * Fills an array with cnt random npy_uint32 between off and off + rng
 * inclusive. The numbers wrap if rng is sufficiently large.
 */
void
rk_random_uint32(npy_uint32 off, npy_uint32 rng, npy_intp cnt,
                 npy_uint32 *out, rk_state *state)
{
    npy_uint32 val, mask = rng;
    npy_intp i;

    if (rng == 0) {
        for (i = 0; i < cnt; i++) {
            out[i] = off;
        }
        return;
    }

    /* Smallest bit mask >= max */
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;

    for (i = 0; i < cnt; i++) {
        while ((val = (rk_uint32(state) & mask)) > rng);
        out[i] =  off + val;
    }
}


/*
 * Fills an array with cnt random npy_uint16 between off and off + rng
 * inclusive. The numbers wrap if rng is sufficiently large.
 */
void
rk_random_uint16(npy_uint16 off, npy_uint16 rng, npy_intp cnt,
                 npy_uint16 *out, rk_state *state)
{
    npy_uint16 val, mask = rng;
    npy_intp i;
    npy_uint32 buf;
    int bcnt = 0;

    if (rng == 0) {
        for (i = 0; i < cnt; i++) {
            out[i] = off;
        }
        return;
    }

    /* Smallest bit mask >= max */
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;

    for (i = 0; i < cnt; i++) {
        do {
            if (!bcnt) {
                buf = rk_uint32(state);
                bcnt = 1;
            }
            else {
                buf >>= 16;
                bcnt--;
            }
            val = (npy_uint16)buf & mask;
        } while (val > rng);
        out[i] =  off + val;
    }
}


/*
 * Fills an array with cnt random npy_uint8 between off and off + rng
 * inclusive. The numbers wrap if rng is sufficiently large.
 */
void
rk_random_uint8(npy_uint8 off, npy_uint8 rng, npy_intp cnt,
                npy_uint8 *out, rk_state *state)
{
    npy_uint8 val, mask = rng;
    npy_intp i;
    npy_uint32 buf;
    int bcnt = 0;

    if (rng == 0) {
        for (i = 0; i < cnt; i++) {
            out[i] = off;
        }
        return;
    }

    /* Smallest bit mask >= max */
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;

    for (i = 0; i < cnt; i++) {
        do {
            if (!bcnt) {
                buf = rk_uint32(state);
                bcnt = 3;
            }
            else {
                buf >>= 8;
                bcnt--;
            }
            val = (npy_uint8)buf & mask;
        } while (val > rng);
        out[i] =  off + val;
    }
}


/*
 * Fills an array with cnt random npy_bool between off and off + rng
 * inclusive.
 */
void
rk_random_bool(npy_bool off, npy_bool rng, npy_intp cnt,
                npy_bool *out, rk_state *state)
{
    npy_intp i;
    npy_uint32 buf;
    int bcnt = 0;

    if (rng == 0) {
        for (i = 0; i < cnt; i++) {
            out[i] = off;
        }
        return;
    }

    /* If we reach here rng and mask are one and off is zero */
    assert(rng == 1 && off == 0);
    for (i = 0; i < cnt; i++) {
        if (!bcnt) {
            buf = rk_uint32(state);
            bcnt = 31;
        }
        else {
            buf >>= 1;
            bcnt--;
        }
        out[i] = (buf & 0x00000001) != 0;
    }
}


long
rk_long(rk_state *state)
{
    return rk_ulong(state) >> 1;
}

unsigned long
rk_ulong(rk_state *state)
{
#if ULONG_MAX <= 0xffffffffUL
    return rk_random(state);
#else
    return (rk_random(state) << 32) | (rk_random(state));
#endif
}

unsigned long
rk_interval(unsigned long max, rk_state *state)
{
    unsigned long mask = max, value;

    if (max == 0) {
        return 0;
    }
    /* Smallest bit mask >= max */
    mask |= mask >> 1;
    mask |= mask >> 2;
    mask |= mask >> 4;
    mask |= mask >> 8;
    mask |= mask >> 16;
#if ULONG_MAX > 0xffffffffUL
    mask |= mask >> 32;
#endif

    /* Search a random value in [0..mask] <= max */
#if ULONG_MAX > 0xffffffffUL
    if (max <= 0xffffffffUL) {
        while ((value = (rk_random(state) & mask)) > max);
    }
    else {
        while ((value = (rk_ulong(state) & mask)) > max);
    }
#else
    while ((value = (rk_ulong(state) & mask)) > max);
#endif
    return value;
}

double
rk_double(rk_state *state)
{
    /* shifts : 67108864 = 0x4000000, 9007199254740992 = 0x20000000000000 */
    long a = rk_random(state) >> 5, b = rk_random(state) >> 6;
    return (a * 67108864.0 + b) / 9007199254740992.0;
}

void
rk_fill(void *buffer, size_t size, rk_state *state)
{
    unsigned long r;
    unsigned char *buf = buffer;

    for (; size >= 4; size -= 4) {
        r = rk_random(state);
        *(buf++) = r & 0xFF;
        *(buf++) = (r >> 8) & 0xFF;
        *(buf++) = (r >> 16) & 0xFF;
        *(buf++) = (r >> 24) & 0xFF;
    }

    if (!size) {
        return;
    }
    r = rk_random(state);
    for (; size; r >>= 8, size --) {
        *(buf++) = (unsigned char)(r & 0xFF);
    }
}

rk_error
rk_devfill(void *buffer, size_t size, int strong)
{
#ifndef _WIN32
    FILE *rfile;
    int done;

    if (strong) {
        rfile = fopen(RK_DEV_RANDOM, "rb");
    }
    else {
        rfile = fopen(RK_DEV_URANDOM, "rb");
    }
    if (rfile == NULL) {
        return RK_ENODEV;
    }
    done = fread(buffer, size, 1, rfile);
    fclose(rfile);
    if (done) {
        return RK_NOERR;
    }
#else

#ifndef RK_NO_WINCRYPT
    HCRYPTPROV hCryptProv;
    BOOL done;

    if (!CryptAcquireContext(&hCryptProv, NULL, NULL, PROV_RSA_FULL,
            CRYPT_VERIFYCONTEXT) || !hCryptProv) {
        return RK_ENODEV;
    }
    done = CryptGenRandom(hCryptProv, size, (unsigned char *)buffer);
    CryptReleaseContext(hCryptProv, 0);
    if (done) {
        return RK_NOERR;
    }
#endif

#endif
    return RK_ENODEV;
}

rk_error
rk_altfill(void *buffer, size_t size, int strong, rk_state *state)
{
    rk_error err;

    err = rk_devfill(buffer, size, strong);
    if (err) {
        rk_fill(buffer, size, state);
    }
    return err;
}

double
rk_gauss(rk_state *state)
{
    if (state->has_gauss) {
        const double tmp = state->gauss;
        state->gauss = 0;
        state->has_gauss = 0;
        return tmp;
    }
    else {
        double f, x1, x2, r2;

        do {
            x1 = 2.0*rk_double(state) - 1.0;
            x2 = 2.0*rk_double(state) - 1.0;
            r2 = x1*x1 + x2*x2;
        }
        while (r2 >= 1.0 || r2 == 0.0);

        /* Box-Muller transform */
        f = sqrt(-2.0*log(r2)/r2);
        /* Keep for next call */
        state->gauss = f*x1;
        state->has_gauss = 1;
        return f*x2;
    }
}
