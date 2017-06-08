/* Random kit 1.3 */

/*
 * Copyright (c) 2003-2005, Jean-Sebastien Roy (js@jeannot.org)
 *
 * The vrk_random and vrk_seed functions algorithms and the original design of
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
 * Original algorithm for the implementation of vrk_interval function from
 * Richard J. Wagner's implementation of the Mersenne Twister RNG, optimised by
 * Magnus Jonsson.
 *
 * Constants used in the vrk_double implementation by Isaku Wada.
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
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <assert.h>

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

#else
/* Unix */
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#endif

#include "randomkit.h"

#ifndef RK_DEV_URANDOM
#define RK_DEV_URANDOM "/dev/urandom"
#endif

#ifndef RK_DEV_RANDOM
#define RK_DEV_RANDOM "/dev/random"
#endif

char *vrk_strerror[RK_ERR_MAX] =
{
    "no error",
    "random device unvavailable"
};

/* static functions */
static unsigned long vrk_hash(unsigned long key);

void
vrk_dealloc_stream(vrk_state *state)
{
    VSLStreamStatePtr stream = state->stream;

    if(stream) {
        vslDeleteStream(&stream);
    }
}

const MKL_INT brng_list[BRNG_KINDS] = {
    VSL_BRNG_MT19937,
    VSL_BRNG_SFMT19937,
    VSL_BRNG_WH,
    VSL_BRNG_MT2203,
    VSL_BRNG_MCG31,
    VSL_BRNG_R250,
    VSL_BRNG_MRG32K3A,
    VSL_BRNG_MCG59,
    VSL_BRNG_PHILOX4X32X10
};

int vrk_get_brng_mkl(vrk_state *state)
{
    int i, mkl_brng_id = vslGetStreamStateBrng(state->stream);

    if ((VSL_BRNG_MT2203 <= mkl_brng_id) && (mkl_brng_id < VSL_BRNG_MT2203 + 6024))
        mkl_brng_id = VSL_BRNG_MT2203;
    else if ((VSL_BRNG_WH <= mkl_brng_id ) && (mkl_brng_id < VSL_BRNG_WH + 273))
        mkl_brng_id = VSL_BRNG_WH;

    for(i = 0; i < BRNG_KINDS; i++)
        if(mkl_brng_id == brng_list[i])
            return i;

    return -1;
}

void vrk_seed_mkl(vrk_state *state, const unsigned int seed, const vrk_brng_t brng, const unsigned int stream_id)
{
    VSLStreamStatePtr stream_loc;
    int err = VSL_STATUS_OK;
    const MKL_INT mkl_brng = brng_list[brng];

    if(NULL == state->stream) {
        err = vslNewStream(&(state->stream), mkl_brng + stream_id, seed);

        assert(err == VSL_STATUS_OK);
    } else {
        err = vslNewStream(&stream_loc, mkl_brng + stream_id, seed);
        assert(err == VSL_STATUS_OK);

        err = vslDeleteStream(&(state->stream));
        assert(err == VSL_STATUS_OK);

        state->stream = stream_loc;
    }

}

void
vrk_seed_mkl_array(vrk_state *state, const unsigned int seed_vec[], const int seed_len,
    const vrk_brng_t brng, const unsigned int stream_id)
{
    VSLStreamStatePtr stream_loc;
    int err = VSL_STATUS_OK;
    const MKL_INT mkl_brng = brng_list[brng];

    if(NULL == state->stream) {

        err = vslNewStreamEx(&(state->stream), mkl_brng + stream_id, (MKL_INT) seed_len, seed_vec);

        assert(err == VSL_STATUS_OK);

    } else {

        err = vslNewStreamEx(&stream_loc, mkl_brng + stream_id, (MKL_INT) seed_len, seed_vec);
        if(err == VSL_STATUS_OK) {

            err = vslDeleteStream(&(state->stream));
            assert(err == VSL_STATUS_OK);

            state->stream = stream_loc;

         }

    }
}

vrk_error
vrk_randomseed_mkl(vrk_state *state, const vrk_brng_t brng, const unsigned int stream_id)
{
#ifndef _WIN32
    struct timeval tv;
#else
    struct _timeb  tv;
#endif
    int i, no_err;
    unsigned int *seed_array;
    size_t buf_size = 624;
    size_t seed_array_len = buf_size*sizeof(unsigned int);

    seed_array =  (unsigned int *) malloc(seed_array_len);
    no_err = vrk_devfill(seed_array, seed_array_len, 0) == RK_NOERR;

    if (no_err) {
        /* ensures non-zero seed */
        seed_array[0] |= 0x80000000UL;
        vrk_seed_mkl_array(state, seed_array, buf_size, brng, stream_id);
        free(seed_array);

        return RK_NOERR;
    } else {
        free(seed_array);
    }

#ifndef _WIN32
    gettimeofday(&tv, NULL);
    vrk_seed_mkl(state, vrk_hash(getpid()) ^ vrk_hash(tv.tv_sec) ^ vrk_hash(tv.tv_usec)
            ^ vrk_hash(clock()), brng, stream_id);
#else
    _FTIME(&tv);
    vrk_seed_mkl(state, vrk_hash(tv.time) ^ vrk_hash(tv.millitm) ^ vrk_hash(clock()), brng, stream_id);
#endif

    return RK_ENODEV;
}

/*
 *  Python needs this to determine the amount memory to allocate for the buffer
 */
int vrk_get_stream_size(vrk_state *state)
{
    return vslGetStreamSize(state->stream);
}

void
vrk_get_state_mkl(vrk_state *state, char * buf)
{
    int err = vslSaveStreamM(state->stream, buf);

    assert(err == VSL_STATUS_OK);

}

int
vrk_set_state_mkl(vrk_state *state, char * buf)
{
    int err = vslLoadStreamM(&(state->stream), buf);

    return (err == VSL_STATUS_OK) ? 0 : 1;
}

int
vrk_leapfrog_stream_mkl(vrk_state *state, const MKL_INT k, const MKL_INT nstreams)
{
    int err;

    err = vslLeapfrogStream(state->stream, k, nstreams);

    switch(err) {
        case VSL_STATUS_OK:
            return 0;
        case VSL_RNG_ERROR_LEAPFROG_UNSUPPORTED:
            return 1;
        default:
            return -1;
    }
}

int
vrk_skipahead_stream_mkl(vrk_state *state, const long long int nskip)
{
    int err;

    err = vslSkipAheadStream(state->stream, nskip);

    switch(err) {
        case VSL_STATUS_OK:
            return 0;
        case VSL_RNG_ERROR_SKIPAHEAD_UNSUPPORTED:
            return 1;
        default:
            return -1;
    }
}


/* Thomas Wang 32 bits integer hash function */
static unsigned long
vrk_hash(unsigned long key)
{
    key += ~(key << 15);
    key ^=  (key >> 10);
    key +=  (key << 3);
    key ^=  (key >> 6);
    key += ~(key << 11);
    key ^=  (key >> 16);
    return key;
}


void
vrk_random_vec(vrk_state *state, const int len, unsigned int *res)
{
    viRngUniformBits(VSL_RNG_METHOD_UNIFORMBITS_STD, state->stream, len, res);
}


void
vrk_fill(void *buffer, size_t size, vrk_state *state)
{
    unsigned int r;
    unsigned char *buf = buffer;
    int err, len;

    /* len = size / 4 */
    len = (size >> 2);
    err = viRngUniformBits32(VSL_RNG_METHOD_UNIFORMBITS32_STD, state->stream, len, (unsigned int *) buf);
    assert(err == VSL_STATUS_OK);

    /* size = size % 4 */
    size &= 0x03;
    if (!size) {
        return;
    }

    buf += (len << 2);
    err = viRngUniformBits32(VSL_RNG_METHOD_UNIFORMBITS32_STD, state->stream, 1, &r);
    assert(err == VSL_STATUS_OK);

    for (; size; r >>= 8, size --) {
        *(buf++) = (unsigned char)(r & 0xFF);
    }
}

vrk_error
vrk_devfill(void *buffer, size_t size, int strong)
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

vrk_error
vrk_altfill(void *buffer, size_t size, int strong, vrk_state *state)
{
    vrk_error err;

    err = vrk_devfill(buffer, size, strong);
    if (err) {
        vrk_fill(buffer, size, state);
    }
    return err;
}
