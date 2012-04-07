/*
 * These function have been adapted from Python 2.4.1's _randommodule.c
 *
 * The following changes have been made to it in 2005 by Robert Kern:
 *
 *   * init_by_array has been declared extern, has a void return, and uses the
 *     rk_state structure to hold its data.
 *
 *  The original file has the following verbatim comments:
 *
 *  ------------------------------------------------------------------
 *  The code in this module was based on a download from:
 *     http://www.math.keio.ac.jp/~matumoto/MT2002/emt19937ar.html
 *
 *  It was modified in 2002 by Raymond Hettinger as follows:
 *
 *   * the principal computational lines untouched except for tabbing.
 *
 *   * renamed genrand_res53() to random_random() and wrapped
 *     in python calling/return code.
 *
 *   * genrand_int32() and the helper functions, init_genrand()
 *     and init_by_array(), were declared static, wrapped in
 *     Python calling/return code.  also, their global data
 *     references were replaced with structure references.
 *
 *   * unused functions from the original were deleted.
 *     new, original C python code was added to implement the
 *     Random() interface.
 *
 *  The following are the verbatim comments from the original code:
 *
 *  A C-program for MT19937, with initialization improved 2002/1/26.
 *  Coded by Takuji Nishimura and Makoto Matsumoto.
 *
 *  Before using, initialize the state by using init_genrand(seed)
 *  or init_by_array(init_key, key_length).
 *
 *  Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    1. Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 *    2. Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 *    3. The names of its contributors may not be used to endorse or promote
 *   products derived from this software without specific prior written
 *   permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 *  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 *  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 *  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 *  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 *  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *
 *  Any feedback is very welcome.
 *  http://www.math.keio.ac.jp/matumoto/emt.html
 *  email: matumoto@math.keio.ac.jp
 */
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "initarray.h"

static void
init_genrand(rk_state *self, unsigned long s);

/* initializes mt[RK_STATE_LEN] with a seed */
static void
init_genrand(rk_state *self, unsigned long s)
{
    int mti;
    unsigned long *mt = self->key;

    mt[0] = s & 0xffffffffUL;
    for (mti = 1; mti < RK_STATE_LEN; mti++) {
        /*
         * See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier.
         * In the previous versions, MSBs of the seed affect
         * only MSBs of the array mt[].
         * 2002/01/09 modified by Makoto Matsumoto
         */
        mt[mti] = (1812433253UL * (mt[mti-1] ^ (mt[mti-1] >> 30)) + mti);
        /* for > 32 bit machines */
        mt[mti] &= 0xffffffffUL;
    }
    self->pos = mti;
    return;
}


/*
 * initialize by an array with array-length
 * init_key is the array for initializing keys
 * key_length is its length
 */
extern void
init_by_array(rk_state *self, unsigned long init_key[], npy_intp key_length)
{
    /* was signed in the original code. RDH 12/16/2002 */
    npy_intp i = 1;
    npy_intp j = 0;
    unsigned long *mt = self->key;
    npy_intp k;

    init_genrand(self, 19650218UL);
    k = (RK_STATE_LEN > key_length ? RK_STATE_LEN : key_length);
    for (; k; k--) {
        /* non linear */
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525UL))
            + init_key[j] + j;
        /* for > 32 bit machines */
        mt[i] &= 0xffffffffUL;
        i++;
        j++;
        if (i >= RK_STATE_LEN) {
            mt[0] = mt[RK_STATE_LEN - 1];
            i = 1;
        }
        if (j >= key_length) {
            j = 0;
        }
    }
    for (k = RK_STATE_LEN - 1; k; k--) {
        mt[i] = (mt[i] ^ ((mt[i-1] ^ (mt[i-1] >> 30)) * 1566083941UL))
             - i; /* non linear */
        mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
        i++;
        if (i >= RK_STATE_LEN) {
            mt[0] = mt[RK_STATE_LEN - 1];
            i = 1;
        }
    }

    mt[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */
    self->gauss = 0;
    self->has_gauss = 0;
    self->has_binomial = 0;
}
