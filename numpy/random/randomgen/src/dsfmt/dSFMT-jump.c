/**
 * @file dSFMT-jump.c
 *
 * @brief do jump using jump polynomial.
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (The University of Tokyo)
 *
 * Copyright (C) 2012 Mutsuo Saito, Makoto Matsumoto,
 * Hiroshima University and The University of Tokyo.
 * All rights reserved.
 *
 * The 3-clause BSD License is applied to this software, see
 * LICENSE.txt
 */

#include <assert.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include "dSFMT-params.h"
#include "dSFMT.h"
#include "dSFMT-jump.h"
#include "dSFMT-common.h"

#if defined(__cplusplus)
extern "C" {
#endif

    struct FIX_T {
	int mexp;
	uint64_t fix[4];
    };

    struct FIX_T fix_table[] = {
	{521, {UINT64_C(0x3fff56977f035125),
	       UINT64_C(0x3ff553857b015035),
	       UINT64_C(0x4034434434434434),
	       UINT64_C(0x0140151151351371)}},
	{1279, {UINT64_C(0x3ff87befce70e89f),
		UINT64_C(0x3ff5f6afa3c60868),
		UINT64_C(0xa4ca4caccaccacdb),
		UINT64_C(0x40444444444c44c4)}},
	{4253, {UINT64_C(0x3ff85a66da51a81a),
		UINT64_C(0x3ff4f4aeab9688eb),
		UINT64_C(0x20524524534d34d3),
		UINT64_C(0xc9cc9cc9cc9ccdcf)}},
	{216091, {UINT64_C(0x3ff096d54a871071),
		  UINT64_C(0x3ffafa9bfbd5d55d),
		  UINT64_C(0x0470470470573573),
		  UINT64_C(0x0250250251259259)}},
	{0}
    };

    inline static void next_state(dsfmt_t * dsfmt);

#if defined(HAVE_SSE2)
/**
 * add internal state of src to dest as F2-vector.
 * @param dest destination state
 * @param src source state
 */
    inline static void add(dsfmt_t *dest, dsfmt_t *src) {
	int dp = dest->idx / 2;
	int sp = src->idx / 2;
	int diff = (sp - dp + DSFMT_N) % DSFMT_N;
	int p;
	int i;
	for (i = 0; i < DSFMT_N - diff; i++) {
	    p = i + diff;
	    dest->status[i].si
		= _mm_xor_si128(dest->status[i].si, src->status[p].si);
	}
	for (i = DSFMT_N - diff; i < DSFMT_N; i++) {
	    p = i + diff - DSFMT_N;
	    dest->status[i].si
		= _mm_xor_si128(dest->status[i].si, src->status[p].si);
	}
	dest->status[DSFMT_N].si
	    = _mm_xor_si128(dest->status[DSFMT_N].si,
			    src->status[DSFMT_N].si);
    }
#else
    inline static void add(dsfmt_t *dest, dsfmt_t *src) {
	int dp = dest->idx / 2;
	int sp = src->idx / 2;
	int diff = (sp - dp + DSFMT_N) % DSFMT_N;
	int p;
	int i;
	for (i = 0; i < DSFMT_N - diff; i++) {
	    p = i + diff;
	    dest->status[i].u[0] ^= src->status[p].u[0];
	    dest->status[i].u[1] ^= src->status[p].u[1];
	}
	for (; i < DSFMT_N; i++) {
	    p = i + diff - DSFMT_N;
	    dest->status[i].u[0] ^= src->status[p].u[0];
	    dest->status[i].u[1] ^= src->status[p].u[1];
	}
	dest->status[DSFMT_N].u[0] ^= src->status[DSFMT_N].u[0];
	dest->status[DSFMT_N].u[1] ^= src->status[DSFMT_N].u[1];
    }
#endif

/**
 * calculate next state
 * @param dsfmt dSFMT internal state
 */
    inline static void next_state(dsfmt_t * dsfmt) {
	int idx = (dsfmt->idx / 2) % DSFMT_N;
	w128_t * lung;
	w128_t * pstate = &dsfmt->status[0];

	lung = &pstate[DSFMT_N];
	do_recursion(&pstate[idx],
		     &pstate[idx],
		     &pstate[(idx + DSFMT_POS1) % DSFMT_N],
		     lung);
	dsfmt->idx = (dsfmt->idx + 2) % DSFMT_N64;
    }

    inline static void add_fix(dsfmt_t * dsfmt) {
	int i;
	int index = -1;
	for (i = 0; fix_table[i].mexp != 0; i++) {
	    if (fix_table[i].mexp == DSFMT_MEXP) {
		index = i;
	    }
	    if (fix_table[i].mexp > DSFMT_MEXP) {
		break;
	    }
	}
	if (index < 0) {
	    return;
	}
	for (i = 0; i < DSFMT_N; i++) {
	    dsfmt->status[i].u[0] ^= fix_table[index].fix[0];
	    dsfmt->status[i].u[1] ^= fix_table[index].fix[1];
	}
	dsfmt->status[DSFMT_N].u[0] ^= fix_table[index].fix[2];
	dsfmt->status[DSFMT_N].u[1] ^= fix_table[index].fix[3];
    }

/**
 * jump ahead using jump_string
 * @param dsfmt dSFMT internal state input and output.
 * @param jump_string string which represents jump polynomial.
 */
    void dSFMT_jump(dsfmt_t * dsfmt, const char * jump_string) {
	dsfmt_t work;
	int index = dsfmt->idx;
	int bits;
	int i;
	int j;
	memset(&work, 0, sizeof(dsfmt_t));
	add_fix(dsfmt);
	dsfmt->idx = DSFMT_N64;

	for (i = 0; jump_string[i] != '\0'; i++) {
	    bits = jump_string[i];
	    assert(isxdigit(bits));
	    bits = tolower(bits);
	    if (bits >= 'a' && bits <= 'f') {
		bits = bits - 'a' + 10;
	    } else {
		bits = bits - '0';
	    }
	    bits = bits & 0x0f;
	    for (j = 0; j < 4; j++) {
		if ((bits & 1) != 0) {
		    add(&work, dsfmt);
		}
		next_state(dsfmt);
		bits = bits >> 1;
	    }
	}
	*dsfmt = work;
	add_fix(dsfmt);
	dsfmt->idx = index;
    }

#if defined(__cplusplus)
}
#endif
