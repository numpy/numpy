#ifndef _RANDOMDGEN__XOSHIRO512STARSTAR_H_
#define _RANDOMDGEN__XOSHIRO512STARSTAR_H_

#ifdef _WIN32
#if _MSC_VER == 1500
#include "../common/inttypes.h"
#define INLINE __forceinline
#else
#include <inttypes.h>
#define INLINE __inline __forceinline
#endif
#else
#include <inttypes.h>
#define INLINE inline
#endif
#include <string.h>

typedef struct s_xoshiro512starstar_state
{
	uint64_t s[8];
	int has_uint32;
	uint32_t uinteger;
} xoshiro512starstar_state;

static INLINE uint64_t rotl(const uint64_t x, int k)
{
	return (x << k) | (x >> (64 - k));
}

static INLINE uint64_t xoshiro512starstar_next(uint64_t *s)
{
	const uint64_t result_starstar = rotl(s[1] * 5, 7) * 9;

	const uint64_t t = s[1] << 11;

	s[2] ^= s[0];
	s[5] ^= s[1];
	s[1] ^= s[2];
	s[7] ^= s[3];
	s[3] ^= s[4];
	s[4] ^= s[5];
	s[0] ^= s[6];
	s[6] ^= s[7];

	s[6] ^= t;

	s[7] = rotl(s[7], 21);

	return result_starstar;
}

static INLINE uint64_t
xoshiro512starstar_next64(xoshiro512starstar_state *state)
{
	return xoshiro512starstar_next(&state->s[0]);
}

static INLINE uint32_t
xoshiro512starstar_next32(xoshiro512starstar_state *state)
{
	uint64_t next;
	if (state->has_uint32)
	{
		state->has_uint32 = 0;
		return state->uinteger;
	}
	next = xoshiro512starstar_next(&state->s[0]);
	state->has_uint32 = 1;
	state->uinteger = (uint32_t)(next >> 32);
	return (uint32_t)(next & 0xffffffff);
}

void xoshiro512starstar_jump(xoshiro512starstar_state *state);

#endif
