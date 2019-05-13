/*  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

#include "xoshiro256.h"

/* This is xoshiro256** 1.0, our all-purpose, rock-solid generator. It has
   excellent (sub-ns) speed, a state (256 bits) that is large enough for
   any parallel application, and it passes all tests we are aware of.

   For generating just floating-point numbers, xoshiro256+ is even faster.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */

extern NPY_INLINE uint64_t xoshiro256_next64(xoshiro256_state *state);

extern NPY_INLINE uint32_t xoshiro256_next32(xoshiro256_state *state);


/* This is the jump function for the generator. It is equivalent
   to 2^128 calls to next(); it can be used to generate 2^128
   non-overlapping subsequences for parallel computations. */

void xoshiro256_jump(xoshiro256_state *state)
{
	int i, b;
	static const uint64_t JUMP[] = {0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c};

	uint64_t s0 = 0;
	uint64_t s1 = 0;
	uint64_t s2 = 0;
	uint64_t s3 = 0;
	for (i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
		for (b = 0; b < 64; b++)
		{
			if (JUMP[i] & UINT64_C(1) << b)
			{
				s0 ^= state->s[0];
				s1 ^= state->s[1];
				s2 ^= state->s[2];
				s3 ^= state->s[3];
			}
			xoshiro256_next(&state->s[0]);
		}

	state->s[0] = s0;
	state->s[1] = s1;
	state->s[2] = s2;
	state->s[3] = s3;
}
