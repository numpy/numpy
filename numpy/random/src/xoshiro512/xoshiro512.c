/*  Written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

#include "xoshiro512.h"

/* This is xoshiro512** 1.0, an all-purpose, rock-solid generator. It has
   excellent (about 1ns) speed, an increased state (512 bits) that is
   large enough for any parallel application, and it passes all tests we
   are aware of.

   For generating just floating-point numbers, xoshiro512+ is even faster.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */

extern NPY_INLINE uint64_t xoshiro512_next64(xoshiro512_state *state);

extern NPY_INLINE uint32_t xoshiro512_next32(xoshiro512_state *state);


/* This is the jump function for the generator. It is equivalent
   to 2^256 calls to next(); it can be used to generate 2^256
   non-overlapping subsequences for parallel computations. */

static uint64_t s_placeholder[8];

void xoshiro512_jump(xoshiro512_state *state) {

  int i, b, w;
  static const uint64_t JUMP[] = {0x33ed89b6e7a353f9, 0x760083d7955323be,
                                  0x2837f2fbb5f22fae, 0x4b8c5674d309511c,
                                  0xb11ac47a7ba28c25, 0xf1be7667092bcc1c,
                                  0x53851efdb6df0aaf, 0x1ebbc8b23eaf25db};

  uint64_t t[sizeof s_placeholder / sizeof *s_placeholder];
  memset(t, 0, sizeof t);
  for (i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
    for (b = 0; b < 64; b++) {
      if (JUMP[i] & UINT64_C(1) << b)
        for (w = 0; w < sizeof s_placeholder / sizeof *s_placeholder; w++)
          t[w] ^= state->s[w];
      xoshiro512_next(&state->s[0]);
    }

  memcpy(state->s, t, sizeof s_placeholder);
}
