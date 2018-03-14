/*  Written in 2016 by David Blackman and Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

/* This is the successor to xorshift128+. It is the fastest full-period
   generator passing BigCrush without systematic failures, but due to the
   relatively short period it is acceptable only for applications with a
   mild amount of parallelism; otherwise, use a xorshift1024* generator.

   Beside passing BigCrush, this generator passes the PractRand test suite
   up to (and included) 16TB, with the exception of binary rank tests, as
   the lowest bit of this generator is an LFSR of degree 128. The next bit
   can be described by an LFSR of degree 8256, but in the long run it will
   fail linearity tests, too. The other bits needs a much higher degree to
   be represented as LFSRs.

   We suggest to use a sign test to extract a random Boolean value, and
   right shifts to extract subsets of bits.

   Note that the generator uses a simulated rotate operation, which most C
   compilers will turn into a single instruction. In Java, you can use
   Long.rotateLeft(). In languages that do not make low-level rotation
   instructions accessible xorshift128+ could be faster.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */

#include "xoroshiro128.h"

extern INLINE uint64_t xoroshiro128_next64(xoroshiro128_state *state);

extern INLINE uint32_t xoroshiro128_next32(xoroshiro128_state *state);

void xoroshiro128_jump(xoroshiro128_state *state) {
  int i, b;
  uint64_t s0;
  uint64_t s1;
  static const uint64_t JUMP[] = {0xbeac0467eba5facb, 0xd86b048b86aa9922};

  s0 = 0;
  s1 = 0;
  for (i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
    for (b = 0; b < 64; b++) {
      if (JUMP[i] & UINT64_C(1) << b) {
        s0 ^= state->s[0];
        s1 ^= state->s[1];
      }
      xoroshiro128_next(&state->s[0]);
    }

  state->s[0] = s0;
  state->s[1] = s1;
}
