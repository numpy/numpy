/*  Written in 2017 by Sebastiano Vigna (vigna@acm.org)

To the extent possible under law, the author has dedicated all copyright
and related and neighboring rights to this software to the public domain
worldwide. This software is distributed without any warranty.

See <http://creativecommons.org/publicdomain/zero/1.0/>. */

#include <stdint.h>
#include <string.h>

/* NOTE: as of 2017-10-08, this generator has a different multiplier (a
   fixed-point representation of the golden ratio), which eliminates
   linear dependencies from one of the lowest bits. The previous
   multiplier was 1181783497276652981 (M_8 in the paper). If you need to
   tell apart the two generators, you can refer to this generator as
   xorshift1024*φ and to the previous one as xorshift1024*M_8.

   This is a fast, high-quality generator. If 1024 bits of state are too
   much, try a xoroshiro128+ generator.

   Note that the two lowest bits of this generator are LFSRs of degree
   1024, and thus will fail binary rank tests. The other bits needs a much
   higher degree to be represented as LFSRs.

   We suggest to use a sign test to extract a random Boolean value, and
   right shifts to extract subsets of bits.

   The state must be seeded so that it is not everywhere zero. If you have
   a 64-bit seed, we suggest to seed a splitmix64 generator and use its
   output to fill s. */

uint64_t s[16];
int p;

uint64_t next(void) {
  const uint64_t s0 = s[p];
  uint64_t s1 = s[p = (p + 1) & 15];
  s1 ^= s1 << 31;                           // a
  s[p] = s1 ^ s0 ^ (s1 >> 11) ^ (s0 >> 30); // b,c
  return s[p] * 0x9e3779b97f4a7c13;
}

/* This is the jump function for the generator. It is equivalent
   to 2^512 calls to next(); it can be used to generate 2^512
   non-overlapping subsequences for parallel computations. */

void jump(void) {
  static const uint64_t JUMP[] = {
      0x84242f96eca9c41d, 0xa3c65b8776f96855, 0x5b34a39f070b5837,
      0x4489affce4f31a1e, 0x2ffeeb0a48316f40, 0xdc2d9891fe68c022,
      0x3659132bb12fea70, 0xaac17d8efa43cab8, 0xc4cb815590989b13,
      0x5ee975283d71c93b, 0x691548c86c1bd540, 0x7910c41d10a1e6a5,
      0x0b5fc64563b3e2a8, 0x047f7684e9fc949d, 0xb99181f2d8f685ca,
      0x284600e3f30e38c3};

  uint64_t t[16] = {0};
  for (int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
    for (int b = 0; b < 64; b++) {
      if (JUMP[i] & UINT64_C(1) << b)
        for (int j = 0; j < 16; j++)
          t[j] ^= s[(j + p) & 15];
      next();
    }

  for (int j = 0; j < 16; j++)
    s[(j + p) & 15] = t[j];
}
