#include "xorshift1024.h"

/* This is the jump function for the generator. It is equivalent
   to 2^512 calls to next(); it can be used to generate 2^512
   non-overlapping subsequences for parallel computations. */

extern INLINE uint64_t xorshift1024_next(xorshift1024_state *state);
extern INLINE uint64_t xorshift1024_next64(xorshift1024_state *state);
extern INLINE uint32_t xorshift1024_next32(xorshift1024_state *state);

void xorshift1024_jump(xorshift1024_state *state) {
  int i, j, b;
  static const uint64_t JUMP[] = {
      0x84242f96eca9c41d, 0xa3c65b8776f96855, 0x5b34a39f070b5837,
      0x4489affce4f31a1e, 0x2ffeeb0a48316f40, 0xdc2d9891fe68c022,
      0x3659132bb12fea70, 0xaac17d8efa43cab8, 0xc4cb815590989b13,
      0x5ee975283d71c93b, 0x691548c86c1bd540, 0x7910c41d10a1e6a5,
      0x0b5fc64563b3e2a8, 0x047f7684e9fc949d, 0xb99181f2d8f685ca,
      0x284600e3f30e38c3};

  uint64_t t[16] = {0};
  for (i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
    for (b = 0; b < 64; b++) {
      if (JUMP[i] & UINT64_C(1) << b)
        for (j = 0; j < 16; j++)
          t[j] ^= state->s[(j + state->p) & 15];
      xorshift1024_next(state);
    }

  for (j = 0; j < 16; j++)
    state->s[(j + state->p) & 15] = t[j];
}
