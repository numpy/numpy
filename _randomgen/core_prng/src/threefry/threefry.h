/*
Adapted from https://github.com/pdebuyl/threefry

Copyright (c) 2017, Pierre de Buyl

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
contributors may be used to endorse or promote products derived from
this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include <stdint.h>

#define N_WORDS 2
#define KEY_LENGTH 3
#define C240 0x1BD11BDAA9FC1A22
#define N_ROUNDS 20
#define MASK 0xffffffffffffffff
#define DOUBLE_MULT 5.421010862427522e-20

typedef struct {
  uint64_t c0, c1;
} threefry_t;

typedef struct s_threefry_state {
  threefry_t *c;
  threefry_t *k;
  int has_uint32;
  uint32_t uinteger;
} threefry_state;

static const int ROTATION[] = {16, 42, 12, 31, 16, 32, 24, 21};

static inline uint64_t rotl_64(uint64_t x, int d) {
  return ((x << d) | (x >> (64 - d)));
}

static inline threefry_t mix(threefry_t x, int R) {
  x.c0 += x.c1;
  x.c1 = rotl_64(x.c1, R) ^ x.c0;
  return x;
}

static inline threefry_t threefry(threefry_t p, threefry_t k) {
  const uint64_t K[] = {k.c0, k.c1, C240 ^ k.c0 ^ k.c1};
  int rmod4, rdiv4;
  threefry_t x;
  x = p;
  for (int r = 0; r < N_ROUNDS; r++) {
    rmod4 = r % 4;
    if (rmod4 == 0) {
      rdiv4 = r / 4;
      x.c0 += K[rdiv4 % KEY_LENGTH];
      x.c1 += K[(rdiv4 + 1) % KEY_LENGTH] + rdiv4;
    }
    x = mix(x, ROTATION[r % 8]);
  }
  x.c0 += K[(N_ROUNDS / 4) % KEY_LENGTH];
  x.c1 += K[(N_ROUNDS / 4 + 1) % KEY_LENGTH] + N_ROUNDS / 4;
  return x;
}

static inline uint64_t threefry_next(threefry_t *c, threefry_t *k) {
  threefry_t x;
  x = threefry(*c, *k);
  c->c0++;
  return x.c0;
}

static inline uint64_t threefry_next64(threefry_state *state) {
  return threefry_next(state->c, state->k);
}

static inline uint64_t threefry_next32(threefry_state *state) {
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  uint64_t next = threefry_next(state->c, state->k);
  ;
  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next & 0xffffffff);
  return (uint32_t)(next >> 32);
}
