/*
 * PCG64 Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 * Copyright 2015 Robert Kern <robert.kern@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *     https://www.pcg-random.org
 *
 * Relicensed MIT in May 2019
 *
 * The MIT License
 *
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "pcg64.h"

void pcg64_fill_uint32(pcg64_state *state, size_t count, uint32_t *out) {
  size_t i;
  uint64_t next = 0;
  pcg64_random_t rng;

  if (count == 0) {
    return;
  }

  if (state->has_uint32) {
    out[0] = state->uinteger;
    out++;
    state->has_uint32 = 0;
    count--;
    if (count == 0) {
      return;
    }
  }

  rng = *state->pcg_state;
  for (i = 0; i < count / 2; i++) {
    next = pcg64_random_r(&rng);
    out[2 * i] = (uint32_t)(next & 0xffffffff);
    out[2 * i + 1] = (uint32_t)(next >> 32);
  }

  if (count & 1) {
    next = pcg64_random_r(&rng);
    out[count - 1] = (uint32_t)(next & 0xffffffff);
    state->has_uint32 = 1;
  }

  state->uinteger = (uint32_t)(next >> 32);
  state->pcg_state->state = rng.state;
}

void pcg64_fill_uint64(pcg64_state *state, size_t count, uint64_t *out) {
  size_t i;
  pcg64_random_t rng;

  if (count == 0) {
    return;
  }
  rng = *state->pcg_state;
  for (i = 0; i < count; i++) {
    out[i] = pcg64_random_r(&rng);
  }
  state->pcg_state->state = rng.state;
}

void pcg64_fill_double(pcg64_state *state, size_t count, double *out) {
  size_t i;
  pcg64_random_t rng;

  if (count == 0) {
    return;
  }
  rng = *state->pcg_state;
  for (i = 0; i < count; i++) {
    out[i] = (pcg64_random_r(&rng) >> 11) *
             (1.0 / 9007199254740992.0);
  }
  state->pcg_state->state = rng.state;
}

extern inline void pcg_setseq_128_step_r(pcg_state_setseq_128 *rng);
extern inline uint64_t pcg_output_xsl_rr_128_64(pcg128_t state);
extern inline void pcg_setseq_128_srandom_r(pcg_state_setseq_128 *rng,
                                            pcg128_t initstate,
                                            pcg128_t initseq);
extern inline uint64_t
pcg_setseq_128_xsl_rr_64_random_r(pcg_state_setseq_128 *rng);
extern inline uint64_t
pcg_setseq_128_xsl_rr_64_boundedrand_r(pcg_state_setseq_128 *rng,
                                       uint64_t bound);
extern inline void pcg_setseq_128_advance_r(pcg_state_setseq_128 *rng,
                                            pcg128_t delta);
extern inline uint64_t pcg_cm_random_r(pcg_state_setseq_128 *rng);
extern inline void pcg_cm_step_r(pcg_state_setseq_128 *rng);
extern inline uint64_t pcg_output_cm_128_64(pcg128_t state);
extern inline void pcg_cm_srandom_r(pcg_state_setseq_128 *rng, pcg128_t initstate, pcg128_t initseq);

/* Multi-step advance functions (jump-ahead, jump-back)
 *
 * The method used here is based on Brown, "Random Number Generation
 * with Arbitrary Stride,", Transactions of the American Nuclear
 * Society (Nov. 1994).  The algorithm is very similar to fast
 * exponentiation.
 *
 * Even though delta is an unsigned integer, we can pass a
 * signed integer to go backwards, it just goes "the long way round".
 */

#ifndef PCG_EMULATED_128BIT_MATH

pcg128_t pcg_advance_lcg_128(pcg128_t state, pcg128_t delta, pcg128_t cur_mult,
                             pcg128_t cur_plus) {
  pcg128_t acc_mult = 1u;
  pcg128_t acc_plus = 0u;
  while (delta > 0) {
    if (delta & 1) {
      acc_mult *= cur_mult;
      acc_plus = acc_plus * cur_mult + cur_plus;
    }
    cur_plus = (cur_mult + 1) * cur_plus;
    cur_mult *= cur_mult;
    delta /= 2;
  }
  return acc_mult * state + acc_plus;
}

#else

pcg128_t pcg_advance_lcg_128(pcg128_t state, pcg128_t delta, pcg128_t cur_mult,
                             pcg128_t cur_plus) {
  pcg128_t acc_mult = PCG_128BIT_CONSTANT(0u, 1u);
  pcg128_t acc_plus = PCG_128BIT_CONSTANT(0u, 0u);
  while ((delta.high > 0) || (delta.low > 0)) {
    if (delta.low & 1) {
      acc_mult = pcg128_mult(acc_mult, cur_mult);
      acc_plus = pcg128_add(pcg128_mult(acc_plus, cur_mult), cur_plus);
    }
    cur_plus = pcg128_mult(pcg128_add(cur_mult, PCG_128BIT_CONSTANT(0u, 1u)),
                            cur_plus);
    cur_mult = pcg128_mult(cur_mult, cur_mult);
    delta.low = (delta.low >> 1) | (delta.high << 63);
    delta.high >>= 1;
  }
  return pcg128_add(pcg128_mult(acc_mult, state), acc_plus);
}

#endif

extern inline uint64_t pcg64_next64(pcg64_state *state);
extern inline uint32_t pcg64_next32(pcg64_state *state);

extern inline uint64_t pcg64_cm_next64(pcg64_state *state);
extern inline uint32_t pcg64_cm_next32(pcg64_state *state);

extern void pcg64_advance(pcg64_state *state, uint64_t *step) {
  pcg128_t delta;
#ifndef PCG_EMULATED_128BIT_MATH
  delta = (((pcg128_t)step[0]) << 64) | step[1];
#else
  delta.high = step[0];
  delta.low = step[1];
#endif
  pcg64_advance_r(state->pcg_state, delta);
}

extern void pcg64_cm_advance(pcg64_state *state, uint64_t *step) {
  pcg128_t delta;
#ifndef PCG_EMULATED_128BIT_MATH
  delta = (((pcg128_t)step[0]) << 64) | step[1];
#else
  delta.high = step[0];
  delta.low = step[1];
#endif
  pcg_cm_advance_r(state->pcg_state, delta);
}

extern void pcg64_set_seed(pcg64_state *state, uint64_t *seed, uint64_t *inc) {
  pcg128_t s, i;
#ifndef PCG_EMULATED_128BIT_MATH
  s = (((pcg128_t)seed[0]) << 64) | seed[1];
  i = (((pcg128_t)inc[0]) << 64) | inc[1];
#else
  s.high = seed[0];
  s.low = seed[1];
  i.high = inc[0];
  i.low = inc[1];
#endif
  pcg64_srandom_r(state->pcg_state, s, i);
}

extern void pcg64_get_state(pcg64_state *state, uint64_t *state_arr,
                            int *has_uint32, uint32_t *uinteger) {
  /*
   * state_arr contains state.high, state.low, inc.high, inc.low
   *    which are interpreted as the upper 64 bits (high) or lower
   *    64 bits of a uint128_t variable
   *
   */
#ifndef PCG_EMULATED_128BIT_MATH
  state_arr[0] = (uint64_t)(state->pcg_state->state >> 64);
  state_arr[1] = (uint64_t)(state->pcg_state->state & 0xFFFFFFFFFFFFFFFFULL);
  state_arr[2] = (uint64_t)(state->pcg_state->inc >> 64);
  state_arr[3] = (uint64_t)(state->pcg_state->inc & 0xFFFFFFFFFFFFFFFFULL);
#else
  state_arr[0] = (uint64_t)state->pcg_state->state.high;
  state_arr[1] = (uint64_t)state->pcg_state->state.low;
  state_arr[2] = (uint64_t)state->pcg_state->inc.high;
  state_arr[3] = (uint64_t)state->pcg_state->inc.low;
#endif
  has_uint32[0] = state->has_uint32;
  uinteger[0] = state->uinteger;
}

extern void pcg64_set_state(pcg64_state *state, uint64_t *state_arr,
                            int has_uint32, uint32_t uinteger) {
  /*
   * state_arr contains state.high, state.low, inc.high, inc.low
   *    which are interpreted as the upper 64 bits (high) or lower
   *    64 bits of a uint128_t variable
   *
   */
#ifndef PCG_EMULATED_128BIT_MATH
  state->pcg_state->state = (((pcg128_t)state_arr[0]) << 64) | state_arr[1];
  state->pcg_state->inc = (((pcg128_t)state_arr[2]) << 64) | state_arr[3];
#else
  state->pcg_state->state.high = state_arr[0];
  state->pcg_state->state.low = state_arr[1];
  state->pcg_state->inc.high = state_arr[2];
  state->pcg_state->inc.low = state_arr[3];
#endif
  state->has_uint32 = has_uint32;
  state->uinteger = uinteger;
}
