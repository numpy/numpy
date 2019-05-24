#include "pcg32.h"

extern inline uint64_t pcg32_next64(pcg32_state *state);
extern inline uint32_t pcg32_next32(pcg32_state *state);
extern inline double pcg32_next_double(pcg32_state *state);

uint64_t pcg_advance_lcg_64(uint64_t state, uint64_t delta, uint64_t cur_mult,
                            uint64_t cur_plus) {
  uint64_t acc_mult, acc_plus;
  acc_mult = 1u;
  acc_plus = 0u;
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

extern void pcg32_advance_state(pcg32_state *state, uint64_t step) {
  pcg32_advance_r(state->pcg_state, step);
}

extern void pcg32_set_seed(pcg32_state *state, uint64_t seed, uint64_t inc) {
  pcg32_srandom_r(state->pcg_state, seed, inc);
}
