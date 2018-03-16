
#ifdef _WIN32
#ifndef _INTTYPES
#include "../common/stdint.h"
#endif
#define inline __inline __forceinline
#else
#include <inttypes.h>
#endif

#define PCG_DEFAULT_MULTIPLIER_64 6364136223846793005ULL

struct pcg_state_setseq_64 {
  uint64_t state;
  uint64_t inc;
};

static inline uint32_t pcg_rotr_32(uint32_t value, unsigned int rot) {
#if PCG_USE_INLINE_ASM && __clang__ && (__x86_64__ || __i386__)
  asm("rorl   %%cl, %0" : "=r"(value) : "0"(value), "c"(rot));
  return value;
#else
  return (value >> rot) | (value << ((-rot) & 31));
#endif
}

static inline void pcg_setseq_64_step_r(struct pcg_state_setseq_64 *rng) {
  rng->state = rng->state * PCG_DEFAULT_MULTIPLIER_64 + rng->inc;
}

static inline uint32_t pcg_output_xsh_rr_64_32(uint64_t state) {
  return pcg_rotr_32(((state >> 18u) ^ state) >> 27u, state >> 59u);
}

static inline uint32_t
pcg_setseq_64_xsh_rr_32_random_r(struct pcg_state_setseq_64 *rng) {
  uint64_t oldstate;
  oldstate = rng->state;
  pcg_setseq_64_step_r(rng);
  return pcg_output_xsh_rr_64_32(oldstate);
}

static inline void pcg_setseq_64_srandom_r(struct pcg_state_setseq_64 *rng,
                                           uint64_t initstate,
                                           uint64_t initseq) {
  rng->state = 0U;
  rng->inc = (initseq << 1u) | 1u;
  pcg_setseq_64_step_r(rng);
  rng->state += initstate;
  pcg_setseq_64_step_r(rng);
}

extern uint64_t pcg_advance_lcg_64(uint64_t state, uint64_t delta,
                                   uint64_t cur_mult, uint64_t cur_plus);

static inline void pcg_setseq_64_advance_r(struct pcg_state_setseq_64 *rng,
                                           uint64_t delta) {
  rng->state = pcg_advance_lcg_64(rng->state, delta, PCG_DEFAULT_MULTIPLIER_64,
                                  rng->inc);
}

typedef struct pcg_state_setseq_64 pcg32_random_t;
#define pcg32_random_r pcg_setseq_64_xsh_rr_32_random_r
#define pcg32_srandom_r pcg_setseq_64_srandom_r
#define pcg32_advance_r pcg_setseq_64_advance_r

typedef struct s_pcg32_state { pcg32_random_t *pcg_state; } pcg32_state;

static inline uint64_t pcg32_next64(pcg32_state *state) {
  return (uint64_t)(pcg32_random_r(state->pcg_state)) << 32 |
         pcg32_random_r(state->pcg_state);
}

static inline uint32_t pcg32_next32(pcg32_state *state) {
  return pcg32_random_r(state->pcg_state);
}

static inline double pcg32_next_double(pcg32_state *state) {
  int32_t a = pcg32_random_r(state->pcg_state) >> 5,
          b = pcg32_random_r(state->pcg_state) >> 6;
  return (a * 67108864.0 + b) / 9007199254740992.0;
}

void pcg32_advance_state(pcg32_state *state, uint64_t step);
void pcg32_set_seed(pcg32_state *state, uint64_t seed, uint64_t inc);
