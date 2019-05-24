#include "pcg64.orig.h"

extern inline void pcg_setseq_128_srandom_r(pcg64_random_t *rng,
                                            pcg128_t initstate,
                                            pcg128_t initseq);

extern uint64_t pcg_rotr_64(uint64_t value, unsigned int rot);
extern inline uint64_t pcg_output_xsl_rr_128_64(pcg128_t state);
extern void pcg_setseq_128_step_r(struct pcg_state_setseq_128 *rng);
extern uint64_t
pcg_setseq_128_xsl_rr_64_random_r(struct pcg_state_setseq_128 *rng);
