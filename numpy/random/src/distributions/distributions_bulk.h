#ifndef NUMPY_RANDOM_DISTRIBUTIONS_BULK_H_
#define NUMPY_RANDOM_DISTRIBUTIONS_BULK_H_

#include "numpy/random/bitgen.h"
#include "numpy/random/bitgen_bulk.h"
#include "numpy/ndarraytypes.h"

void random_standard_uniform_fill_with_bulk(
    bitgen_t *bitgen_state, const bitgen_bulk_v1 *bulk,
    npy_intp cnt, double *out);
void random_standard_uniform_fill_f_with_bulk(
    bitgen_t *bitgen_state, const bitgen_bulk_v1 *bulk,
    npy_intp cnt, float *out);
void random_bounded_uint64_fill_with_bulk(
    bitgen_t *bitgen_state, const bitgen_bulk_v1 *bulk,
    uint64_t off, uint64_t rng, npy_intp cnt, bool use_masked,
    uint64_t *out);
void random_bounded_uint32_fill_with_bulk(
    bitgen_t *bitgen_state, const bitgen_bulk_v1 *bulk,
    uint32_t off, uint32_t rng, npy_intp cnt, bool use_masked,
    uint32_t *out);

#endif  /* NUMPY_RANDOM_DISTRIBUTIONS_BULK_H_ */
