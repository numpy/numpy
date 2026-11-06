#include "distributions_bulk.h"
#include "numpy/random/distributions.h"

/* Keep the original fill loops in distributions.c independent of bulk support. */
#define BITGEN_BULK_HAS(bulk, capability, callback)                         \
  ((bulk) != NULL && ((bulk)->capabilities & (capability)) != 0 &&          \
   (bulk)->callback != NULL)

void random_standard_uniform_fill_with_bulk(
    bitgen_t *bitgen_state, const bitgen_bulk_v1 *bulk,
    npy_intp cnt, double *out) {
  if (BITGEN_BULK_HAS(bulk, BITGEN_BULK_DOUBLE, fill_double)) {
    bulk->fill_double(bitgen_state->state, (size_t)cnt, out);
  } else {
    random_standard_uniform_fill(bitgen_state, cnt, out);
  }
}

void random_standard_uniform_fill_f_with_bulk(
    bitgen_t *bitgen_state, const bitgen_bulk_v1 *bulk,
    npy_intp cnt, float *out) {
  if (BITGEN_BULK_HAS(bulk, BITGEN_BULK_UINT32, fill_uint32)) {
    enum { BULK_UINT32_COUNT = 256 };
    uint32_t raw[BULK_UINT32_COUNT];
    npy_intp produced = 0;

    while (produced < cnt) {
      size_t requested = (size_t)(cnt - produced);
      size_t j;
      if (requested > BULK_UINT32_COUNT) {
        requested = BULK_UINT32_COUNT;
      }
      bulk->fill_uint32(bitgen_state->state, requested, raw);
      for (j = 0; j < requested; j++) {
        out[produced++] = (raw[j] >> 8) * (1.0f / 16777216.0f);
      }
    }
  } else {
    random_standard_uniform_fill_f(bitgen_state, cnt, out);
  }
}

void random_bounded_uint64_fill_with_bulk(
    bitgen_t *bitgen_state, const bitgen_bulk_v1 *bulk,
    uint64_t off, uint64_t rng, npy_intp cnt, bool use_masked,
    uint64_t *out) {
  if (rng == 0xFFFFFFFFUL && cnt >= 32 &&
      BITGEN_BULK_HAS(bulk, BITGEN_BULK_UINT32, fill_uint32)) {
    enum { BULK_UINT32_COUNT = 256 };
    uint32_t raw[BULK_UINT32_COUNT];
    npy_intp produced = 0;

    while (produced < cnt) {
      size_t requested = (size_t)(cnt - produced);
      size_t j;
      if (requested > BULK_UINT32_COUNT) {
        requested = BULK_UINT32_COUNT;
      }
      bulk->fill_uint32(bitgen_state->state, requested, raw);
      for (j = 0; j < requested; j++) {
        out[produced++] = off + (uint64_t)raw[j];
      }
    }
    return;
  }

  if (rng > 0 && rng < 0xFFFFFFFFUL && !use_masked && cnt >= 32 &&
      BITGEN_BULK_HAS(bulk, BITGEN_BULK_UINT32, fill_uint32)) {
    enum { BULK_UINT32_COUNT = 256 };
    uint32_t raw[BULK_UINT32_COUNT];
    const uint32_t rng_excl = (uint32_t)rng + 1;
    const uint32_t threshold = (UINT32_MAX - (uint32_t)rng) % rng_excl;
    npy_intp produced = 0;

    while (produced < cnt) {
      size_t requested = (size_t)(cnt - produced);
      size_t j;
      if (requested > BULK_UINT32_COUNT) {
        requested = BULK_UINT32_COUNT;
      }
      bulk->fill_uint32(bitgen_state->state, requested, raw);
      for (j = 0; j < requested; j++) {
        uint64_t product = ((uint64_t)raw[j]) * rng_excl;
        if ((uint32_t)product >= threshold) {
          out[produced++] = off + (uint64_t)(product >> 32);
        }
      }
    }
    return;
  }

  if (rng == UINT64_MAX && off == 0 &&
      BITGEN_BULK_HAS(bulk, BITGEN_BULK_UINT64, fill_uint64)) {
    bulk->fill_uint64(bitgen_state->state, (size_t)cnt, out);
    return;
  }

#if __SIZEOF_INT128__
  if (rng > UINT32_MAX && rng < UINT64_MAX && !use_masked && cnt >= 32 &&
      BITGEN_BULK_HAS(bulk, BITGEN_BULK_UINT64, fill_uint64)) {
    enum { BULK_UINT64_COUNT = 256 };
    uint64_t raw[BULK_UINT64_COUNT];
    const uint64_t rng_excl = rng + 1;
    const uint64_t threshold = (UINT64_MAX - rng) % rng_excl;
    npy_intp produced = 0;

    while (produced < cnt) {
      size_t requested = (size_t)(cnt - produced);
      size_t j;
      if (requested > BULK_UINT64_COUNT) {
        requested = BULK_UINT64_COUNT;
      }
      bulk->fill_uint64(bitgen_state->state, requested, raw);
      for (j = 0; j < requested; j++) {
        __uint128_t product = ((__uint128_t)raw[j]) * rng_excl;
        if ((uint64_t)product >= threshold) {
          out[produced++] = off + (uint64_t)(product >> 64);
        }
      }
    }
    return;
  }
#endif

  random_bounded_uint64_fill(bitgen_state, off, rng, cnt, use_masked, out);
}

void random_bounded_uint32_fill_with_bulk(
    bitgen_t *bitgen_state, const bitgen_bulk_v1 *bulk,
    uint32_t off, uint32_t rng, npy_intp cnt, bool use_masked,
    uint32_t *out) {
  if (rng == UINT32_MAX && off == 0 &&
      BITGEN_BULK_HAS(bulk, BITGEN_BULK_UINT32, fill_uint32)) {
    bulk->fill_uint32(bitgen_state->state, (size_t)cnt, out);
    return;
  }

  if (rng > 0 && rng < UINT32_MAX && !use_masked && cnt >= 32 &&
      BITGEN_BULK_HAS(bulk, BITGEN_BULK_UINT32, fill_uint32)) {
    enum { BULK_UINT32_COUNT = 256 };
    uint32_t raw[BULK_UINT32_COUNT];
    const uint32_t rng_excl = rng + 1;
    const uint32_t threshold = (UINT32_MAX - rng) % rng_excl;
    npy_intp produced = 0;

    while (produced < cnt) {
      size_t requested = (size_t)(cnt - produced);
      size_t j;
      if (requested > BULK_UINT32_COUNT) {
        requested = BULK_UINT32_COUNT;
      }
      bulk->fill_uint32(bitgen_state->state, requested, raw);
      for (j = 0; j < requested; j++) {
        uint64_t product = ((uint64_t)raw[j]) * rng_excl;
        if ((uint32_t)product >= threshold) {
          out[produced++] = off + (uint32_t)(product >> 32);
        }
      }
    }
    return;
  }

  random_bounded_uint32_fill(bitgen_state, off, rng, cnt, use_masked, out);
}
