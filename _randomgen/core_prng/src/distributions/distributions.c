#include "distributions.h"
#include "ziggurat.h"
#include "ziggurat_constants.h"

static NPY_INLINE float uint32_to_float(prng_t *prng_state)
{
    return (prng_state->next_uint32(prng_state->state) >> 9) * (1.0f / 8388608.0f);
}

uint32_t random_uint32(prng_t *prng_state) {
  return prng_state->next_uint32(prng_state->state);
}

float random_float(prng_t *prng_state) {
  return uint32_to_float(prng_state);
}

double random_double(prng_t *prng_state) {
  return prng_state->next_double(prng_state->state);
}

double random_standard_exponential(prng_t *prng_state) {
  return -log(1.0 - prng_state->next_double(prng_state->state));
}

float random_standard_exponential_float(prng_t *prng_state) {
    return -logf(1.0f - uint32_to_float(prng_state));
}

double random_gauss(prng_t *prng_state)
{
    if (prng_state->has_gauss)
    {
        const double temp = prng_state->gauss;
        prng_state->has_gauss = false;
        return temp;
    }
    else
    {
        double f, x1, x2, r2;

        do
        {
            x1 = 2.0 * prng_state->next_double(prng_state->state) - 1.0;
            x2 = 2.0 * prng_state->next_double(prng_state->state) - 1.0;
            r2 = x1 * x1 + x2 * x2;
        } while (r2 >= 1.0 || r2 == 0.0);

        /* Box-Muller transform */
        f = sqrt(-2.0 * log(r2) / r2);
        /* Keep for next call */
        prng_state->gauss = f * x1;
        prng_state->has_gauss = true;
        return f * x2;
    }
}

float random_gauss_float(prng_t *prng_state)
{
    if (prng_state->has_gauss_f)
    {
        const float temp = prng_state->gauss_f;
        prng_state->has_gauss_f = false;
        return temp;
    }
    else
    {
        float f, x1, x2, r2;

        do
        {
            x1 = 2.0f * uint32_to_float(prng_state) - 1.0f;
            x2 = 2.0f * uint32_to_float(prng_state) - 1.0f;
            r2 = x1 * x1 + x2 * x2;
        } while (r2 >= 1.0 || r2 == 0.0);

        /* Box-Muller transform */
        f = sqrtf(-2.0f * logf(r2) / r2);
        /* Keep for next call */
        prng_state->gauss_f = f * x1;
        prng_state->has_gauss_f = true;
        return f * x2;
    }
}

double standard_exponential_zig_double(prng_t *prng_state);

static double standard_exponential_zig_double_unlikely(prng_t *prng_state, uint8_t idx, double x)
{
    if (idx == 0)
    {
        return ziggurat_exp_r - log(random_double(prng_state));
    }
    else if ((fe_double[idx - 1] - fe_double[idx]) * random_double(prng_state) + fe_double[idx] < exp(-x))
    {
        return x;
    }
    else
    {
        return standard_exponential_zig_double(prng_state);
    }
}

double standard_exponential_zig_double(prng_t *prng_state)
{
    uint64_t ri;
    uint8_t idx;
    double x;
    ri = prng_state->next_uint64(prng_state->state);
    ri >>= 3;
    idx = ri & 0xFF;
    ri >>= 8;
    x = ri * we_double[idx];
    if (ri < ke_double[idx])
    {
        return x; // 98.9% of the time we return here 1st try
    }
    return standard_exponential_zig_double_unlikely(prng_state, idx, x);
}

double random_standard_exponential_zig_double(prng_t *prng_state)
{
    return standard_exponential_zig_double(prng_state);
}