#include "distributions.h"
#include "ziggurat.h"
#include "ziggurat_constants.h"

static NPY_INLINE float next_float(prng_t *prng_state){
    return (prng_state->next_uint32(prng_state->state) >> 9) * (1.0f / 8388608.0f);
}

uint32_t random_uint32(prng_t *prng_state) {
  return prng_state->next_uint32(prng_state->state);
}

float random_sample_f(prng_t *prng_state) {
  return next_float(prng_state);
}

double random_sample(prng_t *prng_state) {
  return prng_state->next_double(prng_state->state);
}

double random_standard_exponential(prng_t *prng_state) {
  return -log(1.0 - prng_state->next_double(prng_state->state));
}

float random_standard_exponential_f(prng_t *prng_state) {
    return -logf(1.0f - next_float(prng_state));
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

float random_gauss_f(prng_t *prng_state)
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
            x1 = 2.0f * next_float(prng_state) - 1.0f;
            x2 = 2.0f * next_float(prng_state) - 1.0f;
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

double standard_exponential_zig(prng_t *prng_state);

static double standard_exponential_zig_unlikely(prng_t *prng_state, uint8_t idx, double x)
{
    if (idx == 0)
    {
        return ziggurat_exp_r - log(prng_state->next_double(prng_state->state));
    }
    else if ((fe_double[idx - 1] - fe_double[idx]) * prng_state->next_double(prng_state->state) + fe_double[idx] < exp(-x))
    {
        return x;
    }
    else
    {
        return standard_exponential_zig(prng_state);
    }
}

double standard_exponential_zig(prng_t *prng_state)
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
    return standard_exponential_zig_unlikely(prng_state, idx, x);
}

double random_standard_exponential_zig(prng_t *prng_state)
{
    return standard_exponential_zig(prng_state);
}

static NPY_INLINE float standard_exponential_zig_f(prng_t *prng_state);

static float standard_exponential_zig_unlikely_f(prng_t *prng_state, uint8_t idx, float x)
{
    if (idx == 0)
    {
        return ziggurat_exp_r_f - logf(next_float(prng_state));
    }
    else if ((fe_float[idx - 1] - fe_float[idx]) * next_float(prng_state) + fe_float[idx] < expf(-x))
    {
        return x;
    }
    else
    {
        return standard_exponential_zig_f(prng_state);
    }
}

static NPY_INLINE float standard_exponential_zig_f(prng_t *prng_state)
{
    uint32_t ri;
    uint8_t idx;
    float x;
    ri = prng_state->next_uint32(prng_state->state);
    ri >>= 1;
    idx = ri & 0xFF;
    ri >>= 8;
    x = ri * we_float[idx];
    if (ri < ke_float[idx])
    {
        return x; // 98.9% of the time we return here 1st try
    }
    return standard_exponential_zig_unlikely_f(prng_state, idx, x);
}

float random_standard_exponential_zig_f(prng_t *prng_state)
{
    return standard_exponential_zig_f(prng_state);
}


double random_gauss_zig(prng_t* prng_state)
{
    uint64_t r;
    int sign;
    int64_t rabs;
    int idx;
    double x, xx, yy;
    for (;;)
    {
        /* r = e3n52sb8 */
        r = prng_state->next_uint64(prng_state->state);
        idx = r & 0xff;
        r >>= 8;
        sign = r & 0x1;
        rabs = (int64_t)((r >> 1) & 0x000fffffffffffff);
        x = rabs * wi_double[idx];
        if (sign & 0x1)
            x = -x;
        if (rabs < ki_double[idx])
            return x; // # 99.3% of the time return here
        if (idx == 0)
        {
            for (;;)
            {
                xx = -ziggurat_nor_inv_r * log(prng_state->next_double(prng_state->state));
                yy = -log(prng_state->next_double(prng_state->state));
                if (yy + yy > xx * xx)
                    return ((rabs >> 8) & 0x1) ? -(ziggurat_nor_r + xx) : ziggurat_nor_r + xx;
            }
        }
        else
        {
            if (((fi_double[idx - 1] - fi_double[idx]) * prng_state->next_double(prng_state->state) + fi_double[idx]) < exp(-0.5 * x * x))
                return x;
        }
    }
}

float random_gauss_zig_f(prng_t* prng_state)
{
    uint32_t r;
    int sign;
    int32_t rabs;
    int idx;
    float x, xx, yy;
    for (;;)
    {
        /* r = n23sb8 */
        r = prng_state->next_uint32(prng_state->state);
        idx = r & 0xff;
        sign = (r >> 8) & 0x1;
        rabs = (int32_t)((r >> 9) & 0x0007fffff);
        x = rabs * wi_float[idx];
        if (sign & 0x1)
            x = -x;
        if (rabs < ki_float[idx])
            return x; // # 99.3% of the time return here
        if (idx == 0)
        {
            for (;;)
            {
                xx = -ziggurat_nor_inv_r_f * logf(next_float(prng_state));
                yy = -logf(next_float(prng_state));
                if (yy + yy > xx * xx)
                    return ((rabs >> 8) & 0x1) ? -(ziggurat_nor_r_f + xx) : ziggurat_nor_r_f + xx;
            }
        }
        else
        {
            if (((fi_float[idx - 1] - fi_float[idx]) * next_float(prng_state) + fi_float[idx]) < exp(-0.5 * x * x))
                return x;
        }
    }
}

static NPY_INLINE double standard_gamma(prng_t* prng_state, double shape)
{
    double b, c;
    double U, V, X, Y;

    if (shape == 1.0)
    {
        return random_standard_exponential(prng_state);
    }
    else if (shape < 1.0)
    {
        for (;;)
        {
            U = prng_state->next_double(prng_state->state);
            V = random_standard_exponential(prng_state);
            if (U <= 1.0 - shape)
            {
                X = pow(U, 1. / shape);
                if (X <= V)
                {
                    return X;
                }
            }
            else
            {
                Y = -log((1 - U) / shape);
                X = pow(1.0 - shape + shape * Y, 1. / shape);
                if (X <= (V + Y))
                {
                    return X;
                }
            }
        }
    }
    else
    {
        b = shape - 1. / 3.;
        c = 1. / sqrt(9 * b);
        for (;;)
        {
            do
            {
                X = random_gauss(prng_state);
                V = 1.0 + c * X;
            } while (V <= 0.0);

            V = V * V * V;
            U = random_sample(prng_state);
            if (U < 1.0 - 0.0331 * (X * X) * (X * X))
                return (b * V);
            if (log(U) < 0.5 * X * X + b * (1. - V + log(V)))
                return (b * V);
        }
    }
}

static NPY_INLINE float standard_gamma_float(prng_t *prng_state, float shape)
{
    float b, c;
    float U, V, X, Y;

    if (shape == 1.0f)
    {
        return random_standard_exponential_f(prng_state);
    }
    else if (shape < 1.0f)
    {
        for (;;)
        {
            U = random_sample_f(prng_state);
            V = random_standard_exponential_f(prng_state);
            if (U <= 1.0f - shape)
            {
                X = powf(U, 1.0f / shape);
                if (X <= V)
                {
                    return X;
                }
            }
            else
            {
                Y = -logf((1.0f - U) / shape);
                X = powf(1.0f - shape + shape * Y, 1.0f / shape);
                if (X <= (V + Y))
                {
                    return X;
                }
            }
        }
    }
    else
    {
        b = shape - 1.0f / 3.0f;
        c = 1.0f / sqrtf(9.0f * b);
        for (;;)
        {
            do
            {
                X = random_gauss_f(prng_state);
                V = 1.0f + c * X;
            } while (V <= 0.0f);

            V = V * V * V;
            U = random_sample_f(prng_state);
            if (U < 1.0f - 0.0331f * (X * X) * (X * X))
                return (b * V);
            if (logf(U) < 0.5f * X * X + b * (1.0f - V + logf(V)))
                return (b * V);
        }
    }
}


double random_standard_gamma(prng_t* prng_state, double shape)
{
    return standard_gamma(prng_state, shape);
}

float random_standard_gamma_f(prng_t* prng_state, float shape)
{
    return standard_gamma_float(prng_state, shape);
}


static NPY_INLINE double standard_gamma_zig(prng_t *prng_state, double shape)
{
    double b, c;
    double U, V, X, Y;

    if (shape == 1.0)
    {
        return random_standard_exponential_zig(prng_state);
    }
    else if (shape < 1.0)
    {
        for (;;)
        {
            U = random_sample(prng_state);
            V = random_standard_exponential_zig(prng_state);
            if (U <= 1.0 - shape)
            {
                X = pow(U, 1. / shape);
                if (X <= V)
                {
                    return X;
                }
            }
            else
            {
                Y = -log((1 - U) / shape);
                X = pow(1.0 - shape + shape * Y, 1. / shape);
                if (X <= (V + Y))
                {
                    return X;
                }
            }
        }
    }
    else
    {
        b = shape - 1. / 3.;
        c = 1. / sqrt(9 * b);
        for (;;)
        {
            do
            {
                X = random_gauss_zig(prng_state);
                V = 1.0 + c * X;
            } while (V <= 0.0);

            V = V * V * V;
            U = random_sample(prng_state);
            if (U < 1.0 - 0.0331 * (X * X) * (X * X))
                return (b * V);
            if (log(U) < 0.5 * X * X + b * (1. - V + log(V)))
                return (b * V);
        }
    }
}

static NPY_INLINE float standard_gamma_zig_f(prng_t *prng_state, float shape)
{
    float b, c;
    float U, V, X, Y;

    if (shape == 1.0f)
    {
        return random_standard_exponential_zig_f(prng_state);
    }
    else if (shape < 1.0f)
    {
        for (;;)
        {
            U = random_sample_f(prng_state);
            V = random_standard_exponential_zig_f(prng_state);
            if (U <= 1.0f - shape)
            {
                X = powf(U, 1.0f / shape);
                if (X <= V)
                {
                    return X;
                }
            }
            else
            {
                Y = -logf((1.0f - U) / shape);
                X = powf(1.0f - shape + shape * Y, 1.0f / shape);
                if (X <= (V + Y))
                {
                    return X;
                }
            }
        }
    }
    else
    {
        b = shape - 1.0f / 3.0f;
        c = 1.0f / sqrtf(9.0f * b);
        for (;;)
        {
            do
            {
                X = random_gauss_zig_f(prng_state);
                V = 1.0f + c * X;
            } while (V <= 0.0f);

            V = V * V * V;
            U = random_sample_f(prng_state);
            if (U < 1.0f - 0.0331f * (X * X) * (X * X))
                return (b * V);
            if (logf(U) < 0.5f * X * X + b * (1.0f - V + logf(V)))
                return (b * V);
        }
    }
}

double random_standard_gamma_zig(prng_t *prng_state, double shape)
{
    return standard_gamma_zig(prng_state, shape);
}

float random_standard_gamma_zig_f(prng_t *prng_state, float shape)
{
    return standard_gamma_zig_f(prng_state, shape);
}
