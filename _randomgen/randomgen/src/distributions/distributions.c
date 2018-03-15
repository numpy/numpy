#include "distributions.h"
#include "ziggurat.h"
#include "ziggurat_constants.h"

/* Random generators for external use */
float random_float(brng_t *brng_state) { return next_float(brng_state); }

double random_double(brng_t *brng_state) { return next_double(brng_state); }

static NPY_INLINE double next_standard_exponential(brng_t *brng_state)
{
    return -log(1.0 - next_double(brng_state));
}

double random_standard_exponential(brng_t *brng_state) {
  return next_standard_exponential(brng_state);
}

void random_standard_exponential_fill(brng_t *brng_state, npy_intp cnt, double *out)
{
  npy_intp i;
  for (i = 0; i < cnt; i++) {
    out[i] = next_standard_exponential(brng_state);
  }
}

float random_standard_exponential_f(brng_t *brng_state) {
  return -logf(1.0f - next_float(brng_state));
}

void random_double_fill(brng_t* brng_state, npy_intp cnt, double *out)
{
  npy_intp i;
  for (i = 0; i < cnt; i++) {
    out[i] = next_double(brng_state);
  }
}
/*
double random_gauss(brng_t *brng_state) {
  if (brng_state->has_gauss) {
    const double temp = brng_state->gauss;
    brng_state->has_gauss = false;
    brng_state->gauss = 0.0;
    return temp;
  } else {
    double f, x1, x2, r2;

    do {
      x1 = 2.0 * next_double(brng_state) - 1.0;
      x2 = 2.0 * next_double(brng_state) - 1.0;
      r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 == 0.0);

    // Box-Muller transform
    f = sqrt(-2.0 * log(r2) / r2);
    // Keep for next call
    brng_state->gauss = f * x1;
    brng_state->has_gauss = true;
    return f * x2;
  }
}

float random_gauss_f(brng_t *brng_state) {
  if (brng_state->has_gauss_f) {
    const float temp = brng_state->gauss_f;
    brng_state->has_gauss_f = false;
    brng_state->gauss_f = 0.0f;
    return temp;
  } else {
    float f, x1, x2, r2;

    do {
      x1 = 2.0f * next_float(brng_state) - 1.0f;
      x2 = 2.0f * next_float(brng_state) - 1.0f;
      r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 == 0.0);

    // Box-Muller transform
    f = sqrtf(-2.0f * logf(r2) / r2);
    // Keep for next call
    brng_state->gauss_f = f * x1;
    brng_state->has_gauss_f = true;
    return f * x2;
  }
}
*/

static NPY_INLINE double standard_exponential_zig(brng_t *brng_state);

static double standard_exponential_zig_unlikely(brng_t *brng_state, uint8_t idx,
                                                double x) {
  if (idx == 0) {
    return ziggurat_exp_r - log(next_double(brng_state));
  } else if ((fe_double[idx - 1] - fe_double[idx]) * next_double(brng_state) +
                 fe_double[idx] <
             exp(-x)) {
    return x;
  } else {
    return standard_exponential_zig(brng_state);
  }
}

static NPY_INLINE double standard_exponential_zig(brng_t *brng_state) {
  uint64_t ri;
  uint8_t idx;
  double x;
  ri = next_uint64(brng_state);
  ri >>= 3;
  idx = ri & 0xFF;
  ri >>= 8;
  x = ri * we_double[idx];
  if (ri < ke_double[idx]) {
    return x; // 98.9% of the time we return here 1st try
  }
  return standard_exponential_zig_unlikely(brng_state, idx, x);
}

double random_standard_exponential_zig(brng_t *brng_state) {
  return standard_exponential_zig(brng_state);
}


void random_standard_exponential_zig_fill(brng_t *brng_state, npy_intp cnt, double *out)
{
  npy_intp i;
  for (i = 0; i < cnt; i++) {
    out[i] = standard_exponential_zig(brng_state);
  }
}


static NPY_INLINE float standard_exponential_zig_f(brng_t *brng_state);

static float standard_exponential_zig_unlikely_f(brng_t *brng_state,
                                                 uint8_t idx, float x) {
  if (idx == 0) {
    return ziggurat_exp_r_f - logf(next_float(brng_state));
  } else if ((fe_float[idx - 1] - fe_float[idx]) * next_float(brng_state) +
                 fe_float[idx] <
             expf(-x)) {
    return x;
  } else {
    return standard_exponential_zig_f(brng_state);
  }
}

static NPY_INLINE float standard_exponential_zig_f(brng_t *brng_state) {
  uint32_t ri;
  uint8_t idx;
  float x;
  ri = next_uint32(brng_state);
  ri >>= 1;
  idx = ri & 0xFF;
  ri >>= 8;
  x = ri * we_float[idx];
  if (ri < ke_float[idx]) {
    return x; // 98.9% of the time we return here 1st try
  }
  return standard_exponential_zig_unlikely_f(brng_state, idx, x);
}

float random_standard_exponential_zig_f(brng_t *brng_state) {
  return standard_exponential_zig_f(brng_state);
}

static NPY_INLINE double next_gauss_zig(brng_t *brng_state) {
  uint64_t r;
  int sign;
  int64_t rabs;
  int idx;
  double x, xx, yy;
  for (;;) {
    /* r = e3n52sb8 */
    r = next_uint64(brng_state);
    idx = r & 0xff;
    r >>= 8;
    sign = r & 0x1;
    rabs = (int64_t)((r >> 1) & 0x000fffffffffffff);
    x = rabs * wi_double[idx];
    if (sign & 0x1)
      x = -x;
    if (rabs < ki_double[idx])
      return x; // # 99.3% of the time return here
    if (idx == 0) {
      for (;;) {
        xx = -ziggurat_nor_inv_r * log(next_double(brng_state));
        yy = -log(next_double(brng_state));
        if (yy + yy > xx * xx)
          return ((rabs >> 8) & 0x1) ? -(ziggurat_nor_r + xx)
                                     : ziggurat_nor_r + xx;
      }
    } else {
      if (((fi_double[idx - 1] - fi_double[idx]) * next_double(brng_state) +
           fi_double[idx]) < exp(-0.5 * x * x))
        return x;
    }
  }
}

double random_gauss_zig(brng_t *brng_state) {
  return next_gauss_zig(brng_state);
}

void random_gauss_zig_fill(brng_t *brng_state, npy_intp cnt, double *out) {
  npy_intp i;
  for (i = 0; i < cnt; i++) {
    out[i] = next_gauss_zig(brng_state);
  }
}

float random_gauss_zig_f(brng_t *brng_state) {
  uint32_t r;
  int sign;
  int32_t rabs;
  int idx;
  float x, xx, yy;
  for (;;) {
    /* r = n23sb8 */
    r = next_uint32(brng_state);
    idx = r & 0xff;
    sign = (r >> 8) & 0x1;
    rabs = (int32_t)((r >> 9) & 0x0007fffff);
    x = rabs * wi_float[idx];
    if (sign & 0x1)
      x = -x;
    if (rabs < ki_float[idx])
      return x; // # 99.3% of the time return here
    if (idx == 0) {
      for (;;) {
        xx = -ziggurat_nor_inv_r_f * logf(next_float(brng_state));
        yy = -logf(next_float(brng_state));
        if (yy + yy > xx * xx)
          return ((rabs >> 8) & 0x1) ? -(ziggurat_nor_r_f + xx)
                                     : ziggurat_nor_r_f + xx;
      }
    } else {
      if (((fi_float[idx - 1] - fi_float[idx]) * next_float(brng_state) +
           fi_float[idx]) < exp(-0.5 * x * x))
        return x;
    }
  }
}

/*
static NPY_INLINE double standard_gamma(brng_t *brng_state, double shape) {
  double b, c;
  double U, V, X, Y;

  if (shape == 1.0) {
    return random_standard_exponential(brng_state);
  } else if (shape < 1.0) {
    for (;;) {
      U = next_double(brng_state);
      V = random_standard_exponential(brng_state);
      if (U <= 1.0 - shape) {
        X = pow(U, 1. / shape);
        if (X <= V) {
          return X;
        }
      } else {
        Y = -log((1 - U) / shape);
        X = pow(1.0 - shape + shape * Y, 1. / shape);
        if (X <= (V + Y)) {
          return X;
        }
      }
    }
  } else {
    b = shape - 1. / 3.;
    c = 1. / sqrt(9 * b);
    for (;;) {
      do {
        X = random_gauss(brng_state);
        V = 1.0 + c * X;
      } while (V <= 0.0);

      V = V * V * V;
      U = next_double(brng_state);
      if (U < 1.0 - 0.0331 * (X * X) * (X * X))
        return (b * V);
      if (log(U) < 0.5 * X * X + b * (1. - V + log(V)))
        return (b * V);
    }
  }
}

static NPY_INLINE float standard_gamma_float(brng_t *brng_state, float shape) {
  float b, c;
  float U, V, X, Y;

  if (shape == 1.0f) {
    return random_standard_exponential_f(brng_state);
  } else if (shape < 1.0f) {
    for (;;) {
      U = next_float(brng_state);
      V = random_standard_exponential_f(brng_state);
      if (U <= 1.0f - shape) {
        X = powf(U, 1.0f / shape);
        if (X <= V) {
          return X;
        }
      } else {
        Y = -logf((1.0f - U) / shape);
        X = powf(1.0f - shape + shape * Y, 1.0f / shape);
        if (X <= (V + Y)) {
          return X;
        }
      }
    }
  } else {
    b = shape - 1.0f / 3.0f;
    c = 1.0f / sqrtf(9.0f * b);
    for (;;) {
      do {
        X = random_gauss_f(brng_state);
        V = 1.0f + c * X;
      } while (V <= 0.0f);

      V = V * V * V;
      U = next_float(brng_state);
      if (U < 1.0f - 0.0331f * (X * X) * (X * X))
        return (b * V);
      if (logf(U) < 0.5f * X * X + b * (1.0f - V + logf(V)))
        return (b * V);
    }
  }
}


double random_standard_gamma(brng_t *brng_state, double shape) {
  return standard_gamma(brng_state, shape);
}

float random_standard_gamma_f(brng_t *brng_state, float shape) {
  return standard_gamma_float(brng_state, shape);
}
*/

static NPY_INLINE double standard_gamma_zig(brng_t *brng_state, double shape) {
  double b, c;
  double U, V, X, Y;

  if (shape == 1.0) {
    return random_standard_exponential_zig(brng_state);
  } else if (shape < 1.0) {
    for (;;) {
      U = next_double(brng_state);
      V = random_standard_exponential_zig(brng_state);
      if (U <= 1.0 - shape) {
        X = pow(U, 1. / shape);
        if (X <= V) {
          return X;
        }
      } else {
        Y = -log((1 - U) / shape);
        X = pow(1.0 - shape + shape * Y, 1. / shape);
        if (X <= (V + Y)) {
          return X;
        }
      }
    }
  } else {
    b = shape - 1. / 3.;
    c = 1. / sqrt(9 * b);
    for (;;) {
      do {
        X = random_gauss_zig(brng_state);
        V = 1.0 + c * X;
      } while (V <= 0.0);

      V = V * V * V;
      U = next_double(brng_state);
      if (U < 1.0 - 0.0331 * (X * X) * (X * X))
        return (b * V);
      if (log(U) < 0.5 * X * X + b * (1. - V + log(V)))
        return (b * V);
    }
  }
}

static NPY_INLINE float standard_gamma_zig_f(brng_t *brng_state, float shape) {
  float b, c;
  float U, V, X, Y;

  if (shape == 1.0f) {
    return random_standard_exponential_zig_f(brng_state);
  } else if (shape < 1.0f) {
    for (;;) {
      U = next_float(brng_state);
      V = random_standard_exponential_zig_f(brng_state);
      if (U <= 1.0f - shape) {
        X = powf(U, 1.0f / shape);
        if (X <= V) {
          return X;
        }
      } else {
        Y = -logf((1.0f - U) / shape);
        X = powf(1.0f - shape + shape * Y, 1.0f / shape);
        if (X <= (V + Y)) {
          return X;
        }
      }
    }
  } else {
    b = shape - 1.0f / 3.0f;
    c = 1.0f / sqrtf(9.0f * b);
    for (;;) {
      do {
        X = random_gauss_zig_f(brng_state);
        V = 1.0f + c * X;
      } while (V <= 0.0f);

      V = V * V * V;
      U = next_float(brng_state);
      if (U < 1.0f - 0.0331f * (X * X) * (X * X))
        return (b * V);
      if (logf(U) < 0.5f * X * X + b * (1.0f - V + logf(V)))
        return (b * V);
    }
  }
}

double random_standard_gamma_zig(brng_t *brng_state, double shape) {
  return standard_gamma_zig(brng_state, shape);
}

float random_standard_gamma_zig_f(brng_t *brng_state, float shape) {
  return standard_gamma_zig_f(brng_state, shape);
}

int64_t random_positive_int64(brng_t *brng_state) {
  return next_uint64(brng_state) >> 1;
}

int32_t random_positive_int32(brng_t *brng_state) {
  return next_uint32(brng_state) >> 1;
}

int64_t random_positive_int(brng_t *brng_state) {
#if ULONG_MAX <= 0xffffffffUL
  return (int64_t)(next_uint32(brng_state) >> 1);
#else
  return (int64_t)(next_uint64(brng_state) >> 1);
#endif
}

uint64_t random_uint(brng_t *brng_state) {
#if ULONG_MAX <= 0xffffffffUL
  return next_uint32(brng_state);
#else
  return next_uint64(brng_state);
#endif
}

/*
 * log-gamma function to support some of these distributions. The
 * algorithm comes from SPECFUN by Shanjie Zhang and Jianming Jin and their
 * book "Computation of Special Functions", 1996, John Wiley & Sons, Inc.
 */
static double loggam(double x) {
  double x0, x2, xp, gl, gl0;
  int64_t k, n;

  static double a[10] = {8.333333333333333e-02, -2.777777777777778e-03,
                         7.936507936507937e-04, -5.952380952380952e-04,
                         8.417508417508418e-04, -1.917526917526918e-03,
                         6.410256410256410e-03, -2.955065359477124e-02,
                         1.796443723688307e-01, -1.39243221690590e+00};
  x0 = x;
  n = 0;
  if ((x == 1.0) || (x == 2.0)) {
    return 0.0;
  } else if (x <= 7.0) {
    n = (int64_t)(7 - x);
    x0 = x + n;
  }
  x2 = 1.0 / (x0 * x0);
  xp = 2 * M_PI;
  gl0 = a[9];
  for (k = 8; k >= 0; k--) {
    gl0 *= x2;
    gl0 += a[k];
  }
  gl = gl0 / x0 + 0.5 * log(xp) + (x0 - 0.5) * log(x0) - x0;
  if (x <= 7.0) {
    for (k = 1; k <= n; k++) {
      gl -= log(x0 - 1.0);
      x0 -= 1.0;
    }
  }
  return gl;
}

/*
double random_normal(brng_t *brng_state, double loc, double scale) {
  return loc + scale * random_gauss(brng_state);
}
*/

double random_normal_zig(brng_t *brng_state, double loc, double scale) {
  return loc + scale * random_gauss_zig(brng_state);
}

double random_exponential(brng_t *brng_state, double scale) {
  return scale * random_standard_exponential(brng_state);
}

double random_uniform(brng_t *brng_state, double lower, double range) {
  return lower + range * next_double(brng_state);
}

double random_gamma(brng_t *brng_state, double shape, double scale) {
  return scale * random_standard_gamma_zig(brng_state, shape);
}

float random_gamma_float(brng_t *brng_state, float shape, float scale) {
  return scale * random_standard_gamma_zig_f(brng_state, shape);
}

double random_beta(brng_t *brng_state, double a, double b) {
  double Ga, Gb;

  if ((a <= 1.0) && (b <= 1.0)) {
    double U, V, X, Y;
    /* Use Johnk's algorithm */

    while (1) {
      U = next_double(brng_state);
      V = next_double(brng_state);
      X = pow(U, 1.0 / a);
      Y = pow(V, 1.0 / b);

      if ((X + Y) <= 1.0) {
        if (X + Y > 0) {
          return X / (X + Y);
        } else {
          double logX = log(U) / a;
          double logY = log(V) / b;
          double logM = logX > logY ? logX : logY;
          logX -= logM;
          logY -= logM;

          return exp(logX - log(exp(logX) + exp(logY)));
        }
      }
    }
  } else {
    Ga = random_standard_gamma_zig(brng_state, a);
    Gb = random_standard_gamma_zig(brng_state, b);
    return Ga / (Ga + Gb);
  }
}

double random_chisquare(brng_t *brng_state, double df) {
  return 2.0 * random_standard_gamma_zig(brng_state, df / 2.0);
}

double random_f(brng_t *brng_state, double dfnum, double dfden) {
  return ((random_chisquare(brng_state, dfnum) * dfden) /
          (random_chisquare(brng_state, dfden) * dfnum));
}

double random_standard_cauchy(brng_t *brng_state) {
  return random_gauss_zig(brng_state) / random_gauss_zig(brng_state);
}

double random_pareto(brng_t *brng_state, double a) {
  return exp(random_standard_exponential(brng_state) / a) - 1;
}

double random_weibull(brng_t *brng_state, double a) {
  return pow(random_standard_exponential(brng_state), 1. / a);
}

double random_power(brng_t *brng_state, double a) {
  return pow(1 - exp(-random_standard_exponential(brng_state)), 1. / a);
}

double random_laplace(brng_t *brng_state, double loc, double scale) {
  double U;

  U = next_double(brng_state);
  if (U < 0.5) {
    U = loc + scale * log(U + U);
  } else {
    U = loc - scale * log(2.0 - U - U);
  }
  return U;
}

double random_gumbel(brng_t *brng_state, double loc, double scale) {
  double U;

  U = 1.0 - next_double(brng_state);
  return loc - scale * log(-log(U));
}

double random_logistic(brng_t *brng_state, double loc, double scale) {
  double U;

  U = next_double(brng_state);
  return loc + scale * log(U / (1.0 - U));
}

double random_lognormal(brng_t *brng_state, double mean, double sigma) {
  return exp(random_normal_zig(brng_state, mean, sigma));
}

double random_rayleigh(brng_t *brng_state, double mode) {
  return mode * sqrt(-2.0 * log(1.0 - next_double(brng_state)));
}

double random_standard_t(brng_t *brng_state, double df) {
  double num, denom;

  num = random_gauss_zig(brng_state);
  denom = random_standard_gamma_zig(brng_state, df / 2);
  return sqrt(df / 2) * num / sqrt(denom);
}

static int64_t random_poisson_mult(brng_t *brng_state, double lam) {
  int64_t X;
  double prod, U, enlam;

  enlam = exp(-lam);
  X = 0;
  prod = 1.0;
  while (1) {
    U = next_double(brng_state);
    prod *= U;
    if (prod > enlam) {
      X += 1;
    } else {
      return X;
    }
  }
}

/*
 * The transformed rejection method for generating Poisson random variables
 * W. Hoermann
 * Insurance: Mathematics and Economics 12, 39-45 (1993)
 */
#define LS2PI 0.91893853320467267
#define TWELFTH 0.083333333333333333333333
static int64_t random_poisson_ptrs(brng_t *brng_state, double lam) {
  int64_t k;
  double U, V, slam, loglam, a, b, invalpha, vr, us;

  slam = sqrt(lam);
  loglam = log(lam);
  b = 0.931 + 2.53 * slam;
  a = -0.059 + 0.02483 * b;
  invalpha = 1.1239 + 1.1328 / (b - 3.4);
  vr = 0.9277 - 3.6224 / (b - 2);

  while (1) {
    U = next_double(brng_state) - 0.5;
    V = next_double(brng_state);
    us = 0.5 - fabs(U);
    k = (int64_t)floor((2 * a / us + b) * U + lam + 0.43);
    if ((us >= 0.07) && (V <= vr)) {
      return k;
    }
    if ((k < 0) || ((us < 0.013) && (V > us))) {
      continue;
    }
    if ((log(V) + log(invalpha) - log(a / (us * us) + b)) <=
        (-lam + k * loglam - loggam(k + 1))) {
      return k;
    }
  }
}

int64_t random_poisson(brng_t *brng_state, double lam) {
  if (lam >= 10) {
    return random_poisson_ptrs(brng_state, lam);
  } else if (lam == 0) {
    return 0;
  } else {
    return random_poisson_mult(brng_state, lam);
  }
}

int64_t random_negative_binomial(brng_t *brng_state, double n, double p) {
  double Y = random_gamma(brng_state, n, (1 - p) / p);
  return random_poisson(brng_state, Y);
}

int64_t random_binomial_btpe(brng_t *brng_state, int64_t n, double p,
                             binomial_t *binomial) {
  double r, q, fm, p1, xm, xl, xr, c, laml, lamr, p2, p3, p4;
  double a, u, v, s, F, rho, t, A, nrq, x1, x2, f1, f2, z, z2, w, w2, x;
  int64_t m, y, k, i;

  if (!(binomial->has_binomial) || (binomial->nsave != n) ||
      (binomial->psave != p)) {
    /* initialize */
    binomial->nsave = n;
    binomial->psave = p;
    binomial->has_binomial = 1;
    binomial->r = r = min(p, 1.0 - p);
    binomial->q = q = 1.0 - r;
    binomial->fm = fm = n * r + r;
    binomial->m = m = (int64_t)floor(binomial->fm);
    binomial->p1 = p1 = floor(2.195 * sqrt(n * r * q) - 4.6 * q) + 0.5;
    binomial->xm = xm = m + 0.5;
    binomial->xl = xl = xm - p1;
    binomial->xr = xr = xm + p1;
    binomial->c = c = 0.134 + 20.5 / (15.3 + m);
    a = (fm - xl) / (fm - xl * r);
    binomial->laml = laml = a * (1.0 + a / 2.0);
    a = (xr - fm) / (xr * q);
    binomial->lamr = lamr = a * (1.0 + a / 2.0);
    binomial->p2 = p2 = p1 * (1.0 + 2.0 * c);
    binomial->p3 = p3 = p2 + c / laml;
    binomial->p4 = p4 = p3 + c / lamr;
  } else {
    r = binomial->r;
    q = binomial->q;
    fm = binomial->fm;
    m = binomial->m;
    p1 = binomial->p1;
    xm = binomial->xm;
    xl = binomial->xl;
    xr = binomial->xr;
    c = binomial->c;
    laml = binomial->laml;
    lamr = binomial->lamr;
    p2 = binomial->p2;
    p3 = binomial->p3;
    p4 = binomial->p4;
  }

/* sigh ... */
Step10:
  nrq = n * r * q;
  u = next_double(brng_state) * p4;
  v = next_double(brng_state);
  if (u > p1)
    goto Step20;
  y = (int64_t)floor(xm - p1 * v + u);
  goto Step60;

Step20:
  if (u > p2)
    goto Step30;
  x = xl + (u - p1) / c;
  v = v * c + 1.0 - fabs(m - x + 0.5) / p1;
  if (v > 1.0)
    goto Step10;
  y = (int64_t)floor(x);
  goto Step50;

Step30:
  if (u > p3)
    goto Step40;
  y = (int64_t)floor(xl + log(v) / laml);
  if (y < 0)
    goto Step10;
  v = v * (u - p2) * laml;
  goto Step50;

Step40:
  y = (int64_t)floor(xr - log(v) / lamr);
  if (y > n)
    goto Step10;
  v = v * (u - p3) * lamr;

Step50:
  k = llabs(y - m);
  if ((k > 20) && (k < ((nrq) / 2.0 - 1)))
    goto Step52;

  s = r / q;
  a = s * (n + 1);
  F = 1.0;
  if (m < y) {
    for (i = m + 1; i <= y; i++) {
      F *= (a / i - s);
    }
  } else if (m > y) {
    for (i = y + 1; i <= m; i++) {
      F /= (a / i - s);
    }
  }
  if (v > F)
    goto Step10;
  goto Step60;

Step52:
  rho =
      (k / (nrq)) * ((k * (k / 3.0 + 0.625) + 0.16666666666666666) / nrq + 0.5);
  t = -k * k / (2 * nrq);
  A = log(v);
  if (A < (t - rho))
    goto Step60;
  if (A > (t + rho))
    goto Step10;

  x1 = y + 1;
  f1 = m + 1;
  z = n + 1 - m;
  w = n - y + 1;
  x2 = x1 * x1;
  f2 = f1 * f1;
  z2 = z * z;
  w2 = w * w;
  if (A > (xm * log(f1 / x1) + (n - m + 0.5) * log(z / w) +
           (y - m) * log(w * r / (x1 * q)) +
           (13680. - (462. - (132. - (99. - 140. / f2) / f2) / f2) / f2) / f1 /
               166320. +
           (13680. - (462. - (132. - (99. - 140. / z2) / z2) / z2) / z2) / z /
               166320. +
           (13680. - (462. - (132. - (99. - 140. / x2) / x2) / x2) / x2) / x1 /
               166320. +
           (13680. - (462. - (132. - (99. - 140. / w2) / w2) / w2) / w2) / w /
               166320.)) {
    goto Step10;
  }

Step60:
  if (p > 0.5) {
    y = n - y;
  }

  return y;
}

int64_t random_binomial_inversion(brng_t *brng_state, int64_t n, double p,
                                  binomial_t *binomial) {
  double q, qn, np, px, U;
  int64_t X, bound;

  if (!(binomial->has_binomial) || (binomial->nsave != n) ||
      (binomial->psave != p)) {
    binomial->nsave = n;
    binomial->psave = p;
    binomial->has_binomial = 1;
    binomial->q = q = 1.0 - p;
    binomial->r = qn = exp(n * log(q));
    binomial->c = np = n * p;
    binomial->m = bound = (int64_t)min(n, np + 10.0 * sqrt(np * q + 1));
  } else {
    q = binomial->q;
    qn = binomial->r;
    np = binomial->c;
    bound = binomial->m;
  }
  X = 0;
  px = qn;
  U = next_double(brng_state);
  while (U > px) {
    X++;
    if (X > bound) {
      X = 0;
      px = qn;
      U = next_double(brng_state);
    } else {
      U -= px;
      px = ((n - X + 1) * p * px) / (X * q);
    }
  }
  return X;
}

int64_t random_binomial(brng_t *brng_state, double p, int64_t n,
                        binomial_t *binomial) {
  double q;

  if (p <= 0.5) {
    if (p * n <= 30.0) {
      return random_binomial_inversion(brng_state, n, p, binomial);
    } else {
      return random_binomial_btpe(brng_state, n, p, binomial);
    }
  } else {
    q = 1.0 - p;
    if (q * n <= 30.0) {
      return n - random_binomial_inversion(brng_state, n, q, binomial);
    } else {
      return n - random_binomial_btpe(brng_state, n, q, binomial);
    }
  }
}

double random_noncentral_chisquare(brng_t *brng_state, double df, double nonc) {
  if (nonc == 0) {
    return random_chisquare(brng_state, df);
  }
  if (1 < df) {
    const double Chi2 = random_chisquare(brng_state, df - 1);
    const double n = random_gauss_zig(brng_state) + sqrt(nonc);
    return Chi2 + n * n;
  } else {
    const int64_t i = random_poisson(brng_state, nonc / 2.0);
    return random_chisquare(brng_state, df + 2 * i);
  }
}

double random_noncentral_f(brng_t *brng_state, double dfnum, double dfden,
                           double nonc) {
  double t = random_noncentral_chisquare(brng_state, dfnum, nonc) * dfden;
  return t / (random_chisquare(brng_state, dfden) * dfnum);
}

double random_wald(brng_t *brng_state, double mean, double scale) {
  double U, X, Y;
  double mu_2l;

  mu_2l = mean / (2 * scale);
  Y = random_gauss_zig(brng_state);
  Y = mean * Y * Y;
  X = mean + mu_2l * (Y - sqrt(4 * scale * Y + Y * Y));
  U = next_double(brng_state);
  if (U <= mean / (mean + X)) {
    return X;
  } else {
    return mean * mean / X;
  }
}

double random_vonmises(brng_t *brng_state, double mu, double kappa) {
  double s;
  double U, V, W, Y, Z;
  double result, mod;
  int neg;

  if (kappa < 1e-8) {
    return M_PI * (2 * next_double(brng_state) - 1);
  } else {
    /* with double precision rho is zero until 1.4e-8 */
    if (kappa < 1e-5) {
      /*
       * second order taylor expansion around kappa = 0
       * precise until relatively large kappas as second order is 0
       */
      s = (1. / kappa + kappa);
    } else {
      double r = 1 + sqrt(1 + 4 * kappa * kappa);
      double rho = (r - sqrt(2 * r)) / (2 * kappa);
      s = (1 + rho * rho) / (2 * rho);
    }

    while (1) {
      U = next_double(brng_state);
      Z = cos(M_PI * U);
      W = (1 + s * Z) / (s + Z);
      Y = kappa * (s - W);
      V = next_double(brng_state);
      if ((Y * (2 - Y) - V >= 0) || (log(Y / V) + 1 - Y >= 0)) {
        break;
      }
    }

    U = next_double(brng_state);

    result = acos(W);
    if (U < 0.5) {
      result = -result;
    }
    result += mu;
    neg = (result < 0);
    mod = fabs(result);
    mod = (fmod(mod + M_PI, 2 * M_PI) - M_PI);
    if (neg) {
      mod *= -1;
    }

    return mod;
  }
}

int64_t random_logseries(brng_t *brng_state, double p) {
  double q, r, U, V;
  int64_t result;

  r = log(1.0 - p);

  while (1) {
    V = next_double(brng_state);
    if (V >= p) {
      return 1;
    }
    U = next_double(brng_state);
    q = 1.0 - exp(r * U);
    if (V <= q * q) {
      result = (int64_t)floor(1 + log(V) / log(q));
      if (result < 1) {
        continue;
      } else {
        return result;
      }
    }
    if (V >= q) {
      return 1;
    }
    return 2;
  }
}

int64_t random_geometric_search(brng_t *brng_state, double p) {
  double U;
  int64_t X;
  double sum, prod, q;

  X = 1;
  sum = prod = p;
  q = 1.0 - p;
  U = next_double(brng_state);
  while (U > sum) {
    prod *= q;
    sum += prod;
    X++;
  }
  return X;
}

int64_t random_geometric_inversion(brng_t *brng_state, double p) {
  return (int64_t)ceil(log(1.0 - next_double(brng_state)) / log(1.0 - p));
}

int64_t random_geometric(brng_t *brng_state, double p) {
  if (p >= 0.333333333333333333333333) {
    return random_geometric_search(brng_state, p);
  } else {
    return random_geometric_inversion(brng_state, p);
  }
}

int64_t random_zipf(brng_t *brng_state, double a) {
  double T, U, V;
  int64_t X;
  double am1, b;

  am1 = a - 1.0;
  b = pow(2.0, am1);
  do {
    U = 1.0 - next_double(brng_state);
    V = next_double(brng_state);
    X = (int64_t)floor(pow(U, -1.0 / am1));
    /* The real result may be above what can be represented in a int64.
     * It will get casted to -sys.maxint-1. Since this is
     * a straightforward rejection algorithm, we can just reject this value
     * in the rejection condition below. This function then models a Zipf
     * distribution truncated to sys.maxint.
     */
    T = pow(1.0 + 1.0 / X, am1);
  } while (((V * X * (T - 1.0) / (b - 1.0)) > (T / b)) || X < 1);
  return X;
}

double random_triangular(brng_t *brng_state, double left, double mode,
                         double right) {
  double base, leftbase, ratio, leftprod, rightprod;
  double U;

  base = right - left;
  leftbase = mode - left;
  ratio = leftbase / base;
  leftprod = leftbase * base;
  rightprod = (right - mode) * base;

  U = next_double(brng_state);
  if (U <= ratio) {
    return left + sqrt(U * leftprod);
  } else {
    return right - sqrt((1.0 - U) * rightprod);
  }
}

int64_t random_hypergeometric_hyp(brng_t *brng_state, int64_t good, int64_t bad,
                                  int64_t sample) {
  int64_t d1, k, z;
  double d2, u, y;

  d1 = bad + good - sample;
  d2 = (double)min(bad, good);

  y = d2;
  k = sample;
  while (y > 0.0) {
    u = next_double(brng_state);
    y -= (int64_t)floor(u + y / (d1 + k));
    k--;
    if (k == 0)
      break;
  }
  z = (int64_t)(d2 - y);
  if (good > bad)
    z = sample - z;
  return z;
}

/* D1 = 2*sqrt(2/e) */
/* D2 = 3 - 2*sqrt(3/e) */
#define D1 1.7155277699214135
#define D2 0.8989161620588988
int64_t random_hypergeometric_hrua(brng_t *brng_state, int64_t good,
                                   int64_t bad, int64_t sample) {
  int64_t mingoodbad, maxgoodbad, popsize, m, d9;
  double d4, d5, d6, d7, d8, d10, d11;
  int64_t Z;
  double T, W, X, Y;

  mingoodbad = min(good, bad);
  popsize = good + bad;
  maxgoodbad = max(good, bad);
  m = min(sample, popsize - sample);
  d4 = ((double)mingoodbad) / popsize;
  d5 = 1.0 - d4;
  d6 = m * d4 + 0.5;
  d7 = sqrt((double)(popsize - m) * sample * d4 * d5 / (popsize - 1) + 0.5);
  d8 = D1 * d7 + D2;
  d9 = (int64_t)floor((double)(m + 1) * (mingoodbad + 1) / (popsize + 2));
  d10 = (loggam(d9 + 1) + loggam(mingoodbad - d9 + 1) + loggam(m - d9 + 1) +
         loggam(maxgoodbad - m + d9 + 1));
  d11 = min(min(m, mingoodbad) + 1.0, floor(d6 + 16 * d7));
  /* 16 for 16-decimal-digit precision in D1 and D2 */

  while (1) {
    X = next_double(brng_state);
    Y = next_double(brng_state);
    W = d6 + d8 * (Y - 0.5) / X;

    /* fast rejection: */
    if ((W < 0.0) || (W >= d11))
      continue;

    Z = (int64_t)floor(W);
    T = d10 - (loggam(Z + 1) + loggam(mingoodbad - Z + 1) + loggam(m - Z + 1) +
               loggam(maxgoodbad - m + Z + 1));

    /* fast acceptance: */
    if ((X * (4.0 - X) - 3.0) <= T)
      break;

    /* fast rejection: */
    if (X * (X - T) >= 1)
      continue;

    if (2.0 * log(X) <= T)
      break; /* acceptance */
  }

  /* this is a correction to HRUA* by Ivan Frohne in rv.py */
  if (good > bad)
    Z = m - Z;

  /* another fix from rv.py to allow sample to exceed popsize/2 */
  if (m < sample)
    Z = good - Z;

  return Z;
}
#undef D1
#undef D2

int64_t random_hypergeometric(brng_t *brng_state, int64_t good, int64_t bad,
                              int64_t sample) {
  if (sample > 10) {
    return random_hypergeometric_hrua(brng_state, good, bad, sample);
  } else {
    return random_hypergeometric_hyp(brng_state, good, bad, sample);
  }
}

uint64_t random_interval(brng_t *brng_state, uint64_t max) {
  uint64_t mask, value;
  if (max == 0) {
    return 0;
  }

  mask = max;

  /* Smallest bit mask >= max */
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;
  mask |= mask >> 8;
  mask |= mask >> 16;
  mask |= mask >> 32;

  /* Search a random value in [0..mask] <= max */
  if (max <= 0xffffffffUL) {
    while ((value = (next_uint32(brng_state) & mask)) > max)
      ;
  } else {
    while ((value = (next_uint64(brng_state) & mask)) > max)
      ;
  }
  return value;
}

static NPY_INLINE uint64_t gen_mask(uint64_t max) {
  uint64_t mask = max;
  mask |= mask >> 1;
  mask |= mask >> 2;
  mask |= mask >> 4;
  mask |= mask >> 8;
  mask |= mask >> 16;
  mask |= mask >> 32;
  return mask;
}

/*
 * Fills an array with cnt random npy_uint64 between off and off + rng
 * inclusive. The numbers wrap if rng is sufficiently large.
 */

static NPY_INLINE uint64_t bounded_uint64(brng_t *brng_state, uint64_t off,
                                          uint64_t rng, uint64_t mask) {
  uint64_t val;
  if (rng == 0)
    return off;

  if (rng <= 0xffffffffUL) {
    while ((val = (next_uint32(brng_state) & mask)) > rng)
      ;
  } else {
    while ((val = (next_uint64(brng_state) & mask)) > rng)
      ;
  }
  return off + val;
}

uint64_t random_bounded_uint64(brng_t *brng_state, uint64_t off, uint64_t rng,
                               uint64_t mask) {
  return bounded_uint64(brng_state, off, rng, mask);
}

static NPY_INLINE uint32_t bounded_uint32(brng_t *brng_state, uint32_t off,
                                          uint32_t rng, uint32_t mask) {
  /*
   * The buffer and buffer count are not used here but are included to allow
   * this function to be templated with the similar uint8 and uint16
   * functions
   */

  uint32_t val;
  if (rng == 0)
    return off;

  while ((val = (next_uint32(brng_state) & mask)) > rng)
    ;
  return off + val;
}

uint32_t random_buffered_bounded_uint32(brng_t *brng_state, uint32_t off,
                                        uint32_t rng, uint32_t mask, int *bcnt,
                                        uint32_t *buf) {
  /*
   *  Unused bcnt and buf are here only to allow templating with other uint
   * generators
   */
  return bounded_uint32(brng_state, off, rng, mask);
}

static NPY_INLINE uint16_t buffered_bounded_uint16(brng_t *brng_state,
                                                   uint16_t off, uint16_t rng,
                                                   uint16_t mask, int *bcnt,
                                                   uint32_t *buf) {
  uint16_t val;
  if (rng == 0)
    return off;

  do {
    if (!(bcnt[0])) {
      buf[0] = next_uint32(brng_state);
      bcnt[0] = 1;
    } else {
      buf[0] >>= 16;
      bcnt[0] -= 1;
    }
    val = (uint16_t)buf[0] & mask;
  } while (val > rng);
  return off + val;
}

uint16_t random_buffered_bounded_uint16(brng_t *brng_state, uint16_t off,
                                        uint16_t rng, uint16_t mask, int *bcnt,
                                        uint32_t *buf) {
  return buffered_bounded_uint16(brng_state, off, rng, mask, bcnt, buf);
}

static NPY_INLINE uint8_t buffered_bounded_uint8(brng_t *brng_state,
                                                 uint8_t off, uint8_t rng,
                                                 uint8_t mask, int *bcnt,
                                                 uint32_t *buf) {
  uint8_t val;
  if (rng == 0)
    return off;
  do {
    if (!(bcnt[0])) {
      buf[0] = next_uint32(brng_state);
      bcnt[0] = 3;
    } else {
      buf[0] >>= 8;
      bcnt[0] -= 1;
    }
    val = (uint8_t)buf[0] & mask;
  } while (val > rng);
  return off + val;
}

uint8_t random_buffered_bounded_uint8(brng_t *brng_state, uint8_t off,
                                      uint8_t rng, uint8_t mask, int *bcnt,
                                      uint32_t *buf) {
  return buffered_bounded_uint8(brng_state, off, rng, mask, bcnt, buf);
}

static NPY_INLINE npy_bool buffered_bounded_bool(brng_t *brng_state,
                                                 npy_bool off, npy_bool rng,
                                                 npy_bool mask, int *bcnt,
                                                 uint32_t *buf) {
  if (rng == 0)
    return off;
  if (!(bcnt[0])) {
    buf[0] = next_uint32(brng_state);
    bcnt[0] = 31;
  } else {
    buf[0] >>= 1;
    bcnt[0] -= 1;
  }
  return (buf[0] & 0x00000001UL) != 0;
}

npy_bool random_buffered_bounded_bool(brng_t *brng_state, npy_bool off,
                                      npy_bool rng, npy_bool mask, int *bcnt,
                                      uint32_t *buf) {
  return buffered_bounded_bool(brng_state, off, rng, mask, bcnt, buf);
}

void random_bounded_uint64_fill(brng_t *brng_state, uint64_t off, uint64_t rng,
                                npy_intp cnt, uint64_t *out) {
  uint64_t mask;
  npy_intp i;

  /* Smallest bit mask >= max */
  mask = gen_mask(rng);
  for (i = 0; i < cnt; i++) {
    out[i] = bounded_uint64(brng_state, off, rng, mask);
  }
}

/*
 * Fills an array with cnt random npy_uint32 between off and off + rng
 * inclusive. The numbers wrap if rng is sufficiently large.
 */
void random_bounded_uint32_fill(brng_t *brng_state, uint32_t off, uint32_t rng,
                                npy_intp cnt, uint32_t *out) {
  uint32_t mask;
  npy_intp i;

  /* Smallest bit mask >= max */
  mask = (uint32_t)gen_mask(rng);
  for (i = 0; i < cnt; i++) {
    out[i] = bounded_uint32(brng_state, off, rng, mask);
  }
}

/*
 * Fills an array with cnt random npy_uint16 between off and off + rng
 * inclusive. The numbers wrap if rng is sufficiently large.
 */
void random_bounded_uint16_fill(brng_t *brng_state, uint16_t off, uint16_t rng,
                                npy_intp cnt, uint16_t *out) {
  uint16_t mask;
  npy_intp i;
  uint32_t buf = 0;
  int bcnt = 0;

  /* Smallest bit mask >= max */
  mask = (uint16_t)gen_mask(rng);
  for (i = 0; i < cnt; i++) {
    out[i] = buffered_bounded_uint16(brng_state, off, rng, mask, &bcnt, &buf);
  }
}

/*
 * Fills an array with cnt random npy_uint8 between off and off + rng
 * inclusive. The numbers wrap if rng is sufficiently large.
 */
void random_bounded_uint8_fill(brng_t *brng_state, uint8_t off, uint8_t rng,
                               npy_intp cnt, uint8_t *out) {
  uint8_t mask;
  npy_intp i;
  uint32_t buf = 0;
  int bcnt = 0;

  /* Smallest bit mask >= max */
  mask = (uint8_t)gen_mask(rng);
  for (i = 0; i < cnt; i++) {
    out[i] = buffered_bounded_uint8(brng_state, off, rng, mask, &bcnt, &buf);
  }
}

/*
 * Fills an array with cnt random npy_bool between off and off + rng
 * inclusive.
 */
void random_bounded_bool_fill(brng_t *brng_state, npy_bool off, npy_bool rng,
                              npy_intp cnt, npy_bool *out) {
  npy_bool mask = 0;
  npy_intp i;
  uint32_t buf = 0;
  int bcnt = 0;

  for (i = 0; i < cnt; i++) {
    out[i] = buffered_bounded_bool(brng_state, off, rng, mask, &bcnt, &buf);
  }
}
