
#include "randomkit.h"
#include "logfactorial.h"
#include <math.h>

#ifndef min
#define min(x,y) ((x<y)?x:y)
#define max(x,y) ((x>y)?x:y)
#endif


/*
 *  This is an alternative to rk_hypergeometric_hyp(), implemented
 *  without using floating point.
 *
 *  It is assumed that when this function is called:
 *    * good, bad and sample are nonnegative;
 *    * the sum good+bad will not result in overflow; 
 *    * sample <= good+bad.
 */

static long hypergeometric_sample(rk_state *state,
                                  long good, long bad, long sample)
{
    long remaining_total, remaining_good, result, computed_sample;
    long total = good + bad;

    if (sample > total/2) {
        computed_sample = total - sample;
    }
    else {
        computed_sample = sample;
    }

    remaining_total = total;
    remaining_good = good;

    while ((computed_sample > 0) && (remaining_good > 0) &&
           (remaining_total > remaining_good)) {
        /*
         *  rk_interval(max, state) returns an integer in [0, max] *inclusive*,
         *  so we decrement remaining_total before passing it to rk_interval().
         */
        --remaining_total;
        if ((long) rk_interval(remaining_total, state) < remaining_good) {
            /* Selected a "good" one, so decrement remaining_good. */
            --remaining_good;
        }
        --computed_sample;
    }

    if (remaining_total == remaining_good) {
        /* Only "good" choices are left. */
        remaining_good -= computed_sample;
    }

    if (sample > total/2) {
        result = remaining_good;
    }
    else {
        result = good - remaining_good;
    }

    return result;
}


/* D1 = 2*sqrt(2/e)      */
/* D2 = 3 - 2*sqrt(3/e)  */
#define D1 1.7155277699214135
#define D2 0.8989161620588988

/*
 *  Generate variates from the hypergeometric distribution
 *  using the ratio-of-uniforms method.
 *
 *  This is an alternative to rk_hypergeometric_hrua.
 *  It fixes a mistake in that code, and uses logfactorial()
 *  instead of loggam().
 *
 *  Variables have been renamed to match the original source.
 *  In the code, the variable names a, b, c, g, h, m, p, q, K, T,
 *  U and X match the names used in "Algorithm HRUA" beginning on
 *  page 82 of Stadlober's 1989 thesis.
 *
 *  It is assumed that when this function is called:
 *    * good, bad and sample are nonnegative;
 *    * the sum good+bad will not result in overflow; 
 *    * sample <= good+bad.
 *
 * References:
 * -  Ernst Stadlober's thesis "Sampling from Poisson, Binomial and
 *    Hypergeometric Distributions: Ratio of Uniforms as a Simple and
 *    Fast Alternative" (1989)
 * -  Ernst Stadlober, "The ratio of uniforms approach for generating
 *    discrete random variates", Journal of Computational and Applied
 *    Mathematics, 31, pp. 181-189 (1990).
 */

static long hypergeometric_hrua(rk_state *state,
                                long good, long bad, long sample)
{
    long mingoodbad, maxgoodbad, popsize;
    long computed_sample;
    double p, q;
    double mu, var;
    double a, c, b, h, g;
    long m, K;

    popsize = good + bad;
    computed_sample = min(sample, popsize - sample);
    mingoodbad = min(good, bad);
    maxgoodbad = max(good, bad);

    /*
     *  Variables that do not match Stadlober (1989)
     *    Here               Stadlober
     *    ----------------   ---------
     *    mingoodbad            M
     *    popsize               N
     *    computed_sample       n
     */

    p = ((double) mingoodbad) / popsize;
    q = ((double) maxgoodbad) / popsize;

    /* mu is the mean of the distribution. */
    mu = computed_sample * p;

    a = mu + 0.5;

    /* var is the variance of the distribution. */
    var = ((double)(popsize - computed_sample) *
           computed_sample * p * q / (popsize - 1));

    c = sqrt(var + 0.5);

    /*
     *  h is 2*s_hat (See Stadlober's theses (1989), Eq. (5.17); or
     *  Stadlober (1990), Eq. 8).  s_hat is the scale of the "table mountain"
     *  function that dominates the scaled hypergeometric PMF ("scaled" means
     *  normalized to have a maximum value of 1).
     */
    h = D1*c + D2;

    m = (long)floor((double)(computed_sample + 1) * (mingoodbad + 1) /
                            (popsize + 2));

    g = (logfactorial(m) +
         logfactorial(mingoodbad - m) +
         logfactorial(computed_sample - m) +
         logfactorial(maxgoodbad - computed_sample + m));

    /*
     *  b is the upper bound for random samples:
     *  ... min(computed_sample, mingoodbad) + 1 is the length of the support.
     *  ... floor(a + 16*c) is 16 standard deviations beyond the mean.
     *
     *  The idea behind the second upper bound is that values that far out in
     *  the tail have negligible probabilities.
     *
     *  There is a comment in a previous version of this algorithm that says
     *      "16 for 16-decimal-digit precision in D1 and D2",
     *  but there is no documented justification for this value.  A lower value
     *  might work just as well, but I've kept the value 16 here.
     */
    b = min(min(computed_sample, mingoodbad) + 1.0, floor(a + 16*c));

    while (1) {
        double U, V, X, T;
        double gp;
        U = rk_double(state);
        V = rk_double(state);  /*  "U star" in Stadlober (1989) */
        X = a + h*(V - 0.5) / U;

        /* fast rejection: */
        if ((X < 0.0) || (X >= b)) {
            continue;
        }

        K = (long)floor(X);

        gp = (logfactorial(K) +
              logfactorial(mingoodbad - K) +
              logfactorial(computed_sample - K) +
              logfactorial(maxgoodbad - computed_sample + K));

        T = g - gp;

        /* fast acceptance: */
        if ((U*(4.0 - U) - 3.0) <= T) {
            break;
        }

        /* fast rejection: */
        if (U*(U - T) >= 1) {
            continue;
        }

        if (2.0*log(U) <= T) {
            /* acceptance */
            break;  
        }
    }

    if (good > bad) {
        K = computed_sample - K;
    }

    if (computed_sample < sample) {
        K = good - K;
    }

    return K;
}


/*
 *  Draw a sample from the hypergeometric distribution.
 *
 *  It is assumed that when this function is called:
 *    * good, bad and sample are nonnegative;
 *    * the sum good+bad will not result in overflow; 
 *    * sample <= good+bad.
 */

static long hypergeometric(rk_state *state,
                           long good, long bad, long sample)
{
    long r;

    if ((sample >= 10) && (sample <= good + bad - 10)) {
        /* This will use the ratio-of-uniforms method. */
        r = hypergeometric_hrua(state, good, bad, sample);
    }
    else {
        /* The simpler implementation is faster for small samples. */
        r = hypergeometric_sample(state, good, bad, sample);
    }
    return r;
}


/*
 *  mvhg_marginals
 *
 *  Draw samples from the multivariate hypergeometric distribution--
 *  the "marginals" algorithm.
 *
 *  This version generates the sample by iteratively calling
 *  hypergeometric().
 *
 *  Parameters
 *  ----------
 *  rk_state *state
 *      Pointer to a randomkit `rk_state` instance.
 *  long total
 *      The sum of the values in the array `colors`.  (This is redundant
 *      information, but we know the caller has already computed it, so
 *      we might as well use it.)
 *  size_t num_colors
 *      The length of the `colors` array.
 *  long *colors
 *      The array of colors (i.e. the number of each type in the collection
 *      from which the random variate is drawn).
 *  long nsample
 *      The number of objects drawn without replacement for each variate.
 *      `nsample` must not exceed sum(colors).  This condition is not checked;
 *      it is assumed that the caller has already validated the value.
 *  size_t num_sample
 *      The number of random variates to generate.  (This is not the size of
 *      the array `sample`, because each variate requires `num_colors`
 *      values.)
 *  long *sample
 *      The array that will hold the result.  The length of this array must
 *      be the product of `num_colors` and `num_sample`.
 *      The array is not initialized; it is expected that array has been
 *      initialized with zeros when the function is called.
 *
 *  Notes
 *  -----
 *  Here's an example that demonstrates the idea of this algorithm.
 *
 *  Suppose the urn contains red, green, blue and yellow marbles.
 *  Let nred be the number of red marbles, and define the quantities for
 *  the other colors similarly.  The total number of marbles is
 *
 *      total = nred + ngreen + nblue + nyellow.
 *
 *  To generate a sample using rk_hypergeometric:
 *
 *     red_sample = rk_hypergeometric(ngood=nred, nbad=total - nred,
 *                                    nsample=nsample)
 *
 *  This gives us the number of red marbles in the sample.  The number of
 *  marbles in the sample that are *not* red is nsample - red_sample.
 *  To figure out the distribution of those marbles, we again use
 *  rk_hypergeometric:
 *
 *      green_sample = rk_hypergeometric(ngood=ngreen,
 *                                       nbad=total - nred - ngreen,
 *                                       nsample=nsample - red_sample)
 *
 *  Similarly,
 *
 *      blue_sample = rk_hypergeometric(
 *                        ngood=nblue,
 *                        nbad=total - nred - ngreen - nblue,
 *                        nsample=nsample - red_sample - green_sample)
 *
 *  Finally,
 *
 *      yellow_sample = total - (red_sample + green_sample + blue_sample).
 *
 *  The above sequence of steps is implemented as a loop for an arbitrary
 *  number of colors in the innermost loop in the code below.  `remaining`
 *  is the value passed to `nbad`; it is `total - colors[0]` in the first
 *  call to rk_hypergeometric(), and then decreases by `colors[j]` in
 *  each iteration.  `num_to_sample` is the `nsample` argument.  It
 *  starts at this function's `nsample` input, and is decreased by the
 *  result of the call to rk_hypergeometric() in each iteration.
 */

int mvhg_marginals(rk_state *state,
                   long total,
                   npy_intp num_colors, long *colors,
                   long nsample,
                   npy_intp num_sample, long *sample)
{
    npy_intp i;
    int more_than_half;

    more_than_half = nsample > (total / 2);
    if (more_than_half) {
        nsample = total - nsample;
    }

    for (i = 0; i < num_sample; i += num_colors) {
        npy_intp num_to_sample, remaining, j;
        num_to_sample = nsample;
        remaining = total;
        for (j = 0; j < num_colors - 1; ++j) {
            long r;
            if (num_to_sample <= 0) {
                break;
            }
            remaining -= colors[j];
            r = hypergeometric(state, colors[j], remaining, num_to_sample);
            sample[i + j] = r;
            num_to_sample -= r;
        }

        if (j == num_colors - 1) {
            sample[i + j] = num_to_sample;
        }

        if (more_than_half) {
            for (j = 0; j < num_colors; ++j) {
                sample[i + j] = colors[j] - sample[i + j];
            }
        }
    }
    return 0;
}
