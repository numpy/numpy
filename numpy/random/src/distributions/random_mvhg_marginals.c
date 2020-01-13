#include "numpy/random/distributions.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <math.h>

#include "logfactorial.h"


/*
 *  random_multivariate_hypergeometric_marginals
 *
 *  Draw samples from the multivariate hypergeometric distribution--
 *  the "marginals" algorithm.
 *
 *  This version generates the sample by iteratively calling
 *  hypergeometric() (the univariate hypergeometric distribution).
 *
 *  Parameters
 *  ----------
 *  bitgen_t *bitgen_state
 *      Pointer to a `bitgen_t` instance.
 *  int64_t total
 *      The sum of the values in the array `colors`.  (This is redundant
 *      information, but we know the caller has already computed it, so
 *      we might as well use it.)
 *  size_t num_colors
 *      The length of the `colors` array.  The functions assumes
 *      num_colors > 0.
 *  int64_t *colors
 *      The array of colors (i.e. the number of each type in the collection
 *      from which the random variate is drawn).
 *  int64_t nsample
 *      The number of objects drawn without replacement for each variate.
 *      `nsample` must not exceed sum(colors).  This condition is not checked;
 *      it is assumed that the caller has already validated the value.
 *  size_t num_variates
 *      The number of variates to be produced and put in the array
 *      pointed to by `variates`.  One variate is a vector of length
 *      `num_colors`, so the array pointed to by `variates` must have length
 *      `num_variates * num_colors`.
 *  int64_t *variates
 *      The array that will hold the result.  It must have length
 *      `num_variates * num_colors`.
 *      The array is not initialized in the function; it is expected that the
 *      array has been initialized with zeros when the function is called.
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
 *     red_sample = hypergeometric(ngood=nred, nbad=total - nred,
 *                                 nsample=nsample)
 *
 *  This gives us the number of red marbles in the sample.  The number of
 *  marbles in the sample that are *not* red is nsample - red_sample.
 *  To figure out the distribution of those marbles, we again use
 *  rk_hypergeometric:
 *
 *      green_sample = hypergeometric(ngood=ngreen,
 *                                    nbad=total - nred - ngreen,
 *                                    nsample=nsample - red_sample)
 *
 *  Similarly,
 *
 *      blue_sample = hypergeometric(
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
 *  call to random_hypergeometric(), and then decreases by `colors[j]` in
 *  each iteration.  `num_to_sample` is the `nsample` argument.  It
 *  starts at this function's `nsample` input, and is decreased by the
 *  result of the call to random_hypergeometric() in each iteration.
 *
 *  Assumptions on the arguments (not checked in the function):
 *    *  colors[k] >= 0  for k in range(num_colors)
 *    *  total = sum(colors)
 *    *  0 <= nsample <= total
 *    *  the product num_variates * num_colors does not overflow
 */

void random_multivariate_hypergeometric_marginals(bitgen_t *bitgen_state,
                           int64_t total,
                           size_t num_colors, int64_t *colors,
                           int64_t nsample,
                           size_t num_variates, int64_t *variates)
{
    bool more_than_half;

    if ((total == 0) || (nsample == 0) || (num_variates == 0)) {
        // Nothing to do.
        return;
    }

    more_than_half = nsample > (total / 2);
    if (more_than_half) {
        nsample = total - nsample;
    }

    for (size_t i = 0; i < num_variates * num_colors; i += num_colors) {
        int64_t num_to_sample = nsample;
        int64_t remaining = total;
        for (size_t j = 0; (num_to_sample > 0) && (j + 1 < num_colors); ++j) {
            int64_t r;
            remaining -= colors[j];
            r = random_hypergeometric(bitgen_state,
                                      colors[j], remaining, num_to_sample);
            variates[i + j] = r;
            num_to_sample -= r;
        }

        if (num_to_sample > 0) {
            variates[i + num_colors - 1] = num_to_sample;
        }

        if (more_than_half) {
            for (size_t k = 0; k < num_colors; ++k) {
                variates[i + k] = colors[k] - variates[i + k];
            }
        }
    }
}
