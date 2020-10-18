#include "numpy/random/distributions.h"
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>


/*
 *  random_multivariate_hypergeometric_count
 *
 *  Draw variates from the multivariate hypergeometric distribution--
 *  the "count" algorithm.
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
 *      The length of the `colors` array.
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
 *  The "count" algorithm for drawing one variate is roughly equivalent to the
 *  following numpy code:
 *
 *      choices = np.repeat(np.arange(len(colors)), colors)
 *      selection = np.random.choice(choices, nsample, replace=False)
 *      variate = np.bincount(selection, minlength=len(colors))
 *
 *  This function uses a temporary array with length sum(colors).
 *
 *  Assumptions on the arguments (not checked in the function):
 *    *  colors[k] >= 0  for k in range(num_colors)
 *    *  total = sum(colors)
 *    *  0 <= nsample <= total
 *    *  the product total * sizeof(size_t) does not exceed SIZE_MAX
 *    *  the product num_variates * num_colors does not overflow
 */

int random_multivariate_hypergeometric_count(bitgen_t *bitgen_state,
                      int64_t total,
                      size_t num_colors, int64_t *colors,
                      int64_t nsample,
                      size_t num_variates, int64_t *variates)
{
    size_t *choices;
    bool more_than_half;

    if ((total == 0) || (nsample == 0) || (num_variates == 0)) {
        // Nothing to do.
        return 0;
    }

    choices = malloc(total * (sizeof *choices));
    if (choices == NULL) {
        return -1;
    }

    /*
     *  If colors contains, for example, [3 2 5], then choices
     *  will contain [0 0 0 1 1 2 2 2 2 2].
     */
    for (size_t i = 0, k = 0; i < num_colors; ++i) {
        for (int64_t j = 0; j < colors[i]; ++j) {
            choices[k] = i;
            ++k;
        }
    }

    more_than_half = nsample > (total / 2);
    if (more_than_half) {
        nsample = total - nsample;
    }

    for (size_t i = 0; i < num_variates * num_colors; i += num_colors) {
        /*
         *  Fisher-Yates shuffle, but only loop through the first
         *  `nsample` entries of `choices`.  After the loop,
         *  choices[:nsample] contains a random sample from the
         *  the full array.
         */
        for (size_t j = 0; j < (size_t) nsample; ++j) {
            size_t tmp, k;
            // Note: nsample is not greater than total, so there is no danger
            // of integer underflow in `(size_t) total - j - 1`.
            k = j + (size_t) random_interval(bitgen_state,
                                             (size_t) total - j - 1);
            tmp = choices[k];
            choices[k] = choices[j];
            choices[j] = tmp;
        }
        /*
         *  Count the number of occurrences of each value in choices[:nsample].
         *  The result, stored in sample[i:i+num_colors], is the sample from
         *  the multivariate hypergeometric distribution.
         */
        for (size_t j = 0; j < (size_t) nsample; ++j) {
            variates[i + choices[j]] += 1;
        }

        if (more_than_half) {
            for (size_t k = 0; k < num_colors; ++k) {
                variates[i + k] = colors[k] - variates[i + k];
            }
        }
    }

    free(choices);

    return 0;
}
