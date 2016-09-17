#include "randomkit.h"
#include <stdlib.h>


/*
 *  mvhg_count
 *
 *  Draw variates from the multivariate hypergeometric distribution--
 *  the "count" algorithm.
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
 *  The "count" algorithm for drawing one variate is roughly equivalent to the
 *  following numpy code:
 *
 *      choices = np.repeat(np.arange(len(colors)), colors)
 *      selection = np.random.choice(choices, nsample, replace=False)
 *      sample = np.bincount(selection, minlength=len(colors))
 *
 *  This function uses a temporary array with length sum(colors).
 *
 *  On input, it is assumed that:
 *    * num_colors > 0
 *    * colors[i] >= 0
 *    * nsample <= sum(colors)
 *
 */

int mvhg_count(rk_state *state,
               long total,
               npy_intp num_colors, long *colors,
               long nsample,
               npy_intp num_sample, long *sample)
{
    npy_intp i, j, k;
    npy_intp *choices;
    int more_than_half;

    choices = malloc(total * (sizeof *choices));
    if (choices == NULL) {
        return -1;
    }

    /*
     *  If colors contains, for example, [3 2 5], then choices
     *  will contain [0 0 0 1 1 2 2 2 2 2].
     */
    k = 0;
    for (i = 0; i < num_colors; ++i) {
        for (j = 0; j < colors[i]; ++j) {
            choices[k] = i;
            ++k;
        }
    }

    more_than_half = nsample > (total / 2);
    if (more_than_half) {
        nsample = total - nsample;
    }

    for (i = 0; i < num_sample; i += num_colors) {
        /*
         *  Fisher-Yates shuffle, but only loop through the first
         *  `nsample` entries of `choices`.  After the loop,
         *  choices[:nsample] contains a random sample from the
         *  the full array.
         */
        for (j = 0; j < nsample; ++j) {
            npy_intp tmp;
            k = j + rk_interval(total - j - 1, state);
            tmp = choices[k];
            choices[k] = choices[j];
            choices[j] = tmp;
        }
        /*
         *  Count the number of occurrences of each value in choices[:nsample].
         *  The result, stored in sample[i:i+num_colors], is the sample from
         *  the multivariate hypergeometric distribution.
         */
        for (j = 0; j < nsample; ++j) {
            sample[i + choices[j]] += 1;
        }

        if (more_than_half) {
            for (j = 0; j < num_colors; ++j) {
                sample[i + j] = colors[j] - sample[i + j];
            }
        }
    }

    free(choices);

    return 0;
}
