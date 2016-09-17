
#include <numpy/npy_common.h>


/*
 *  Sum the values in the array `colors`.
 *  Return -1 if an overflow occurs.
 *
 *  The values in *colors are assumed to be nonnegative.
 */

long safe_sum_nonneg_long(npy_intp num_colors, long *colors)
{
    npy_intp i;
    long sum;

    sum = 0;
    for (i = 0; i < num_colors; ++i) {
        if (colors[i] > NPY_MAX_LONG - sum) {
            return -1;
        }
        sum += colors[i];
    }
    return sum;
}
