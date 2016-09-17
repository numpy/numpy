
#ifndef MVHG_MARGINALS_H
#define MVHG_MARGINALS_H

#include "randomkit.h"

int mvhg_marginals(rk_state *state,
                   long total,
                   npy_intp num_colors, long *colors,
                   long nsample,
                   npy_intp num_sample, long *sample);

#endif
