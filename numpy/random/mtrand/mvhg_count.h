
#ifndef MVHG_COUNT_H
#define MVHG_COUNT_H

#include "randomkit.h"

int mvhg_count(rk_state *state,
               long total,
               npy_intp num_colors, long *colors,
               long nsample,
               npy_intp num_sample, long *sample);

#endif
