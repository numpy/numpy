/*
 * This file is part of pocketfft.
 * Licensed under a 3-clause BSD style license - see LICENSE.md
 */

/*! \file pocketfft.h
 *  Public interface of the pocketfft library
 *
 *  Copyright (C) 2008-2018 Max-Planck-Society
 *  \author Martin Reinecke
 */

#ifndef POCKETFFT_H
#define POCKETFFT_H

#include <stdlib.h>
#include "numpy/numpyconfig.h"  // for NPY_VISIBILITY_HIDDEN

struct cfft_plan_i;
typedef struct cfft_plan_i * cfft_plan;
NPY_VISIBILITY_HIDDEN cfft_plan make_cfft_plan (size_t length);
NPY_VISIBILITY_HIDDEN void destroy_cfft_plan (cfft_plan plan);
NPY_VISIBILITY_HIDDEN int cfft_backward(cfft_plan plan, double c[], double fct);
NPY_VISIBILITY_HIDDEN int cfft_forward(cfft_plan plan, double c[], double fct);
NPY_VISIBILITY_HIDDEN size_t cfft_length(cfft_plan plan);

struct rfft_plan_i;
typedef struct rfft_plan_i * rfft_plan;
NPY_VISIBILITY_HIDDEN rfft_plan make_rfft_plan (size_t length);
NPY_VISIBILITY_HIDDEN void destroy_rfft_plan (rfft_plan plan);
NPY_VISIBILITY_HIDDEN int rfft_backward(rfft_plan plan, double c[], double fct);
NPY_VISIBILITY_HIDDEN int rfft_forward(rfft_plan plan, double c[], double fct);
NPY_VISIBILITY_HIDDEN size_t rfft_length(rfft_plan plan);

#endif
