#pragma once
#ifndef DSFMT_JUMP_H
#define DSFMT_JUMP_H
/**
 * @file SFMT-jump.h
 *
 * @brief jump header file.
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (The University of Tokyo)
 *
 * Copyright (C) 2012 Mutsuo Saito, Makoto Matsumoto,
 * Hiroshima University and The University of Tokyo.
 * All rights reserved.
 *
 * The 3-clause BSD License is applied to this software, see
 * LICENSE.txt
 */
#if defined(__cplusplus)
extern "C" {
#endif

#include "dSFMT.h"
void dSFMT_jump(dsfmt_t * dsfmt, const char * jump_str);

#if defined(__cplusplus)
}
#endif
#endif
