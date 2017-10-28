/*
 * Copyright (c) 2014 Ryan Juckett
 * http://www.ryanjuckett.com/
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 *
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 *
 * 3. This notice may not be removed or altered from any source
 *    distribution.
 */

/*
 * This file contains a modified version of Ryan Juckett's Dragon4
 * implementation, which has been ported from C++ to C and which has
 * modifications specific to printing floats in numpy.
 */

#ifndef _NPY_DRAGON4_H_
#define _NPY_DRAGON4_H_
#include "numpy/npy_common.h"

typedef enum TrimMode
{
    TrimMode_None,         /* don't trim zeros, always leave a decimal point */
    TrimMode_LeaveOneZero, /* trim all but the zero before the decimal point */
    TrimMode_Zeros,        /* trim all trailing zeros, leave decimal point */
    TrimMode_DptZeros,     /* trim trailing zeros & trailing decimal point */
} TrimMode;


//******************************************************************************
// These functions frint a floating-point number as a decimal string.
// The output string is always NUL terminated and the string length (not
// including the NUL) is returned.
//******************************************************************************
//
// Arguments are:
//   * buffer - buffer to output into
//   * bufferSize - maximum characters that can be printed to buffer
//   * value - value significand
//   * scientific - boolean controlling whether scientific notation is used
//   * precision - If positive, specifies the number of decimals to show after
//                 decimal point. If negative, sufficient digits to uniquely
//                 specify the float will be output.
//   * trim_mode - how to treat trailing zeros and decimal point. See TrimMode.
//   * digits_right - pad the result with '' on the right past the decimal point
//   * digits_left - pad the result with '' on the right past the decimal point
//   * exp_digits - Only affects scientific output. If positive, pads the
//                  exponent with 0s until there are this many digits. If
//                  negative, only use sufficient digits.

npy_uint32
Dragon4_PrintFloat32(char *buffer, npy_uint32 bufferSize, npy_float32 value,
                     npy_bool scientific, npy_int32 precision,
                     TrimMode trim_mode, npy_int32 digits_left,
                     npy_int32 digits_right, npy_int32 exp_digits);

npy_uint32
Dragon4_PrintFloat64(char *buffer, npy_uint32 bufferSize, npy_float64 value,
                     npy_bool scientific, npy_int32 precision,
                     TrimMode trim_mode, npy_int32 digits_left,
                     npy_int32 digits_right, npy_int32 exp_digits);

#endif

