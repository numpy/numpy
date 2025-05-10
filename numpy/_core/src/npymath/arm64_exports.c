#if defined(__arm64__) && defined(__APPLE__)

#include <math.h>
/*
 * Export these for scipy, since the SciPy build for macos arm64
 * downloads the macos x86_64 NumPy, and does not error when the
 * linker fails to use npymathlib.a. Importing numpy will expose
 * these external functions
 * See https://github.com/numpy/numpy/issues/22673#issuecomment-1327520055
 *
 * This file is actually compiled as part of the main module.
 */

double npy_asinh(double x) {
    return asinh(x);
}

double npy_copysign(double y, double x) {
    return copysign(y, x);
}

double npy_log1p(double x) {
    return log1p(x);
}

double npy_nextafter(double x, double y) {
    return nextafter(x, y);
}

#endif
