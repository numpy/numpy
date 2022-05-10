#ifndef NUMPY_CORE_INCLUDE_NUMPY_HALFFLOAT_H_
#define NUMPY_CORE_INCLUDE_NUMPY_HALFFLOAT_H_

#include <Python.h>
#include <numpy/npy_math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* since NumPy 1.24, 2022-05, npy_half are represented as a struct instead of a
 * type alias.
 * To keep ABI compatibility with older version of numpy, npy routines that
 * manipulates are renamed using the convention defined in macro NPY_HALF_API.
 * Each symbol is then redefined based on that naming scheme to keep client code
 * mostly unchanged at API level.
 *
 * If for any reason, the legacy API must be used, define NPY_USE_LEGACY_HALF
 * before including this file. This behavior is not expected for internal Numpy
 * code.
 */

#define NPY_HALF_LEGACY_API(name) npy_##name

#ifdef NPY_USE_LEGACY_HALF

#define NPY_HALF_API(name) NPY_HALF_LEGACY_API(name)
#define NPY_HALF_BITS(v) v
#define NPY_HALF_BUILD(v) v
#define NPY_HALF_INIT(v) v

#else

#define NPY_HALF_API(name) npy_strongly_typed_##name
#define NPY_HALF_BITS(v) ((v).bits)
#define NPY_HALF_BUILD(v) ((npy_half){v})
#define NPY_HALF_INIT(v) {v}

#endif

/*
 * Half-precision routines
 */

/* Conversions */
#define npy_half_to_float NPY_HALF_API(half_to_float)
float npy_half_to_float(npy_half h);

#define npy_half_to_double NPY_HALF_API(half_to_double)
double npy_half_to_double(npy_half h);

#define npy_float_to_half NPY_HALF_API(float_to_half)
npy_half npy_float_to_half(float f);

#define npy_double_to_half NPY_HALF_API(double_to_half)
npy_half npy_double_to_half(double d);

/* Comparisons */
#define npy_half_eq NPY_HALF_API(half_eq)
int npy_half_eq(npy_half h1, npy_half h2);

#define npy_half_ne NPY_HALF_API(half_ne)
int npy_half_ne(npy_half h1, npy_half h2);

#define npy_half_le NPY_HALF_API(half_le)
int npy_half_le(npy_half h1, npy_half h2);

#define npy_half_lt NPY_HALF_API(half_lt)
int npy_half_lt(npy_half h1, npy_half h2);

#define npy_half_ge NPY_HALF_API(half_ge)
int npy_half_ge(npy_half h1, npy_half h2);

#define npy_half_gt NPY_HALF_API(half_gt)
int npy_half_gt(npy_half h1, npy_half h2);

/* faster *_nonan variants for when you know h1 and h2 are not NaN */
#define npy_half_eq_nonan NPY_HALF_API(half_eq_nonan)
int npy_half_eq_nonan(npy_half h1, npy_half h2);

#define npy_half_lt_nonan NPY_HALF_API(half_lt_nonan)
int npy_half_lt_nonan(npy_half h1, npy_half h2);

#define npy_half_le_nonan NPY_HALF_API(half_le_nonan)
int npy_half_le_nonan(npy_half h1, npy_half h2);

/* Miscellaneous functions */
#define npy_half_copysign NPY_HALF_API(half_copysign)
npy_half npy_half_copysign(npy_half x, npy_half y);

#define npy_half_spacing NPY_HALF_API(half_spacing)
npy_half npy_half_spacing(npy_half h);

#define npy_half_nextafter NPY_HALF_API(half_nextafter)
npy_half npy_half_nextafter(npy_half x, npy_half y);

#define npy_half_divmod NPY_HALF_API(half_divmod)
npy_half npy_half_divmod(npy_half x, npy_half y, npy_half *modulus);

#ifdef NPY_USE_LEGACY_HALF

int NPY_HALF_API(half_iszero)(npy_half_bits_t h);
int NPY_HALF_API(half_isnan)(npy_half_bits_t h);
int NPY_HALF_API(half_isinf)(npy_half_bits_t h);
int NPY_HALF_API(half_isfinite)(npy_half_bits_t h);
int NPY_HALF_API(half_signbit)(npy_half_bits_t h);

#else

#define npy_half_iszero NPY_HALF_API(half_iszero)
NPY_INLINE int npy_half_iszero(npy_half h) {
  return (NPY_HALF_BITS(h)&0x7fff) == 0;
}

#define npy_half_isnan NPY_HALF_API(half_isnan)
NPY_INLINE int npy_half_isnan(npy_half h) {
    return ((NPY_HALF_BITS(h)&0x7c00u) == 0x7c00u) && ((NPY_HALF_BITS(h)&0x03ffu) != 0x0000u);
}

#define npy_half_isinf NPY_HALF_API(half_isinf)
NPY_INLINE int npy_half_isinf(npy_half h) {
  return ((NPY_HALF_BITS(h)&0x7fffu) == 0x7c00u);
}

#define npy_half_isfinite NPY_HALF_API(half_isfinite)
NPY_INLINE int npy_half_isfinite(npy_half h) {
  return (NPY_HALF_BITS(h)&0x7c00u) != 0x7c00u;
}

#define npy_half_signbit NPY_HALF_API(half_signbit)
NPY_INLINE int npy_half_signbit(npy_half h) {
  return (NPY_HALF_BITS(h)&0x8000u) != 0;
}

#define npy_half_neg NPY_HALF_API(half_neg)
NPY_INLINE npy_half npy_half_neg(npy_half h) {
  npy_half res = NPY_HALF_INIT((npy_uint16)(NPY_HALF_BITS(h)^0x8000u));
  return res;
}

#define npy_half_abs NPY_HALF_API(half_abs)
NPY_INLINE npy_half npy_half_abs(npy_half h) {
  npy_half res = NPY_HALF_INIT((npy_uint16)(NPY_HALF_BITS(h)&0x7fffu));
  return res;
}

#define npy_half_pos NPY_HALF_API(half_pos)
NPY_INLINE npy_half npy_half_pos(npy_half h) {
  npy_half res = NPY_HALF_INIT((npy_uint16)(+NPY_HALF_BITS(h)));
  return res;
}

#endif

/*
 * Half-precision constants
 */

#define NPY_HALF_ZERO   NPY_HALF_BUILD(0x0000u)
#define NPY_HALF_PZERO  NPY_HALF_BUILD(0x0000u)
#define NPY_HALF_NZERO  NPY_HALF_BUILD(0x8000u)
#define NPY_HALF_ONE    NPY_HALF_BUILD(0x3c00u)
#define NPY_HALF_NEGONE NPY_HALF_BUILD(0xbc00u)
#define NPY_HALF_PINF   NPY_HALF_BUILD(0x7c00u)
#define NPY_HALF_NINF   NPY_HALF_BUILD(0xfc00u)
#define NPY_HALF_NAN    NPY_HALF_BUILD(0x7e00u)

#define NPY_MAX_HALF    NPY_HALF_BUILD(0x7bffu)

/*
 * Bit-level conversions
 */

npy_uint16 npy_floatbits_to_halfbits(npy_uint32 f);
npy_uint16 npy_doublebits_to_halfbits(npy_uint64 d);
npy_uint32 npy_halfbits_to_floatbits(npy_uint16 h);
npy_uint64 npy_halfbits_to_doublebits(npy_uint16 h);

#ifdef __cplusplus
}
#endif


#endif  /* NUMPY_CORE_INCLUDE_NUMPY_HALFFLOAT_H_ */
