#ifndef NPY_EXTINT128_H_
#define NPY_EXTINT128_H_


typedef struct {
    signed char sign;
    npy_uint64 lo, hi;
} npy_extint128_t;


/* Integer addition with overflow checking */
static NPY_INLINE npy_int64
safe_add(npy_int64 a, npy_int64 b, char *overflow_flag)
{
    if (a > 0 && b > NPY_MAX_INT64 - a) {
        *overflow_flag = 1;
    }
    else if (a < 0 && b < NPY_MIN_INT64 - a) {
        *overflow_flag = 1;
    }
    return a + b;
}


/* Integer subtraction with overflow checking */
static NPY_INLINE npy_int64
safe_sub(npy_int64 a, npy_int64 b, char *overflow_flag)
{
    if (a >= 0 && b < a - NPY_MAX_INT64) {
        *overflow_flag = 1;
    }
    else if (a < 0 && b > a - NPY_MIN_INT64) {
        *overflow_flag = 1;
    }
    return a - b;
}


/* Integer multiplication with overflow checking */
static NPY_INLINE npy_int64
safe_mul(npy_int64 a, npy_int64 b, char *overflow_flag)
{
    if (a > 0) {
        if (b > NPY_MAX_INT64 / a || b < NPY_MIN_INT64 / a) {
            *overflow_flag = 1;
        }
    }
    else if (a < 0) {
        if (b > 0 && a < NPY_MIN_INT64 / b) {
            *overflow_flag = 1;
        }
        else if (b < 0 && a < NPY_MAX_INT64 / b) {
            *overflow_flag = 1;
        }
    }
    return a * b;
}


/* Long integer init */
static NPY_INLINE npy_extint128_t
to_128(npy_int64 x)
{
    npy_extint128_t result;
    result.sign = (x >= 0 ? 1 : -1);
    if (x >= 0) {
        result.lo = x;
    }
    else {
        result.lo = (npy_uint64)(-(x + 1)) + 1;
    }
    result.hi = 0;
    return result;
}


static NPY_INLINE npy_int64
to_64(npy_extint128_t x, char *overflow)
{
    if (x.hi != 0 ||
        (x.sign > 0 && x.lo > NPY_MAX_INT64) ||
        (x.sign < 0 && x.lo != 0 && x.lo - 1 > -(NPY_MIN_INT64 + 1))) {
        *overflow = 1;
    }
    return x.lo * x.sign;
}


/* Long integer multiply */
static NPY_INLINE npy_extint128_t
mul_64_64(npy_int64 a, npy_int64 b)
{
    npy_extint128_t x, y, z;
    npy_uint64 x1, x2, y1, y2, r1, r2, prev;

    x = to_128(a);
    y = to_128(b);

    x1 = x.lo & 0xffffffff;
    x2 = x.lo >> 32;

    y1 = y.lo & 0xffffffff;
    y2 = y.lo >> 32;

    r1 = x1*y2;
    r2 = x2*y1;

    z.sign = x.sign * y.sign;
    z.hi = x2*y2 + (r1 >> 32) + (r2 >> 32);
    z.lo = x1*y1;

    /* Add with carry */
    prev = z.lo;
    z.lo += (r1 << 32);
    if (z.lo < prev) {
        ++z.hi;
    }

    prev = z.lo;
    z.lo += (r2 << 32);
    if (z.lo < prev) {
        ++z.hi;
    }

    return z;
}


/* Long integer add */
static NPY_INLINE npy_extint128_t
add_128(npy_extint128_t x, npy_extint128_t y, char *overflow)
{
    npy_extint128_t z;

    if (x.sign == y.sign) {
        z.sign = x.sign;
        z.hi = x.hi + y.hi;
        if (z.hi < x.hi) {
            *overflow = 1;
        }
        z.lo = x.lo + y.lo;
        if (z.lo < x.lo) {
            if (z.hi == NPY_MAX_UINT64) {
                *overflow = 1;
            }
            ++z.hi;
        }
    }
    else if (x.hi > y.hi || (x.hi == y.hi && x.lo >= y.lo)) {
        z.sign = x.sign;
        z.hi = x.hi - y.hi;
        z.lo = x.lo;
        z.lo -= y.lo;
        if (z.lo > x.lo) {
            --z.hi;
        }
    }
    else {
        z.sign = y.sign;
        z.hi = y.hi - x.hi;
        z.lo = y.lo;
        z.lo -= x.lo;
        if (z.lo > y.lo) {
            --z.hi;
        }
    }

    return z;
}


/* Long integer negation */
static NPY_INLINE npy_extint128_t
neg_128(npy_extint128_t x)
{
    npy_extint128_t z = x;
    z.sign *= -1;
    return z;
}


static NPY_INLINE npy_extint128_t
sub_128(npy_extint128_t x, npy_extint128_t y, char *overflow)
{
    return add_128(x, neg_128(y), overflow);
}


static NPY_INLINE npy_extint128_t
shl_128(npy_extint128_t v)
{
    npy_extint128_t z;
    z = v;
    z.hi <<= 1;
    z.hi |= (z.lo & (((npy_uint64)1) << 63)) >> 63;
    z.lo <<= 1;
    return z;
}


static NPY_INLINE npy_extint128_t
shr_128(npy_extint128_t v)
{
    npy_extint128_t z;
    z = v;
    z.lo >>= 1;
    z.lo |= (z.hi & 0x1) << 63;
    z.hi >>= 1;
    return z;
}

static NPY_INLINE int
gt_128(npy_extint128_t a, npy_extint128_t b)
{
    if (a.sign > 0 && b.sign > 0) {
        return (a.hi > b.hi) || (a.hi == b.hi && a.lo > b.lo);
    }
    else if (a.sign < 0 && b.sign < 0) {
        return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo);
    }
    else if (a.sign > 0 && b.sign < 0) {
        return a.hi != 0 || a.lo != 0 || b.hi != 0 || b.lo != 0;
    }
    else {
        return 0;
    }
}


/* Long integer divide */
static NPY_INLINE npy_extint128_t
divmod_128_64(npy_extint128_t x, npy_int64 b, npy_int64 *mod)
{
    npy_extint128_t remainder, pointer, result, divisor;
    char overflow = 0;

    assert(b > 0);

    if (b <= 1 || x.hi == 0) {
        result.sign = x.sign;
        result.lo = x.lo / b;
        result.hi = x.hi / b;
        *mod = x.sign * (x.lo % b);
        return result;
    }

    /* Long division, not the most efficient choice */
    remainder = x;
    remainder.sign = 1;

    divisor.sign = 1;
    divisor.hi = 0;
    divisor.lo = b;

    result.sign = 1;
    result.lo = 0;
    result.hi = 0;

    pointer.sign = 1;
    pointer.lo = 1;
    pointer.hi = 0;

    while ((divisor.hi & (((npy_uint64)1) << 63)) == 0 &&
           gt_128(remainder, divisor)) {
        divisor = shl_128(divisor);
        pointer = shl_128(pointer);
    }

    while (pointer.lo || pointer.hi) {
        if (!gt_128(divisor, remainder)) {
            remainder = sub_128(remainder, divisor, &overflow);
            result = add_128(result, pointer, &overflow);
        }
        divisor = shr_128(divisor);
        pointer = shr_128(pointer);
    }

    /* Fix signs and return; cannot overflow */
    result.sign = x.sign;
    *mod = x.sign * remainder.lo;

    return result;
}


/* Divide and round down (positive divisor; no overflows) */
static NPY_INLINE npy_extint128_t
floordiv_128_64(npy_extint128_t a, npy_int64 b)
{
    npy_extint128_t result;
    npy_int64 remainder;
    char overflow = 0;
    assert(b > 0);
    result = divmod_128_64(a, b, &remainder);
    if (a.sign < 0 && remainder != 0) {
        result = sub_128(result, to_128(1), &overflow);
    }
    return result;
}


/* Divide and round up (positive divisor; no overflows) */
static NPY_INLINE npy_extint128_t
ceildiv_128_64(npy_extint128_t a, npy_int64 b)
{
    npy_extint128_t result;
    npy_int64 remainder;
    char overflow = 0;
    assert(b > 0);
    result = divmod_128_64(a, b, &remainder);
    if (a.sign > 0 && remainder != 0) {
        result = add_128(result, to_128(1), &overflow);
    }
    return result;
}

#endif
