/* Adapted from cephes */

static int
signbit_d(double x)
{
    union
    {
        double d;
        short s[4];
        int i[2];
    } u;

    u.d = x;

#if SIZEOF_INT == 4

#ifdef NPY_BIG_ENDIAN
    return u.i[0] < 0;
#else
    return u.i[1] < 0;
#endif

#else  /* SIZEOF_INT != 4 */

#ifdef NPY_BIG_ENDIAN
    return u.s[0] < 0;
#else
    return u.s[3] < 0;
#endif

#endif  /* SIZEOF_INT */
}
