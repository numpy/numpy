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

#ifdef WORDS_BIGENDIAN /* defined in pyconfig.h */
    return u.i[0] < 0;
#else
    return u.i[1] < 0;
#endif

#else  /* SIZEOF_INT != 4 */

#ifdef WORDS_BIGENDIAN
    return u.s[0] < 0;
#else
    return u.s[3] < 0;
#endif

#endif  /* SIZEOF_INT */
}
