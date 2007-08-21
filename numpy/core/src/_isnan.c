/* Adapted from cephes */

static int
isnan(double x)
{
    union
    {
        double d;
        unsigned short s[4];
        unsigned int i[2];
    } u;

    u.d = x;

#if SIZEOF_INT == 4

#ifdef WORDS_BIGENDIAN /* defined in pyconfig.h */
    if( ((u.i[0] & 0x7ff00000) == 0x7ff00000)
        && (((u.i[0] & 0x000fffff) != 0) || (u.i[1] != 0)))
        return 1;
#else
    if( ((u.i[1] & 0x7ff00000) == 0x7ff00000)
        && (((u.i[1] & 0x000fffff) != 0) || (u.i[0] != 0)))
        return 1;
#endif

#else  /* SIZEOF_INT != 4 */

#ifdef WORDS_BIGENDIAN
    if( (u.s[0] & 0x7ff0) == 0x7ff0)
        {
            if( ((u.s[0] & 0x000f) | u.s[1] | u.s[2] | u.s[3]) != 0 )
                return 1;
        }
#else
    if( (u.s[3] & 0x7ff0) == 0x7ff0) 
        {
            if( ((u.s[3] & 0x000f) | u.s[2] | u.s[1] | u.s[0]) != 0 )
                return 1;
        }
#endif

#endif  /* SIZEOF_INT */

    return 0;
}
