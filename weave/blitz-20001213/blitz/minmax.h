#ifndef BZ_MINMAX_H
#define BZ_MINMAX_H

#include <blitz/promote.h>

BZ_NAMESPACE(blitz)

/*
 * These functions are in their own namespace (blitz::minmax) to avoid
 * conflicts with the array reduction operations min and max.
 */

BZ_NAMESPACE(minmax)

template<class T1, class T2>
BZ_PROMOTE(T1,T2) min(const T1& a, const T2& b)
{
    typedef BZ_PROMOTE(T1,T2) T_promote;

    if (a <= b)
        return T_promote(a);
    else
        return T_promote(b);
}

template<class T1, class T2>
BZ_PROMOTE(T1,T2) max(const T1& a, const T2& b)
{
    typedef BZ_PROMOTE(T1,T2) T_promote;

    if (a >= b)
        return T_promote(a);
    else
        return T_promote(b);
}

BZ_NAMESPACE_END

BZ_NAMESPACE_END

#endif
