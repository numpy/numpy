/***************************************************************************
 * blitz/array/shape.h        
 *
 * Copyright (C) 1997-2001 Todd Veldhuizen <tveldhui@oonumerics.org>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * Suggestions:          blitz-dev@oonumerics.org
 * Bugs:                 blitz-bugs@oonumerics.org
 *
 * For more information, please see the Blitz++ Home Page:
 *    http://oonumerics.org/blitz/
 *
 ****************************************************************************/
#ifndef BZ_ARRAYSHAPE_H
#define BZ_ARRAYSHAPE_H

#ifndef BZ_ARRAY_H
 #error <blitz/array/shape.h> must be included via <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

/*
 * These routines make it easier to create shape parameters on
 * the fly: instead of having to write
 *
 * A.resize(TinyVector<int,4>(8,8,8,12));
 *
 * you can just say
 *
 * A.resize(shape(8,8,8,12));
 *
 */
inline TinyVector<int,1> shape(int n1)
{ return TinyVector<int,1>(n1); }

inline TinyVector<int,2> shape(int n1, int n2)
{ return TinyVector<int,2>(n1,n2); }

inline TinyVector<int,3> shape(int n1, int n2, int n3)
{ return TinyVector<int,3>(n1,n2,n3); }

inline TinyVector<int,4> shape(int n1, int n2, int n3, int n4)
{ return TinyVector<int,4>(n1,n2,n3,n4); }

inline TinyVector<int,5> shape(int n1, int n2, int n3, int n4,
    int n5)
{ return TinyVector<int,5>(n1,n2,n3,n4,n5); }

inline TinyVector<int,6> shape(int n1, int n2, int n3, int n4,
    int n5, int n6)
{ return TinyVector<int,6>(n1,n2,n3,n4,n5,n6); }

inline TinyVector<int,7> shape(int n1, int n2, int n3, int n4,
    int n5, int n6, int n7)
{ return TinyVector<int,7>(n1,n2,n3,n4,n5,n6,n7); }

inline TinyVector<int,8> shape(int n1, int n2, int n3, int n4,
    int n5, int n6, int n7, int n8)
{ return TinyVector<int,8>(n1,n2,n3,n4,n5,n6,n7,n8); }

inline TinyVector<int,9> shape(int n1, int n2, int n3, int n4,
    int n5, int n6, int n7, int n8, int n9)
{ return TinyVector<int,9>(n1,n2,n3,n4,n5,n6,n7,n8,n9); }

inline TinyVector<int,10> shape(int n1, int n2, int n3, int n4,
    int n5, int n6, int n7, int n8, int n9, int n10)
{ return TinyVector<int,10>(n1,n2,n3,n4,n5,n6,n7,n8,n9,n10); }

inline TinyVector<int,11> shape(int n1, int n2, int n3, int n4,
    int n5, int n6, int n7, int n8, int n9, int n10, int n11)
{ return TinyVector<int,11>(n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11); }

BZ_NAMESPACE_END

#endif // BZ_ARRAYSHAPE_H

