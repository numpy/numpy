/***************************************************************************
 * blitz/array/slice.h    Helper classes for slicing arrays
 *
 * $Id$
 *
 * Copyright (C) 1997-1999 Todd Veldhuizen <tveldhui@oonumerics.org>
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
 ***************************************************************************
 * $Log$
 * Revision 1.1  2002/01/03 19:50:36  eric
 * renaming compiler to weave
 *
 * Revision 1.1  2001/04/27 17:22:04  ej
 * first attempt to include needed pieces of blitz
 *
 * Revision 1.1.1.1  2000/06/19 12:26:14  tveldhui
 * Imported sources
 *
 * Revision 1.2  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.1  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 */

#ifndef BZ_ARRAYSLICE_H
#define BZ_ARRAYSLICE_H

#ifndef BZ_ARRAY_H
 #error <blitz/array/slice.h> must be included via <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

// Forward declaration
template<class T, int N>
class Array;



class nilArraySection { };

template<class T>
class ArraySectionInfo {
public:
    enum { isValidType = 0, rank = 0, isPick = 0 };
};

template<>
class ArraySectionInfo<Range> {
public:
    enum { isValidType = 1, rank = 1, isPick = 0 };
};

template<>
class ArraySectionInfo<int> {
public:
    enum { isValidType = 1, rank = 0, isPick = 0 };
};

template<>
class ArraySectionInfo<nilArraySection> {
public:
    enum { isValidType = 1, rank = 0, isPick = 0 };
};

template<class T_numtype, class T1, class T2 = nilArraySection, 
    class T3 = nilArraySection, class T4 = nilArraySection, 
    class T5 = nilArraySection, class T6 = nilArraySection, 
    class T7 = nilArraySection, class T8 = nilArraySection, 
    class T9 = nilArraySection, class T10 = nilArraySection, 
    class T11 = nilArraySection>
class SliceInfo {
public:
    enum { 
        numValidTypes = ArraySectionInfo<T1>::isValidType
                      + ArraySectionInfo<T2>::isValidType
                      + ArraySectionInfo<T3>::isValidType
                      + ArraySectionInfo<T4>::isValidType
                      + ArraySectionInfo<T5>::isValidType
                      + ArraySectionInfo<T6>::isValidType
                      + ArraySectionInfo<T7>::isValidType
                      + ArraySectionInfo<T8>::isValidType
                      + ArraySectionInfo<T9>::isValidType
                      + ArraySectionInfo<T10>::isValidType
                      + ArraySectionInfo<T11>::isValidType,

        rank          = ArraySectionInfo<T1>::rank
                      + ArraySectionInfo<T2>::rank
                      + ArraySectionInfo<T3>::rank
                      + ArraySectionInfo<T4>::rank
                      + ArraySectionInfo<T5>::rank
                      + ArraySectionInfo<T6>::rank
                      + ArraySectionInfo<T7>::rank
                      + ArraySectionInfo<T8>::rank
                      + ArraySectionInfo<T9>::rank
                      + ArraySectionInfo<T10>::rank
                      + ArraySectionInfo<T11>::rank,

        isPick        = ArraySectionInfo<T1>::isPick
                      + ArraySectionInfo<T2>::isPick
                      + ArraySectionInfo<T3>::isPick
                      + ArraySectionInfo<T4>::isPick
                      + ArraySectionInfo<T5>::isPick
                      + ArraySectionInfo<T6>::isPick
                      + ArraySectionInfo<T7>::isPick
                      + ArraySectionInfo<T8>::isPick
                      + ArraySectionInfo<T9>::isPick
                      + ArraySectionInfo<T10>::isPick
                      + ArraySectionInfo<T11>::isPick
        };

    typedef Array<T_numtype,numValidTypes> T_array;
    typedef Array<T_numtype,rank> T_slice;
};

BZ_NAMESPACE_END

#endif // BZ_ARRAYSLICE_H
