/*
 * $Id$
 *
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.  Please see <blitz/blitz.h> for terms and
 * conditions of use.
 *
 * $Log$
 * Revision 1.1  2002/01/03 19:50:34  eric
 * renaming compiler to weave
 *
 * Revision 1.1  2001/04/27 17:25:28  ej
 * Looks like I need all the .cc files for blitz also
 *
 * Revision 1.1.1.1  2000/06/19 12:26:08  tveldhui
 * Imported sources
 *
 * Revision 1.4  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.3  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.2  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 */

#ifndef BZ_MEMBLOCK_CC
#define BZ_MEMBLOCK_CC

#ifndef BZ_MEMBLOCK_H
 #include <blitz/memblock.h>
#endif

#ifndef BZ_NUMTRAIT_H
 #include <blitz/numtrait.h>
#endif

BZ_NAMESPACE(blitz)

// Null memory block for each (template) instantiation of MemoryBlockReference
template<class P_type> 
NullMemoryBlock<P_type> MemoryBlockReference<P_type>::nullBlock_;

template<class P_type>
void MemoryBlock<P_type>::deallocate()
{
#ifndef BZ_ALIGN_BLOCKS_ON_CACHELINE_BOUNDARY
    delete [] dataBlockAddress_;
#else
    if (!NumericTypeTraits<T_type>::hasTrivialCtor) {
        for (int i=0; i < length_; ++i)
            data_[i].~T_type();
        delete [] reinterpret_cast<char*>(dataBlockAddress_);
    }
    else {
        delete [] dataBlockAddress_;
    }
#endif
}

template<class P_type>
inline void MemoryBlock<P_type>::allocate(int length)
{
    TAU_TYPE_STRING(p1, "MemoryBlock<T>::allocate() [T="
        + CT(P_type) + "]");
    TAU_PROFILE(p1, "void ()", TAU_BLITZ);

#ifndef BZ_ALIGN_BLOCKS_ON_CACHELINE_BOUNDARY
    data_ =  new T_type[length];
    dataBlockAddress_ = data_;
#else
    int numBytes = length * sizeof(T_type);

    if (numBytes < 1024)
    {
        data_ =  new T_type[length];
        dataBlockAddress_ = data_;
    }
    else
    {
        // We're allocating a large array.  For performance reasons,
        // it's advantageous to force the array to start on a
        // cache line boundary.  We do this by allocating a little
        // more memory than necessary, then shifting the pointer
        // to the next cache line boundary.

        // Patches by Petter Urkedal to support types with nontrivial
        // constructors.

        const int cacheBlockSize = 128;    // Will work for 32, 16 also

        dataBlockAddress_ = reinterpret_cast<T_type*>
            (new char[numBytes + cacheBlockSize - 1]);

        // Shift to the next cache line boundary

        ptrdiff_t offset = ptrdiff_t(dataBlockAddress_) % cacheBlockSize;
        int shift = (offset == 0) ? 0 : (cacheBlockSize - offset);
        data_ = (T_type*)(((char *)dataBlockAddress_) + shift);

        // Use placement new to construct types with nontrival ctors
        if (!NumericTypeTraits<T_type>::hasTrivialCtor) {
            for (int i=0; i < length; ++i)
                new(&data_[i]) T_type;
        }
    }
#endif
}


BZ_NAMESPACE_END

#endif // BZ_MEMBLOCK_CC
