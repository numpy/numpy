/***************************************************************************
 * blitz/memblock.h      MemoryBlock<T> and MemoryBlockReference<T>
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
 * Revision 1.1  2002/01/03 19:50:34  eric
 * renaming compiler to weave
 *
 * Revision 1.1  2001/04/27 17:22:04  ej
 * first attempt to include needed pieces of blitz
 *
 * Revision 1.1.1.1  2000/06/19 12:26:09  tveldhui
 * Imported sources
 *
 * Revision 1.8  1998/12/06 00:00:35  tveldhui
 * Prior to adding UnownedMemoryBlock
 *
 * Revision 1.7  1998/06/15 16:07:01  tveldhui
 * When a memory block is created from an existing block of data,
 * add an additional reference count so that makeUnique() will
 * create a copy of the data.
 *
 * Revision 1.6  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.5  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.4  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 * Revision 1.3  1997/01/23 03:28:28  tveldhui
 * Periodic RCS update
 *
 * Revision 1.2  1997/01/13 22:19:58  tveldhui
 * Periodic RCS update
 *
 * Revision 1.1  1996/11/11 17:29:13  tveldhui
 * Initial revision
 *
 *
 ***************************************************************************
 *
 */

#ifndef __BZ_MEMBLOCK_H__
#define __BZ_MEMBLOCK_H__

#ifndef BZ_BLITZ_H
 #include <blitz/blitz.h>
#endif

#ifndef BZ_NUMTRAIT_H
 #include <blitz/numtrait.h>
#endif

#include <stddef.h>     // ptrdiff_t

BZ_NAMESPACE(blitz)

enum preexistingMemoryPolicy { 
  duplicateData, 
  deleteDataWhenDone, 
  neverDeleteData 
};

// Forward declaration of MemoryBlockReference
template<class T_type> class MemoryBlockReference;

// Class MemoryBlock provides a reference-counted block of memory.  This block
// may be referred to by multiple vector, matrix and array objects.  The memory
// is automatically deallocated when the last referring object is destructed.
// MemoryBlock may be subclassed to provide special allocators.
template<class P_type>
class MemoryBlock {

    friend class MemoryBlockReference<P_type>;

public:
    typedef P_type T_type;

protected:
    MemoryBlock()
    {
        length_ = 0;
        data_ = 0;
        dataBlockAddress_ = 0;
        references_ = 0;
    }

    _bz_explicit MemoryBlock(size_t items)
    {
        length_ = items;
        allocate(length_);

#ifdef BZ_DEBUG_LOG_ALLOCATIONS
    cout << "MemoryBlock: allocated " << setw(8) << length_ 
         << " at " << ((void *)dataBlockAddress_) << endl;
#endif

        BZASSERT(dataBlockAddress_ != 0);

        references_ = 0;
    }

    MemoryBlock(size_t length, T_type* _bz_restrict data)
    {
        length_ = length;
        data_ = data;
        dataBlockAddress_ = 0;
        references_ = 0;
    }

    virtual ~MemoryBlock()
    {
        if (dataBlockAddress_) 
        {

#ifdef BZ_DEBUG_LOG_ALLOCATIONS
    cout << "MemoryBlock:     freed " << setw(8) << length_
         << " at " << ((void *)dataBlockAddress_) << endl;
#endif

            deallocate();
        }
    }

    void          addReference()
    { 
        ++references_; 

#ifdef BZ_DEBUG_LOG_REFERENCES
    cout << "MemoryBlock:    reffed " << setw(8) << length_ 
         << " at " << ((void *)dataBlockAddress_) << " (r=" 
         << (int)references_ << ")" << endl;
#endif

    }

    T_type* _bz_restrict      data() 
    { 
        return data_; 
    }

    const T_type* _bz_restrict data()      const
    { 
        return data_; 
    }

    size_t        length()    const
    { 
        return length_; 
    }

    void              removeReference()
    {
        --references_;

#ifdef BZ_DEBUG_LOG_REFERENCES
    cout << "MemoryBlock: dereffed  " << setw(8) << length_
         << " at " << ((void *)dataBlockAddress_) << " (r=" << (int)references_ 
         << ")" << endl;
#endif
    }

    int references() const
    {
        return references_;
    }

protected:
    inline void allocate(int length);
    void deallocate();

private:   // Disabled member functions
    MemoryBlock(const MemoryBlock<T_type>&)
    { }

    void operator=(const MemoryBlock<T_type>&)
    { }

private:   // Data members
    T_type * _bz_restrict data_;
    T_type * _bz_restrict dataBlockAddress_;

#ifdef BZ_DEBUG_REFERENCE_ROLLOVER
    unsigned char references_;
#else
    int     references_;
#endif

    size_t  length_;
};

template<class P_type>
class UnownedMemoryBlock : public MemoryBlock<P_type> {
public:
    UnownedMemoryBlock(size_t length, P_type* _bz_restrict data)
        : MemoryBlock<P_type>(length,data)
    {
    }

    virtual ~UnownedMemoryBlock()
    {
    }
};

template<class P_type>
class NullMemoryBlock : public MemoryBlock<P_type> {
public:
    NullMemoryBlock()
    { 
        // This ensures that the delete operator will not be invoked
        // on an instance of NullMemoryBlock in removeReference().
        this->addReference();        
    }

    virtual ~NullMemoryBlock()  
    { }
};

template<class P_type>
class MemoryBlockReference {

public:
    typedef P_type T_type;

protected:
    T_type * _bz_restrict data_;

private:
    MemoryBlock<T_type>* block_;
    static NullMemoryBlock<T_type> nullBlock_;

public:

    MemoryBlockReference()
    {
        block_ = &nullBlock_;
        block_->addReference();
        data_ = 0;
    }

    MemoryBlockReference(MemoryBlockReference<T_type>& ref)
    {
        block_ = ref.block_;
        block_->addReference();
        data_ = block_->data();
    }

    MemoryBlockReference(MemoryBlockReference<T_type>& ref, size_t offset)
    {
        block_ = ref.block_;
        block_->addReference();
        data_ = block_->data() + offset;
    }

    MemoryBlockReference(size_t length, T_type* data, 
        preexistingMemoryPolicy deletionPolicy)
    {
        // Create a memory block using already allocated memory. 

        // Note: if the deletionPolicy is duplicateData, this must
        // be handled by the leaf class.  In MemoryBlockReference,
        // this is treated as neverDeleteData; the leaf class (e.g. Array)
        // must duplicate the data.

        if ((deletionPolicy == neverDeleteData) 
          || (deletionPolicy == duplicateData))
            block_ = new UnownedMemoryBlock<T_type>(length, data);
        else if (deletionPolicy == deleteDataWhenDone)
            block_ = new MemoryBlock<T_type>(length, data);
        block_->addReference();

#ifdef BZ_DEBUG_LOG_ALLOCATIONS
    cout << "MemoryBlockReference: created MemoryBlock at "
         << ((void*)block_) << endl;
#endif

        data_ = data;
    }

    _bz_explicit MemoryBlockReference(size_t items)
    {
        block_ = new MemoryBlock<T_type>(items);
        block_->addReference();
        data_ = block_->data();

#ifdef BZ_DEBUG_LOG_ALLOCATIONS
    cout << "MemoryBlockReference: created MemoryBlock at "
         << ((void*)block_) << endl;
#endif

    }

    void blockRemoveReference()
    {
        block_->removeReference();
        if ((block_->references() == 0) && (block_ != &nullBlock_))
        {
#ifdef BZ_DEBUG_LOG_ALLOCATIONS
    cout << "MemoryBlock: no more refs, delete MemoryBlock object at "
         << ((void*)block_) << endl;
#endif

            delete block_;
        }
    }

   ~MemoryBlockReference()
    {
        blockRemoveReference();
    }

    int numReferences() const
    {
        return block_->references();
    }

protected:

    void changeToNullBlock()
    {
        blockRemoveReference();
        block_ = &nullBlock_;
        block_->addReference();
        data_ = 0;
    }

    void changeBlock(MemoryBlockReference<T_type>& ref, size_t offset)
    {
        blockRemoveReference();
        block_ = ref.block_;
        block_->addReference();
        data_ = block_->data() + offset;
    }

    void newBlock(size_t items)
    {
        blockRemoveReference();
        block_ = new MemoryBlock<T_type>(items);
        block_->addReference();
        data_ = block_->data();

#ifdef BZ_DEBUG_LOG_ALLOCATIONS
    cout << "MemoryBlockReference: created MemoryBlock at "
         << ((void*)block_) << endl;
#endif
    }

private:
    void operator=(const MemoryBlockReference<T_type>&)
    { }
};


BZ_NAMESPACE_END

#include <blitz/memblock.cc>

#endif // __BZ_MEMBLOCK_H__
