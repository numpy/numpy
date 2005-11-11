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
 * Revision 1.2  2002/09/12 07:04:04  eric
 * major rewrite of weave.
 *
 * 0.
 * The underlying library code is significantly re-factored and simpler. There used to be a xxx_spec.py and xxx_info.py file for every group of type conversion classes.  The spec file held the python code that handled the conversion and the info file had most of the C code templates that were generated.  This proved pretty confusing in practice, so the two files have mostly been merged into the spec file.
 *
 * Also, there was quite a bit of code duplication running around.  The re-factoring was able to trim the standard conversion code base (excluding blitz and accelerate stuff) by about 40%.  This should be a huge maintainability and extensibility win.
 *
 * 1.
 * With multiple months of using Numeric arrays, I've found some of weave's "magic variable" names unwieldy and want to change them.  The following are the old declarations for an array x of Float32 type:
 *
 *         PyArrayObject* x = convert_to_numpy(...);
 *         float* x_data = (float*) x->data;
 *         int*   _Nx = x->dimensions;
 *         int*   _Sx = x->strides;
 *         int    _Dx = x->nd;
 *
 * The new declaration looks like this:
 *
 *         PyArrayObject* x_array = convert_to_numpy(...);
 *         float* x = (float*) x->data;
 *         int*   Nx = x->dimensions;
 *         int*   Sx = x->strides;
 *         int    Dx = x->nd;
 *
 * This is obviously not backward compatible, and will break some code (including a lot of mine).  It also makes inline() code more readable and natural to write.
 *
 * 2.
 * I've switched from CXX to Gordon McMillan's SCXX for list, tuples, and dictionaries.  I like CXX pretty well, but its use of advanced C++ (templates, etc.) caused some portability problems.  The SCXX library is similar to CXX but doesn't use templates at all.  This, like (1) is not an
 * API compatible change and requires repairing existing code.
 *
 * I have also thought about boost python, but it also makes heavy use of templates.  Moving to SCXX gets rid of almost all template usage for the standard type converters which should help portability.  std::complex and std::string from the STL are the only templates left.  Of course blitz still uses templates in a major way so weave.blitz will continue to be hard on compilers.
 *
 * I've actually considered scrapping the C++ classes for list, tuples, and
 * dictionaries, and just fall back to the standard Python C API because the classes are waaay slower than the raw API in many cases.  They are also more convenient and less error prone in many cases, so I've decided to stick with them.  The PyObject variable will always be made available for variable "x" under the name "py_x" for more speedy operations.  You'll definitely want to use these for anything that needs to be speedy.
 *
 * 3.
 * strings are converted to std::string now.  I found this to be the most useful type in for strings in my code.  Py::String was used previously.
 *
 * 4.
 * There are a number of reference count "errors" in some of the less tested conversion codes such as instance, module, etc.  I've cleaned most of these up.  I put errors in quotes here because I'm actually not positive that objects passed into "inline" really need reference counting applied to them.  The dictionaries passed in by inline() hold references to these objects so it doesn't seem that they could ever be garbage collected inadvertently.  Variables used by ext_tools, though, definitely need the reference counting done.  I don't think this is a major cost in speed, so it probably isn't worth getting rid of the ref count code.
 *
 * 5.
 * Unicode objects are now supported.  This was necessary to support rendering Unicode strings in the freetype wrappers for Chaco.
 *
 * 6.
 * blitz++ was upgraded to the latest CVS.  It compiles about twice as fast as the old blitz and looks like it supports a large number of compilers (though only gcc 2.95.3 is tested).  Compile times now take about 9 seconds on my 850 MHz PIII laptop.
 *
 * Revision 1.9  2002/07/19 20:42:53  jcumming
 * Removed ending semicolon after invocations of BZ_MUTEX_* macros.  This
 * is now handled within the definition of these macros.  This should get
 * rid of compiler warnings from SGI CC and others about extra semicolons
 * being ignored, which happened when these macros were defined as blank.
 *
 * Revision 1.8  2002/07/17 22:10:09  jcumming
 * Added missing semicolon after use of BZ_MUTEX_DECLARE macro.
 *
 * Revision 1.7  2002/05/27 19:35:37  jcumming
 * Changed this->addReference() to MemoryBlock<P_type>::addReference().
 * Use base class name as scoping qualifier rather than "this" pointer.
 *
 * Revision 1.6  2002/03/06 18:07:42  patricg
 *
 * in the constructor
 * MemoryBlock(size_t length, T_type* _bz_restrict data)
 * dataBlockAddress_ = data replaced by dataBlockAddress_ = 0
 * as it was before. (testsuite/extract does not crash then)
 *
 * Revision 1.5  2002/02/28 23:38:20  tveldhui
 * Fixed extra semicolon problem with KCC
 *
 * Revision 1.4  2001/02/04 16:32:28  tveldhui
 * Made memory block reference counting (optionally) threadsafe when
 * BZ_THREADSAFE is defined.  Currently uses pthread mutex.
 * When compiling with gcc -pthread, _REENTRANT automatically causes
 * BZ_THREADSAFE to be enabled.
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

#ifdef BZ_THREADSAFE
 #include <pthread.h>
#endif

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

        BZ_MUTEX_INIT(mutex)
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

        BZ_MUTEX_INIT(mutex)
    }

    MemoryBlock(size_t length, T_type* _bz_restrict data)
    {
        length_ = length;
        data_ = data;
        dataBlockAddress_ = 0;
        references_ = 0;
        BZ_MUTEX_INIT(mutex)
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

        BZ_MUTEX_DESTROY(mutex)
    }

    void          addReference()
    { 
        BZ_MUTEX_LOCK(mutex)
        ++references_; 

#ifdef BZ_DEBUG_LOG_REFERENCES
    cout << "MemoryBlock:    reffed " << setw(8) << length_ 
         << " at " << ((void *)dataBlockAddress_) << " (r=" 
         << (int)references_ << ")" << endl;
#endif
        BZ_MUTEX_UNLOCK(mutex)

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

    int           removeReference()
    {

        BZ_MUTEX_LOCK(mutex)
        int refcount = --references_;

#ifdef BZ_DEBUG_LOG_REFERENCES
    cout << "MemoryBlock: dereffed  " << setw(8) << length_
         << " at " << ((void *)dataBlockAddress_) << " (r=" << (int)references_ 
         << ")" << endl;
#endif
        BZ_MUTEX_UNLOCK(mutex)
        return refcount;
    }

    int references() const
    {
        BZ_MUTEX_LOCK(mutex)
        int refcount = references_;
        BZ_MUTEX_UNLOCK(mutex)

        return refcount;
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
    volatile unsigned char references_;
#else
    volatile int references_;
#endif

    BZ_MUTEX_DECLARE(mutex)
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
        MemoryBlock<P_type>::addReference();        
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
        int refcount = block_->removeReference();
        if ((refcount == 0) && (block_ != &nullBlock_))
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
