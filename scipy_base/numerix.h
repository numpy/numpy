#if !defined(_NUMERIX_)
#define _NUMERIX_

#if defined(NUMARRAY)
#   include "numarray/arrayobject.h"
#else
#   include "Numeric/arrayobject.h"
#endif

#if defined(NUMARRAY)
#   define NX_ZERO(a) PyArray_Zero(a)
#else
#   define NX_ZERO(a) (a)->descr->zero
#endif

#if defined(NUMARRAY)
#   define NX_ONE(a) PyArray_One(a)
#else
#   define NX_ONE(a) (a)->descr->one
#endif

#if defined(NUMARRAY)
#   define NX_GETITEM(a, ptr) (a)->descr->_get((a), ((char*)ptr)-(a)->data)
#else
#   define NX_GETITEM(a, ptr) (a)->descr->getitem(ptr)
#endif

#if defined(NUMARRAY)
#   define NX_SETITEM(a, ptr, value) (a)->descr->_set((a), ((char*)ptr)-(a)->data, value)
#else
#   define NX_SETITEM(a, ptr, value) (a)->descr->setitem(value, ptr)
#endif

#endif
