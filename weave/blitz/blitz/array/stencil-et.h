/***************************************************************************
 * blitz/array/stencil-et.h  Expression-template-capabale stencils
 *
 * $Id$
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
 ***************************************************************************
 * $Log$
 * Revision 1.2  2002/09/12 07:02:06  eric
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
 * Revision 1.5  2002/05/27 19:43:30  jcumming
 * Removed use of this->.  iter_ is now declared in scope of derived classes.
 *
 * Revision 1.4  2002/04/17 16:56:42  patricg
 *
 * replaced T_numtype with P_numtype in every macros definitions. Fixed a
 * compilation problem with aCC/HP in the stencils examples (stencils2.cpp,
 * stencil3.cpp, stencilet.cpp) in the directory examples.
 * Suggested by Robert W. Techentin <techentin.robert@mayo.edu>
 *
 * Revision 1.3  2002/03/21 15:03:01  patricg
 *
 * replaced iter_ by this->iter_ in derived template classes of StencilExpr
 * template class
 *
 * Revision 1.2  2001/01/25 00:25:55  tveldhui
 * Ensured that source files have cvs logs.
 *
 */

#ifndef BZ_ARRAY_STENCIL_ET_H
#define BZ_ARRAY_STENCIL_ET_H

BZ_NAMESPACE(blitz)

template<typename T_ArrayNumtype, int N_rank, typename T_result>
class StencilExpr 
{
public:
    typedef T_result T_numtype;
    typedef Array<T_ArrayNumtype,N_rank> T_array;
    typedef const T_array& T_ctorArg1;
    typedef int T_ctorArg2;

    enum { numArrayOperands = 1, numIndexPlaceholders = 0,
        rank = N_rank };

    StencilExpr(const T_array& array)
        : iter_(array)
    { }

    ~StencilExpr()
    { }

    // operator* must be declared by subclass
  
    int ascending(int rank)
    { return iter_.ascending(rank); }
 
    int ordering(int rank)
    { return iter_.ordering(rank); }
 
    int lbound(int rank)
    { return iter_.lbound(rank); }

    int ubound(int rank)
    { return iter_.ubound(rank); }

    void push(int position)
    { iter_.push(position); }

    void pop(int position)
    { iter_.pop(position); }

    void advance()
    { iter_.advance(); }

    void advance(int n)
    { iter_.advance(n); }

    void loadStride(int rank)
    { iter_.loadStride(rank); }

    _bz_bool isUnitStride(int rank) const
    { return iter_.isUnitStride(rank); }

    void advanceUnitStride()
    { iter_.advanceUnitStride(); }

    _bz_bool canCollapse(int outerLoopRank, int innerLoopRank) const
    {
        // BZ_DEBUG_MESSAGE("_bz_ArrayExpr<>::canCollapse()");
        return iter_.canCollapse(outerLoopRank, innerLoopRank);
    }

    // T_numtype operator[](int i)   -- don't know how to do that.

    // T_numtype fastRead(int i)     -- ditto

    int suggestStride(int rank) const
    { return iter_.suggestStride(rank); }

    _bz_bool isStride(int rank, int stride) const
    { return iter_.isStride(rank,stride); }

    void prettyPrint(string& str) const
    {
        str += "(stencil)";    // lame, needs work
    }

    void prettyPrint(string& str, prettyPrintFormat& format) const
    {   str += "(stencil)"; }

    template<class T_shape>
    _bz_bool shapeCheck(const T_shape& shape)
    { return iter_.shapeCheck(shape); }

    void moveTo(const TinyVector<int,N_rank>& i)
    {
        iter_.moveTo(i);
    }

protected:
    FastArrayIterator<T_ArrayNumtype,N_rank> iter_;
};

#define BZ_ET_STENCIL(name,result) \
template<class P_numtype, int N_rank> \
class name ## _et : public StencilExpr<P_numtype,N_rank,result>, \
  public ETBase<name ## _et<P_numtype,N_rank> > \
 { \
private: \
    typedef StencilExpr<P_numtype,N_rank,result> T_base; \
    using T_base::iter_; \
public: \
    name ## _et(const Array<P_numtype,N_rank>& A) \
        : StencilExpr<P_numtype,N_rank,result>(A) \
    { } \
    result operator*() \
    { return name(iter_); } \
    result operator()(const TinyVector<int,N_rank>& a) \
    { iter_.moveTo(a); return name(iter_); } \
    result fastRead(int i) \
    { \
      const P_numtype* tmp = iter_.data(); \
      iter_._bz_setData(tmp + i); \
      P_numtype r = name(iter_); \
      iter_._bz_setData(tmp); \
      return r; \
    } \
}; \
template<class P_numtype, int N_rank> \
inline _bz_ArrayExpr<name ## _et<P_numtype, N_rank> > \
name(Array<P_numtype,N_rank>& A) \
{ \
    return _bz_ArrayExpr<name ## _et<P_numtype, N_rank> >(A); \
}

#define BZ_ET_STENCILV(name,rank) \
template<class P_numtype, int N_rank> \
class name ## _et : public StencilExpr<P_numtype,N_rank, \
    TinyVector<P_numtype,rank> >, \
  public ETBase<name ## _et<P_numtype,N_rank> > \
 { \
private: \
    typedef StencilExpr<P_numtype,N_rank,TinyVector<P_numtype,rank> > T_base; \
    using T_base::iter_; \
public: \
    typedef TinyVector<P_numtype,rank> result; \
    name ## _et(const Array<P_numtype,N_rank>& A) \
        : StencilExpr<P_numtype,N_rank,result>(A) \
    { } \
    result operator*() \
    { return name(iter_); } \
    result operator()(const TinyVector<int,N_rank>& a) \
    { iter_.moveTo(a); return name(iter_); } \
    result fastRead(int i) \
    { \
      const P_numtype* tmp = iter_.data(); \
      iter_._bz_setData(tmp + i); \
      P_numtype r = name(iter_); \
      iter_._bz_setData(tmp); \
      return r; \
    } \
}; \
template<class P_numtype, int N_rank> \
inline _bz_ArrayExpr<name ## _et<P_numtype, N_rank> > \
name(Array<P_numtype,N_rank>& A) \
{ \
    return _bz_ArrayExpr< name ## _et<P_numtype, N_rank> >(A); \
}

#define BZ_ET_STENCIL_DIFF(name) \
template<class P_numtype, int N_rank> \
class name ## _et : public StencilExpr<P_numtype,N_rank,P_numtype>, \
  public ETBase<name ## _et<P_numtype,N_rank> > \
 { \
private: \
    typedef StencilExpr<P_numtype,N_rank,P_numtype> T_base; \
    using T_base::iter_; \
public: \
    name ## _et(const Array<P_numtype,N_rank>& A, int dim) \
        : StencilExpr<P_numtype,N_rank,P_numtype>(A), dim_(dim) \
    { } \
    P_numtype operator*() \
    { return name(iter_); } \
    P_numtype operator()(const TinyVector<int,N_rank>& a) \
    { iter_.moveTo(a); return name(iter_,dim_); } \
    P_numtype fastRead(int i) \
    { \
      const P_numtype* tmp = iter_.data(); \
      iter_._bz_setData(tmp + i); \
      P_numtype r = name(iter_,dim_); \
      iter_._bz_setData(tmp); \
      return r; \
    } \
private: \
    int dim_; \
}; \
template<class P_numtype, int N_rank> \
inline _bz_ArrayExpr<name ## _et<P_numtype, N_rank> > \
name(Array<P_numtype,N_rank>& A, int dim) \
{ \
    return _bz_ArrayExpr<name ## _et<P_numtype, N_rank> >(A,dim); \
}


BZ_ET_STENCIL(Laplacian2D, P_numtype)
BZ_ET_STENCIL(Laplacian3D, P_numtype)
BZ_ET_STENCIL(Laplacian2D4, P_numtype)
BZ_ET_STENCIL(Laplacian2D4n, P_numtype)
BZ_ET_STENCIL(Laplacian3D4, P_numtype)
BZ_ET_STENCIL(Laplacian3D4n, P_numtype)
BZ_ET_STENCILV(grad2D, 2)
BZ_ET_STENCILV(grad2D4, 2)
BZ_ET_STENCILV(grad3D, 3)
BZ_ET_STENCILV(grad3D4, 3)
BZ_ET_STENCILV(grad2Dn, 2)
BZ_ET_STENCILV(grad2D4n, 2)
BZ_ET_STENCILV(grad3Dn, 3)
BZ_ET_STENCILV(grad3D4n, 3)
BZ_ET_STENCILV(gradSqr2D, 2)
BZ_ET_STENCILV(gradSqr2D4, 2)
BZ_ET_STENCILV(gradSqr3D, 3)
BZ_ET_STENCILV(gradSqr3D4, 3)
BZ_ET_STENCILV(gradSqr2Dn, 2)
BZ_ET_STENCILV(gradSqr2D4n, 2)
BZ_ET_STENCILV(gradSqr3Dn, 3)
BZ_ET_STENCILV(gradSqr3D4n, 3)

// NEEDS_WORK:
// Jacobian
// Curl
// Div
// mixed

BZ_ET_STENCIL_DIFF(central12)
BZ_ET_STENCIL_DIFF(central22)
BZ_ET_STENCIL_DIFF(central32)
BZ_ET_STENCIL_DIFF(central42)
BZ_ET_STENCIL_DIFF(central14)
BZ_ET_STENCIL_DIFF(central24)
BZ_ET_STENCIL_DIFF(central34)
BZ_ET_STENCIL_DIFF(central44)
BZ_ET_STENCIL_DIFF(central12n)
BZ_ET_STENCIL_DIFF(central22n)
BZ_ET_STENCIL_DIFF(central32n)
BZ_ET_STENCIL_DIFF(central42n)
BZ_ET_STENCIL_DIFF(central14n)
BZ_ET_STENCIL_DIFF(central24n)
BZ_ET_STENCIL_DIFF(central34n)
BZ_ET_STENCIL_DIFF(central44n)

BZ_ET_STENCIL_DIFF(backward11)
BZ_ET_STENCIL_DIFF(backward21)
BZ_ET_STENCIL_DIFF(backward31)
BZ_ET_STENCIL_DIFF(backward41)
BZ_ET_STENCIL_DIFF(backward12)
BZ_ET_STENCIL_DIFF(backward22)
BZ_ET_STENCIL_DIFF(backward32)
BZ_ET_STENCIL_DIFF(backward42)
BZ_ET_STENCIL_DIFF(backward11n)
BZ_ET_STENCIL_DIFF(backward21n)
BZ_ET_STENCIL_DIFF(backward31n)
BZ_ET_STENCIL_DIFF(backward41n)
BZ_ET_STENCIL_DIFF(backward12n)
BZ_ET_STENCIL_DIFF(backward22n)
BZ_ET_STENCIL_DIFF(backward32n)
BZ_ET_STENCIL_DIFF(backward42n)

BZ_ET_STENCIL_DIFF(forward11)
BZ_ET_STENCIL_DIFF(forward21)
BZ_ET_STENCIL_DIFF(forward31)
BZ_ET_STENCIL_DIFF(forward41)
BZ_ET_STENCIL_DIFF(forward12)
BZ_ET_STENCIL_DIFF(forward22)
BZ_ET_STENCIL_DIFF(forward32)
BZ_ET_STENCIL_DIFF(forward42)
BZ_ET_STENCIL_DIFF(forward11n)
BZ_ET_STENCIL_DIFF(forward21n)
BZ_ET_STENCIL_DIFF(forward31n)
BZ_ET_STENCIL_DIFF(forward41n)
BZ_ET_STENCIL_DIFF(forward12n)
BZ_ET_STENCIL_DIFF(forward22n)
BZ_ET_STENCIL_DIFF(forward32n)
BZ_ET_STENCIL_DIFF(forward42n)


BZ_NAMESPACE_END

#endif // BZ_ARRAY_STENCIL_ET_H
