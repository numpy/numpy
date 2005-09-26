/***************************************************************************
 * blitz/array/stencils.h  Stencils for arrays
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
 * Revision 1.1  2002/09/12 07:02:06  eric
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
 * Revision 1.2  2001/01/25 00:25:56  tveldhui
 * Ensured that source files have cvs logs.
 *
 */

#ifndef BZ_ARRAYSTENCILS_H
#define BZ_ARRAYSTENCILS_H

#ifndef BZ_ARRAY_H
 #error <blitz/array/stencils.h> must be included after <blitz/array.h>
#endif

#include <blitz/array/stencilops.h>

BZ_NAMESPACE(blitz)

// NEEDS_WORK: currently stencilExtent returns int(1).  What if the
// stencil contains calls to math functions, or divisions, etc.?
// Should at least return a number of the appropriate type.  Probably
// return a sequence of quasi-random floating point numbers.

/*
 * These macros make it easier for users to declare stencil objects.
 * The syntax is:
 *
 * BZ_DECLARE_STENCILN(stencilname, Array1, Array2, ..., ArrayN)
 *    // stencil operations go here
 * BZ_END_STENCIL
 */

#define BZ_DECLARE_STENCIL2(name,A,B)    \
  struct name {                          \
    template<class T1, class T2, class T3, class T4, class T5, class T6, \
        class T7, class T8, class T9, class T10, class T11>         \
    static inline void apply(T1& A, T2& B, T3, T4, T5, T6, T7, T8, T9, T10, T11) \
    {

#define BZ_END_STENCIL_WITH_SHAPE(MINS,MAXS) } \
    template<int N> \
    void getExtent(TinyVector<int,N>& minb, TinyVector<int,N>& maxb) const \
    { \
        minb = MINS; \
        maxb = MAXS; \
    } \
    enum { hasExtent = 1 }; \
};

#define BZ_END_STENCIL } enum { hasExtent = 0 }; };
#define BZ_STENCIL_END BZ_END_STENCIL

#define BZ_DECLARE_STENCIL3(name,A,B,C)         \
  struct name {                                 \
    template<class T1, class T2, class T3, class T4, class T5, class T6, \
        class T7, class T8, class T9, class T10, class T11>      \
    static inline void apply(T1& A, T2& B, T3& C, T4, T5, T6, T7, T8, T9,  \
        T10, T11)      \
    {

#define BZ_DECLARE_STENCIL4(name,A,B,C,D)             \
  struct name {                                       \
    template<class T1, class T2, class T3, class T4, class T5, class T6,  \
        class T7, class T8, class T9, class T10, class T11>  \
    static inline void apply(T1& A, T2& B, T3& C, T4& D, T5, T6, T7, \
        T8, T9, T10, T11)     \
    {

#define BZ_DECLARE_STENCIL5(name,A,B,C,D,E) \
  struct name { \
    template<class T1, class T2, class T3, class T4, class T5, class T6, \
        class T7, class T8, class T9, class T10, class T11> \
    static inline void apply(T1& A, T2& B, T3& C, T4& D, T5& E, T6, T7, T8, \
        T9, T10, T11) \
    {

#define BZ_DECLARE_STENCIL6(name,A,B,C,D,E,F) \
  struct name { \
    template<class T1, class T2, class T3, class T4, class T5, class T6, \
        class T7, class T8, class T9, class T10, class T11> \
    static inline void apply(T1& A, T2& B, T3& C, T4& D, T5& E, T6& F, \
        T7, T8, T9, T10, T11) \
    {

#define BZ_DECLARE_STENCIL7(name,A,B,C,D,E,F,G) \
  struct name { \
    template<class T1, class T2, class T3, class T4, \
      class T5, class T6, class T7, class T8, class T9, class T10, class T11> \
    static inline void apply(T1& A, T2& B, T3& C, T4& D, T5& E, T6& F, T7& G, \
        T8, T9, T10, T11) \
    {

#define BZ_DECLARE_STENCIL8(name,A,B,C,D,E,F,G,H) \
  struct name { \
    template<class T1, class T2, class T3, class T4, \
      class T5, class T6, class T7, class T8, class T9, class T10, class T11> \
    static inline void apply(T1& A, T2& B, T3& C, T4& D, T5& E, T6& F, T7& G, \
      T8& H, T9, T10, T11) \
    {

#define BZ_DECLARE_STENCIL9(name,A,B,C,D,E,F,G,H,I) \
  struct name { \
    template<class T1, class T2, class T3, class T4, \
      class T5, class T6, class T7, class T8, class T9, class T10, \
      class T11> \
    static inline void apply(T1& A, T2& B, T3& C, T4& D, T5& E, T6& F, T7& G, \
      T8& H, T9& I, T10, T11) \
    {

#define BZ_DECLARE_STENCIL10(name,A,B,C,D,E,F,G,H,I,J) \
  struct name { \
    template<class T1, class T2, class T3, class T4, \
      class T5, class T6, class T7, class T8, class T9, class T10, class T11> \
    static inline void apply(T1& A, T2& B, T3& C, T4& D, T5& E, T6& F, T7& G, \
      T8& H, T9& I, T10& J, T11) \
    {

#define BZ_DECLARE_STENCIL11(name,A,B,C,D,E,F,G,H,I,J,K) \
  struct name { \
    template<class T1, class T2, class T3, class T4, \
      class T5, class T6, class T7, class T8, class T9, class T10, \
      class T11> \
    static inline void apply(T1& A, T2& B, T3& C, T4& D, T5& E, T6& F, T7& G, \
      T8& H, T9& I, T10& J, T11& K) \
    {



/*
 * dummyArray is used to provide "dummy" padding parameters to applyStencil(),
 * so that any number of arrays (up to 11) can be given as arguments.
 */

template<class T> class dummy;

struct dummyArray {
    typedef dummy<double> T_iterator;

    const dummyArray& shape() const { return *this; }
};

_bz_global dummyArray _dummyArray;

/*
 * This dummy class pretends to be a scalar of type T, or an array iterator
 * of type T, but really does nothing.
 */
template<class T>
class dummy {
public:
    dummy() { }

    dummy(T value)
      : value_(value)
    { }

    dummy(const dummyArray&)
    { }

    operator T() const { return value_; };

    template<class T2>
    void operator=(T2) { }

    _bz_typename multicomponent_traits<T>::T_element operator[](int i) const
    { return value_[i]; }

    void loadStride(int) { }
    void moveTo(int) { }
    void moveTo(int,int) { }
    void moveTo(int,int,int) { }
    void moveTo(int,int,int,int) { }
    void advance() { }
    T shift(int,int) { return T(); }

private:
    T value_;
};


/*
 * The stencilExtent object is passed to stencil objects to find out
 * the spatial extent of the stencil.  It pretends it's an array,
 * but really it's just recording the locations of the array reads
 * via operator().
 */

template<int N_rank, class P_numtype>
class stencilExtent {
public:
    typedef P_numtype T_numtype;

    stencilExtent()
    {
        min_ = 0;
        max_ = 0;
    }
  
    dummy<T_numtype> operator()(int i)
    {
        update(0, i);
        return dummy<T_numtype>(1);
    }
 
    dummy<T_numtype> operator()(int i, int j)
    {
        update(0, i);
        update(1, j);
        return dummy<T_numtype>(1);
    }

    dummy<T_numtype> operator()(int i, int j, int k)
    {
        update(0, i);
        update(1, j);
        update(2, k);
        return dummy<T_numtype>(1);
    }

    dummy<T_numtype> shift(int offset, int dim)
    {
        update(dim, offset);
        return dummy<T_numtype>(1);
    }
  
    dummy<T_numtype> shift(int offset1, int dim1, int offset2, int dim2)
    {
        update(dim1, offset1);
        update(dim2, offset2);
        return dummy<T_numtype>(1);
    }
 
    dummy<_bz_typename multicomponent_traits<T_numtype>::T_element> 
        operator[](int)
    {
        return dummy<_bz_typename multicomponent_traits<T_numtype>::T_element>
            (1);
    }
 
    void update(int rank, int offset)
    {
        if (offset < min_[rank])
            min_[rank] = offset;
        if (offset > max_[rank])
            max_[rank] = offset;
    }

    template<class T_numtype2>
    void combine(const stencilExtent<N_rank,T_numtype2>& x)
    {
        for (int i=0; i < N_rank; ++i)
        {
            min_[i] = minmax::min(min_[i], x.min(i));
            max_[i] = minmax::max(max_[i], x.max(i));
        }
    }

    template<class T_numtype2>
    void combine(const dummy<T_numtype2>&)
    { }

    int min(int i) const
    { return min_[i]; }

    int max(int i) const
    { return max_[i]; }

    const TinyVector<int,N_rank>& min() const
    { return min_; }

    const TinyVector<int,N_rank>& max() const
    { return max_; }

    template<class T>
    void operator=(T)
    { }

    // NEEDS_WORK: other operators
    template<class T> void operator+=(T) { }
    template<class T> void operator-=(T) { }
    template<class T> void operator*=(T) { }
    template<class T> void operator/=(T) { }

    operator T_numtype()
    { return T_numtype(1); }

    T_numtype operator*()
    { return T_numtype(1); }
 
private:
    _bz_mutable TinyVector<int,N_rank> min_, max_;
};


/*
 * stencilExtent_traits gives a stencilExtent<N,T> object for arrays,
 * and a dummy object for dummy arrays.
 */
template<class T>
struct stencilExtent_traits {
    typedef dummy<double> T_stencilExtent;
};

template<class T_numtype, int N_rank>
struct stencilExtent_traits<Array<T_numtype,N_rank> > {
    typedef stencilExtent<N_rank,T_numtype> T_stencilExtent;
};

/*
 * Specialization of areShapesConformable(), originally
 * defined in <blitz/shapecheck.h>
 */

template<class T_shape1>
inline _bz_bool areShapesConformable(const T_shape1&, const dummyArray&)
{
    return _bz_true;
}

BZ_NAMESPACE_END

#include <blitz/array/stencils.cc>

#endif // BZ_ARRAYSTENCILS_H

