#ifndef BZ_ARRAYSTENCIL_H
#define BZ_ARRAYSTENCIL_H

#ifndef BZ_ARRAY_H
 #error <blitz/array/stencil.h> must be included via <blitz/array.h>
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

#include <blitz/array/stencil.cc>

#endif // BZ_ARRAYSTENCIL_H

