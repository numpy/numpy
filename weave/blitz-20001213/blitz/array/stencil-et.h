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
public: \
    name ## _et(const Array<T_numtype,N_rank>& A) \
        : StencilExpr<T_numtype,N_rank,result>(A) \
    { } \
    result operator*() \
    { return name(iter_); } \
    result operator()(const TinyVector<int,N_rank>& a) \
    { iter_.moveTo(a); return name(iter_); } \
    result fastRead(int i) \
    { \
      const T_numtype* tmp = iter_.data(); \
      iter_._bz_setData(tmp + i); \
      T_numtype r = name(iter_); \
      iter_._bz_setData(tmp); \
      return r; \
    } \
}; \
template<class T_numtype, int N_rank> \
inline _bz_ArrayExpr<name ## _et<T_numtype, N_rank> > \
name(Array<T_numtype,N_rank>& A) \
{ \
    return _bz_ArrayExpr<name ## _et<T_numtype, N_rank> >(A); \
}

#define BZ_ET_STENCILV(name,rank) \
template<class P_numtype, int N_rank> \
class name ## _et : public StencilExpr<P_numtype,N_rank, \
    TinyVector<P_numtype,rank> >, \
  public ETBase<name ## _et<P_numtype,N_rank> > \
 { \
public: \
    typedef TinyVector<T_numtype,rank> result; \
    name ## _et(const Array<T_numtype,N_rank>& A) \
        : StencilExpr<T_numtype,N_rank,result>(A) \
    { } \
    result operator*() \
    { return name(iter_); } \
    result operator()(const TinyVector<int,N_rank>& a) \
    { iter_.moveTo(a); return name(iter_); } \
    result fastRead(int i) \
    { \
      const T_numtype* tmp = iter_.data(); \
      iter_._bz_setData(tmp + i); \
      T_numtype r = name(iter_); \
      iter_._bz_setData(tmp); \
      return r; \
    } \
}; \
template<class T_numtype, int N_rank> \
inline _bz_ArrayExpr<name ## _et<T_numtype, N_rank> > \
name(Array<T_numtype,N_rank>& A) \
{ \
    return _bz_ArrayExpr< name ## _et<T_numtype, N_rank> >(A); \
}

#define BZ_ET_STENCIL_DIFF(name) \
template<class P_numtype, int N_rank> \
class name ## _et : public StencilExpr<P_numtype,N_rank,P_numtype>, \
  public ETBase<name ## _et<P_numtype,N_rank> > \
 { \
public: \
    name ## _et(const Array<T_numtype,N_rank>& A, int dim) \
        : StencilExpr<T_numtype,N_rank,P_numtype>(A), dim_(dim) \
    { } \
    T_numtype operator*() \
    { return name(iter_); } \
    T_numtype operator()(const TinyVector<int,N_rank>& a) \
    { iter_.moveTo(a); return name(iter_,dim_); } \
    T_numtype fastRead(int i) \
    { \
      const T_numtype* tmp = iter_.data(); \
      iter_._bz_setData(tmp + i); \
      T_numtype r = name(iter_,dim_); \
      iter_._bz_setData(tmp); \
      return r; \
    } \
private: \
    int dim_; \
}; \
template<class T_numtype, int N_rank> \
inline _bz_ArrayExpr<name ## _et<T_numtype, N_rank> > \
name(Array<T_numtype,N_rank>& A, int dim) \
{ \
    return _bz_ArrayExpr<name ## _et<T_numtype, N_rank> >(A,dim); \
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
