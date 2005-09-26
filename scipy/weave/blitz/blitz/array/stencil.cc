#ifndef BZ_ARRAYSTENCIL_CC
#define BZ_ARRAYSTENCIL_CC

#ifndef BZ_ARRAYSTENCIL_H
 #error <blitz/array/stencil.cc> must be included via <blitz/array/stencil.h>
#endif

BZ_NAMESPACE(blitz)

// NEEDS_WORK:
// o Need to allow scalar arguments as well as arrays
// o Unit stride optimization
// o Tiling
// o Pass coordinate vector to stencil, so that where-like constructs
//   can depend on location
// o Maybe allow expression templates to be passed as
//   array parameters?

/*
 * There are a lot of kludges in this code to work around the fact that
 * you can't have default template parameters with function templates.
 * Ideally, one would implement applyStencil(..) as:
 *
 * template<class T_stencil, class T_numtype1, class T_array2,
 *    class T_array3, class T_array4, class T_array5, class T_array6,
 *    class T_array7, class T_array8, class T_array9, class T_array10,
 *    class T_array11>
 * void applyStencil(const T_stencil& stencil, Array<T_numtype1,3>& A,
 *    T_array2& B = _dummyArray, T_array3& C = _dummyArray, ......)
 *
 * and allow for up to (say) 11 arrays to be passed.  But this doesn't
 * appear to be legal C++.  Instead, 11 versions of applyStencil are
 * provided, each one with a different number of array parameters,
 * and these stubs fill in the _dummyArray parameters and invoke
 * applyStencil_imp().
 */

template<int N_rank, class T_numtype1, class T_array2,
    class T_array3, class T_array4, class T_array5, class T_array6,
    class T_array7, class T_array8, class T_array9, class T_array10,
    class T_array11>
void checkShapes(const Array<T_numtype1,N_rank>& A,
    const T_array2& B, const T_array3& C, const T_array4& D, 
    const T_array5& E, const T_array6& F, const T_array7& G, 
    const T_array8& H, const T_array9& I, const T_array10& J, 
    const T_array11& K)
{
    BZPRECONDITION(areShapesConformable(A.shape(),B.shape())
        && areShapesConformable(A.shape(),C.shape())
        && areShapesConformable(A.shape(),D.shape())
        && areShapesConformable(A.shape(),E.shape())
        && areShapesConformable(A.shape(),F.shape())
        && areShapesConformable(A.shape(),G.shape())
        && areShapesConformable(A.shape(),H.shape())
        && areShapesConformable(A.shape(),I.shape())
        && areShapesConformable(A.shape(),J.shape())
        && areShapesConformable(A.shape(),K.shape()));
}

template<class T_extent, int N_rank, 
    class T_stencil, class T_numtype1, class T_array2,
    class T_array3, class T_array4, class T_array5, class T_array6,
    class T_array7, class T_array8, class T_array9, class T_array10,
    class T_array11>
void calcStencilExtent(T_extent& At, const T_stencil& stencil, 
    const Array<T_numtype1,N_rank>& A,
    const T_array2& B, const T_array3& C, const T_array4& D, const T_array5& E, 
    const T_array6& F, const T_array7& G, const T_array8& H, const T_array9& I, 
    const T_array10& J, const T_array11& K)
{
    // Interrogate the stencil to find out its extent
    _bz_typename stencilExtent_traits<T_array2>::T_stencilExtent Bt;
    _bz_typename stencilExtent_traits<T_array3>::T_stencilExtent Ct;
    _bz_typename stencilExtent_traits<T_array4>::T_stencilExtent Dt;
    _bz_typename stencilExtent_traits<T_array5>::T_stencilExtent Et;
    _bz_typename stencilExtent_traits<T_array6>::T_stencilExtent Ft;
    _bz_typename stencilExtent_traits<T_array7>::T_stencilExtent Gt;
    _bz_typename stencilExtent_traits<T_array8>::T_stencilExtent Ht;
    _bz_typename stencilExtent_traits<T_array9>::T_stencilExtent It;
    _bz_typename stencilExtent_traits<T_array10>::T_stencilExtent Jt;
    _bz_typename stencilExtent_traits<T_array11>::T_stencilExtent Kt;

    stencil.apply(At, Bt, Ct, Dt, Et, Ft, Gt, Ht, It, Jt, Kt);
    At.combine(Bt);
    At.combine(Ct);
    At.combine(Dt);
    At.combine(Et);
    At.combine(Ft);
    At.combine(Gt);
    At.combine(Ht);
    At.combine(It);
    At.combine(Jt);
    At.combine(Kt);
}

template<int N_rank, class T_stencil, class T_numtype1, class T_array2>
RectDomain<N_rank> interiorDomain(const T_stencil& stencil,
    const Array<T_numtype1,N_rank>& A,
    const T_array2& B)
{
    RectDomain<N_rank> domain = A.domain();

    // Interrogate the stencil to find out its extent
    stencilExtent<3, T_numtype1> At;
    calcStencilExtent(At, stencil, A, B, _dummyArray, _dummyArray, 
        _dummyArray, _dummyArray, _dummyArray, _dummyArray, _dummyArray, 
        _dummyArray, _dummyArray);

    // Shrink the domain according to the stencil size
    TinyVector<int,N_rank> lbound, ubound;
    lbound = domain.lbound() - At.min();
    ubound = domain.ubound() - At.max();
    return RectDomain<N_rank>(lbound,ubound);
}

template<int hasExtents>
struct _getStencilExtent {
template<int N_rank,
    class T_stencil, class T_numtype1, class T_array2,
    class T_array3, class T_array4, class T_array5, class T_array6,
    class T_array7, class T_array8, class T_array9, class T_array10,
    class T_array11>
static void getStencilExtent(TinyVector<int,N_rank>& minb,
    TinyVector<int,N_rank>& maxb,
    const T_stencil& stencil, Array<T_numtype1,N_rank>& A,
    T_array2& B, T_array3& C, T_array4& D, T_array5& E, T_array6& F,
    T_array7& G, T_array8& H, T_array9& I, T_array10& J, T_array11& K)
{
    // Interrogate the stencil to find out its extent
    stencilExtent<N_rank, T_numtype1> At;
    calcStencilExtent(At, stencil, A, B, C, D, E, F, G, H, I, J, K);
    minb = At.min();
    maxb = At.max();
}
};

template<>
struct _getStencilExtent<1> {
template<int N_rank,
    class T_stencil, class T_numtype1, class T_array2,
    class T_array3, class T_array4, class T_array5, class T_array6,
    class T_array7, class T_array8, class T_array9, class T_array10,
    class T_array11>
static inline void getStencilExtent(TinyVector<int,N_rank>& minb,
    TinyVector<int,N_rank>& maxb,
    const T_stencil& stencil, Array<T_numtype1,N_rank>& A,
    T_array2& B, T_array3& C, T_array4& D, T_array5& E, T_array6& F,
    T_array7& G, T_array8& H, T_array9& I, T_array10& J, T_array11& K)
{
    stencil.getExtent(minb, maxb);
}
};

template<int N_rank,
    class T_stencil, class T_numtype1, class T_array2,
    class T_array3, class T_array4, class T_array5, class T_array6,
    class T_array7, class T_array8, class T_array9, class T_array10,
    class T_array11>
inline void getStencilExtent(TinyVector<int,N_rank>& minb,
    TinyVector<int,N_rank>& maxb,
    const T_stencil& stencil, Array<T_numtype1,N_rank>& A,
    T_array2& B, T_array3& C, T_array4& D, T_array5& E, T_array6& F,
    T_array7& G, T_array8& H, T_array9& I, T_array10& J, T_array11& K)
{
    _getStencilExtent<T_stencil::hasExtent>::getStencilExtent(
        minb, maxb, stencil, A, B, C, D, E, F, G, H, I, J, K);
}

/*
 * This version applies a stencil to a set of 3D arrays.  Up to 11 arrays
 * may be used.  Any unused arrays are turned into dummyArray objects.
 * Operations on dummyArray objects are translated into no-ops.
 */
template<class T_stencil, class T_numtype1, class T_array2,
    class T_array3, class T_array4, class T_array5, class T_array6,
    class T_array7, class T_array8, class T_array9, class T_array10,
    class T_array11>
void applyStencil_imp(const T_stencil& stencil, Array<T_numtype1,3>& A,
    T_array2& B, T_array3& C, T_array4& D, T_array5& E, T_array6& F,
    T_array7& G, T_array8& H, T_array9& I, T_array10& J, T_array11& K)
{
    checkShapes(A,B,C,D,E,F,G,H,I,J,K);
 
    // Determine stencil extent
    TinyVector<int,3> minb, maxb;
    getStencilExtent(minb, maxb, stencil, A, B, C, D, E, F, G, H, I, J, K);

    // Now determine the subdomain over which the stencil
    // can be applied without worrying about overrunning the
    // boundaries of the array
    int stencil_lbound0 = minb(0);
    int stencil_lbound1 = minb(1);
    int stencil_lbound2 = minb(2);

    int stencil_ubound0 = maxb(0);
    int stencil_ubound1 = maxb(1);
    int stencil_ubound2 = maxb(2);

    int lbound0 = minmax::max(A.lbound(0), A.lbound(0) - stencil_lbound0);
    int lbound1 = minmax::max(A.lbound(1), A.lbound(1) - stencil_lbound1);
    int lbound2 = minmax::max(A.lbound(2), A.lbound(2) - stencil_lbound2);

    int ubound0 = minmax::min(A.ubound(0), A.ubound(0) - stencil_ubound0);
    int ubound1 = minmax::min(A.ubound(1), A.ubound(1) - stencil_ubound1);
    int ubound2 = minmax::min(A.ubound(2), A.ubound(2) - stencil_ubound2);

#if 0
    cout << "Stencil bounds are:" << endl
     << lbound0 << '\t' << ubound0 << endl
     << lbound1 << '\t' << ubound1 << endl
     << lbound2 << '\t' << ubound2 << endl;
#endif

    // Now do the actual loop
    FastArrayIterator<T_numtype1,3> Aiter(A);
    _bz_typename T_array2::T_iterator Biter(B);
    _bz_typename T_array3::T_iterator Citer(C);
    _bz_typename T_array4::T_iterator Diter(D);
    _bz_typename T_array5::T_iterator Eiter(E);
    _bz_typename T_array6::T_iterator Fiter(F);
    _bz_typename T_array7::T_iterator Giter(G);
    _bz_typename T_array8::T_iterator Hiter(H);
    _bz_typename T_array9::T_iterator Iiter(I);
    _bz_typename T_array10::T_iterator Jiter(J);
    _bz_typename T_array11::T_iterator Kiter(K);

    // Load the strides for the innermost loop
    Aiter.loadStride(2);
    Biter.loadStride(2);
    Citer.loadStride(2);
    Diter.loadStride(2);
    Eiter.loadStride(2);
    Fiter.loadStride(2);
    Giter.loadStride(2);
    Hiter.loadStride(2);
    Iiter.loadStride(2);
    Jiter.loadStride(2);
    Kiter.loadStride(2);

    for (int i=lbound0; i <= ubound0; ++i)
    {
      for (int j=lbound1; j <= ubound1; ++j)
      {
        Aiter.moveTo(i,j,lbound2);
        Biter.moveTo(i,j,lbound2);
        Citer.moveTo(i,j,lbound2);
        Diter.moveTo(i,j,lbound2);
        Eiter.moveTo(i,j,lbound2);
        Fiter.moveTo(i,j,lbound2);
        Giter.moveTo(i,j,lbound2);
        Hiter.moveTo(i,j,lbound2);
        Iiter.moveTo(i,j,lbound2);
        Jiter.moveTo(i,j,lbound2);
        Kiter.moveTo(i,j,lbound2);

        for (int k=lbound2; k <= ubound2; ++k)
        {
            stencil.apply(Aiter, Biter, Citer, Diter, Eiter, Fiter, Giter,
                Hiter, Iiter, Jiter, Kiter);

            Aiter.advance();
            Biter.advance();
            Citer.advance();
            Diter.advance();
            Eiter.advance();
            Fiter.advance();
            Giter.advance();
            Hiter.advance();
            Iiter.advance();
            Jiter.advance();
            Kiter.advance();
        }
      }
    }
}

/*
 * This version applies a stencil to a set of 2D arrays.  Up to 11 arrays
 * may be used.  Any unused arrays are turned into dummyArray objects.
 * Operations on dummyArray objects are translated into no-ops.
 */
template<class T_stencil, class T_numtype1, class T_array2,
    class T_array3, class T_array4, class T_array5, class T_array6,
    class T_array7, class T_array8, class T_array9, class T_array10,
    class T_array11>
void applyStencil_imp(const T_stencil& stencil, Array<T_numtype1,2>& A,
    T_array2& B, T_array3& C, T_array4& D, T_array5& E, T_array6& F, 
    T_array7& G, T_array8& H, T_array9& I, T_array10& J, T_array11& K)
{
    checkShapes(A,B,C,D,E,F,G,H,I,J,K);

    // Determine stencil extent
    TinyVector<int,2> minb, maxb;
    getStencilExtent(minb, maxb, stencil, A, B, C, D, E, F, G, H, I, J, K);

    // Now determine the subdomain over which the stencil
    // can be applied without worrying about overrunning the
    // boundaries of the array
    int stencil_lbound0 = minb(0);
    int stencil_lbound1 = minb(1);

    int stencil_ubound0 = maxb(0);
    int stencil_ubound1 = maxb(1);

    int lbound0 = minmax::max(A.lbound(0), A.lbound(0) - stencil_lbound0);
    int lbound1 = minmax::max(A.lbound(1), A.lbound(1) - stencil_lbound1);

    int ubound0 = minmax::min(A.ubound(0), A.ubound(0) - stencil_ubound0);
    int ubound1 = minmax::min(A.ubound(1), A.ubound(1) - stencil_ubound1);

#if 0
    cout << "Stencil bounds are:" << endl
     << lbound0 << '\t' << ubound0 << endl
     << lbound1 << '\t' << ubound1 << endl;
#endif 

    // Now do the actual loop
    FastArrayIterator<T_numtype1,2> Aiter(A);
    _bz_typename T_array2::T_iterator Biter(B);
    _bz_typename T_array3::T_iterator Citer(C);
    _bz_typename T_array4::T_iterator Diter(D);
    _bz_typename T_array5::T_iterator Eiter(E);
    _bz_typename T_array6::T_iterator Fiter(F);
    _bz_typename T_array7::T_iterator Giter(G);
    _bz_typename T_array8::T_iterator Hiter(H);
    _bz_typename T_array9::T_iterator Iiter(I);
    _bz_typename T_array10::T_iterator Jiter(J);
    _bz_typename T_array11::T_iterator Kiter(K);

    // Load the strides for the innermost loop
    Aiter.loadStride(1);
    Biter.loadStride(1);
    Citer.loadStride(1);
    Diter.loadStride(1);
    Eiter.loadStride(1);
    Fiter.loadStride(1);
    Giter.loadStride(1);
    Hiter.loadStride(1);
    Iiter.loadStride(1);
    Jiter.loadStride(1);
    Kiter.loadStride(1);

    for (int i=lbound0; i <= ubound0; ++i)
    {
        Aiter.moveTo(i,lbound1);
        Biter.moveTo(i,lbound1);
        Citer.moveTo(i,lbound1);
        Diter.moveTo(i,lbound1);
        Eiter.moveTo(i,lbound1);
        Fiter.moveTo(i,lbound1);
        Giter.moveTo(i,lbound1);
        Hiter.moveTo(i,lbound1);
        Iiter.moveTo(i,lbound1);
        Jiter.moveTo(i,lbound1);
        Kiter.moveTo(i,lbound1);

        for (int k=lbound1; k <= ubound1; ++k)
        {
            stencil.apply(Aiter, Biter, Citer, Diter, Eiter, Fiter, Giter,
                Hiter, Iiter, Jiter, Kiter);

            Aiter.advance();
            Biter.advance();
            Citer.advance();
            Diter.advance();
            Eiter.advance();
            Fiter.advance();
            Giter.advance();
            Hiter.advance();
            Iiter.advance();
            Jiter.advance();
            Kiter.advance();
        }
    }
}

/*
 * This version applies a stencil to a set of 1D arrays.  Up to 11 arrays
 * may be used.  Any unused arrays are turned into dummyArray objects.
 * Operations on dummyArray objects are translated into no-ops.
 */
template<class T_stencil, class T_numtype1, class T_array2,
    class T_array3, class T_array4, class T_array5, class T_array6,
    class T_array7, class T_array8, class T_array9, class T_array10,
    class T_array11>
void applyStencil_imp(const T_stencil& stencil, Array<T_numtype1,1>& A,
    T_array2& B, T_array3& C, T_array4& D, T_array5& E, T_array6& F, 
    T_array7& G, T_array8& H, T_array9& I, T_array10& J, T_array11& K)
{
    checkShapes(A,B,C,D,E,F,G,H,I,J,K);

    // Determine stencil extent
    TinyVector<int,1> minb, maxb;
    getStencilExtent(minb, maxb, stencil, A, B, C, D, E, F, G, H, I, J, K);

    // Now determine the subdomain over which the stencil
    // can be applied without worrying about overrunning the
    // boundaries of the array
    int stencil_lbound0 = minb(0);
    int stencil_ubound0 = maxb(0);

    int lbound0 = minmax::max(A.lbound(0), A.lbound(0) - stencil_lbound0);
    int ubound0 = minmax::min(A.ubound(0), A.ubound(0) - stencil_ubound0);

#if 0
    cout << "Stencil bounds are:" << endl
     << lbound0 << '\t' << ubound0 << endl;
#endif

    // Now do the actual loop
    FastArrayIterator<T_numtype1,1> Aiter(A);
    _bz_typename T_array2::T_iterator Biter(B);
    _bz_typename T_array3::T_iterator Citer(C);
    _bz_typename T_array4::T_iterator Diter(D);
    _bz_typename T_array5::T_iterator Eiter(E);
    _bz_typename T_array6::T_iterator Fiter(F);
    _bz_typename T_array7::T_iterator Giter(G);
    _bz_typename T_array8::T_iterator Hiter(H);
    _bz_typename T_array9::T_iterator Iiter(I);
    _bz_typename T_array10::T_iterator Jiter(J);
    _bz_typename T_array11::T_iterator Kiter(K);

    // Load the strides for the innermost loop
    Aiter.loadStride(0);
    Biter.loadStride(0);
    Citer.loadStride(0);
    Diter.loadStride(0);
    Eiter.loadStride(0);
    Fiter.loadStride(0);
    Giter.loadStride(0);
    Hiter.loadStride(0);
    Iiter.loadStride(0);
    Jiter.loadStride(0);
    Kiter.loadStride(0);

    Aiter.moveTo(lbound0);
    Biter.moveTo(lbound0);
    Citer.moveTo(lbound0);
    Diter.moveTo(lbound0);
    Eiter.moveTo(lbound0);
    Fiter.moveTo(lbound0);
    Giter.moveTo(lbound0);
    Hiter.moveTo(lbound0);
    Iiter.moveTo(lbound0);
    Jiter.moveTo(lbound0);
    Kiter.moveTo(lbound0);

    for (int i=lbound0; i <= ubound0; ++i)
    {
        stencil.apply(Aiter, Biter, Citer, Diter, Eiter, Fiter, Giter,
            Hiter, Iiter, Jiter, Kiter);

        Aiter.advance();
        Biter.advance();
        Citer.advance();
        Diter.advance();
        Eiter.advance();
        Fiter.advance();
        Giter.advance();
        Hiter.advance();
        Iiter.advance();
        Jiter.advance();
        Kiter.advance();
    }
}

/*
 * These 11 versions of applyStencil handle from 1 to 11 array parameters.
 * They pad their argument list with enough dummyArray objects to call
 * applyStencil_imp with 11 array parameters.
 */
template<class T_stencil, class T_numtype1, int N_rank>
inline void applyStencil(const T_stencil& stencil, Array<T_numtype1,N_rank>& A)
{
    applyStencil_imp(stencil, A, _dummyArray, _dummyArray,
        _dummyArray, _dummyArray, _dummyArray, _dummyArray,
        _dummyArray, _dummyArray, _dummyArray, _dummyArray);
}

template<class T_stencil, class T_numtype1, int N_rank, class T_array2>
inline void applyStencil(const T_stencil& stencil, Array<T_numtype1,N_rank>& A,
    T_array2& B)
{
    applyStencil_imp(stencil, A, B, _dummyArray, _dummyArray,
        _dummyArray, _dummyArray, _dummyArray, _dummyArray,
        _dummyArray, _dummyArray, _dummyArray);
}

template<class T_stencil, class T_numtype1, int N_rank, class T_array2,
    class T_array3>
inline void applyStencil(const T_stencil& stencil, Array<T_numtype1,N_rank>& A,
    T_array2& B, T_array3& C)
{
    applyStencil_imp(stencil, A, B, C, _dummyArray, _dummyArray,
        _dummyArray, _dummyArray, _dummyArray, _dummyArray, _dummyArray,
        _dummyArray);
}

template<class T_stencil, class T_numtype1, int N_rank, class T_array2,
    class T_array3, class T_array4>
inline void applyStencil(const T_stencil& stencil, Array<T_numtype1,N_rank>& A,
    T_array2& B, T_array3& C, T_array4& D)
{
    applyStencil_imp(stencil, A, B, C, D, _dummyArray, _dummyArray,
        _dummyArray, _dummyArray, _dummyArray, _dummyArray, _dummyArray);
}

template<class T_stencil, class T_numtype1, int N_rank, class T_array2,
    class T_array3, class T_array4, class T_array5>
inline void applyStencil(const T_stencil& stencil, Array<T_numtype1,N_rank>& A,
   T_array2& B, T_array3& C, T_array4& D, T_array5& E)
{
    applyStencil_imp(stencil, A, B, C, D, E, _dummyArray,
        _dummyArray, _dummyArray, _dummyArray, _dummyArray, _dummyArray);
}

template<class T_stencil, class T_numtype1, int N_rank, class T_array2,
    class T_array3, class T_array4, class T_array5, class T_array6>
inline void applyStencil(const T_stencil& stencil, Array<T_numtype1,N_rank>& A,
   T_array2& B, T_array3& C, T_array4& D, T_array5& E, T_array6& F)
{
    applyStencil_imp(stencil, A, B, C, D, E, F,
        _dummyArray, _dummyArray, _dummyArray, _dummyArray, _dummyArray);
}

template<class T_stencil, class T_numtype1, int N_rank, class T_array2,
    class T_array3, class T_array4, class T_array5, class T_array6,
    class T_array7>
inline void applyStencil(const T_stencil& stencil, Array<T_numtype1,N_rank>& A,
   T_array2& B, T_array3& C, T_array4& D, T_array5& E, T_array6& F,
   T_array7& G)
{
    applyStencil_imp(stencil, A, B, C, D, E, F, G,
        _dummyArray, _dummyArray, _dummyArray, _dummyArray);
}

template<class T_stencil, class T_numtype1, int N_rank, class T_array2,
    class T_array3, class T_array4, class T_array5, class T_array6,
    class T_array7, class T_array8>
inline void applyStencil(const T_stencil& stencil, Array<T_numtype1,N_rank>& A,
   T_array2& B, T_array3& C, T_array4& D, T_array5& E, T_array6& F,
   T_array7& G, T_array8& H)
{
    applyStencil_imp(stencil, A, B, C, D, E, F, G, H,
        _dummyArray, _dummyArray, _dummyArray);
}

template<class T_stencil, class T_numtype1, int N_rank, class T_array2,
    class T_array3, class T_array4, class T_array5, class T_array6,
    class T_array7, class T_array8, class T_array9>
inline void applyStencil(const T_stencil& stencil, Array<T_numtype1,N_rank>& A,
   T_array2& B, T_array3& C, T_array4& D, T_array5& E, T_array6& F,
   T_array7& G, T_array8& H, T_array9& I)
{
    applyStencil_imp(stencil, A, B, C, D, E, F, G, H, I,
        _dummyArray, _dummyArray);
}

template<class T_stencil, class T_numtype1, int N_rank, class T_array2,
    class T_array3, class T_array4, class T_array5, class T_array6,
    class T_array7, class T_array8, class T_array9, class T_array10>
inline void applyStencil(const T_stencil& stencil, Array<T_numtype1,N_rank>& A,
   T_array2& B, T_array3& C, T_array4& D, T_array5& E, T_array6& F,
   T_array7& G, T_array8& H, T_array9& I, T_array10& J)
{
    applyStencil_imp(stencil, A, B, C, D, E, F, G, H, I, J,
        _dummyArray);
}

template<class T_stencil, class T_numtype1, int N_rank, class T_array2,
    class T_array3, class T_array4, class T_array5, class T_array6,
    class T_array7, class T_array8, class T_array9, class T_array10,
    class T_array11>
inline void applyStencil(const T_stencil& stencil, Array<T_numtype1,N_rank>& A,
   T_array2& B, T_array3& C, T_array4& D, T_array5& E, T_array6& F,
   T_array7& G, T_array8& H, T_array9& I, T_array10& J, T_array11& K)
{
    applyStencil_imp(stencil, A, B, C, D, E, F, G, H, I, J, K);
}

BZ_NAMESPACE_END

#endif // BZ_ARRAYSTENCIL_CC
