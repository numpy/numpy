#ifndef BZ_ARRAYSTENCILOPS_H
#define BZ_ARRAYSTENCILOPS_H

// NEEDS_WORK: need to factor many of the stencils in terms of the
// integer constants, e.g. 16*(A(-1,0)+A(0,-1)+A(0,1)+A(1,0))

#ifndef BZ_ARRAYSTENCIL_H
 #error <blitz/array/stencilops.h> must be included via <blitz/array/stencil.h>
#endif

#ifndef BZ_GEOMETRY_H
 #include <blitz/array/geometry.h>
#endif

#ifndef BZ_TINYMAT_H
 #include <blitz/tinymat.h>
#endif

BZ_NAMESPACE(blitz)

#define BZ_DECLARE_STENCIL_OPERATOR1(name,A)     \
  template<class T>                              \
  inline _bz_typename T::T_numtype name(T& A)    \
  {

#define BZ_END_STENCIL_OPERATOR   }

#define BZ_DECLARE_STENCIL_OPERATOR2(name,A,B)       \
  template<class T>                                  \
  inline _bz_typename T::T_numtype name(T& A, T& B)  \
  {

#define BZ_DECLARE_STENCIL_OPERATOR3(name,A,B,C) \
  template<class T>                              \
  inline _bz_typename T::T_numtype name(T& A, T& B, T& C)    \
  {

// These constants are accurate to 45 decimal places = 149 bits of mantissa
const double recip_2 = .500000000000000000000000000000000000000000000;
const double recip_4 = .250000000000000000000000000000000000000000000;
const double recip_6 = .166666666666666666666666666666666666666666667;
const double recip_8 = .125000000000000000000000000000000000000000000;
const double recip_12 = .0833333333333333333333333333333333333333333333;
const double recip_144 = .00694444444444444444444444444444444444444444444;

/****************************************************************************
 * Laplacian Operators
 ****************************************************************************/

BZ_DECLARE_STENCIL_OPERATOR1(Laplacian2D, A)
    return -4.0 * A + A.shift(-1,0) + A.shift(1,0) + A.shift(-1,1)
      + A.shift(1,1);
BZ_END_STENCIL_OPERATOR

BZ_DECLARE_STENCIL_OPERATOR1(Laplacian3D, A)
    return -6.0 * A 
      + A.shift(-1,0) + A.shift(1,0) 
      + A.shift(-1,1) + A.shift(1,1)
      + A.shift(-1,2) + A.shift(1,2);
BZ_END_STENCIL_OPERATOR

BZ_DECLARE_STENCIL_OPERATOR1(Laplacian2D4, A)
    return -60. * A 
      + 16.*(A.shift(-1,0) + A.shift(1,0) + A.shift(-1,1) + A.shift(1,1))
      -     (A.shift(-2,0) + A.shift(2,0) + A.shift(-2,1) + A.shift(2,1));
BZ_END_STENCIL_OPERATOR

BZ_DECLARE_STENCIL_OPERATOR1(Laplacian2D4n, A)
    return Laplacian2D4(A) * recip_12;
BZ_END_STENCIL_OPERATOR

BZ_DECLARE_STENCIL_OPERATOR1(Laplacian3D4, A)
    return -90. * A 
      + 16.*(A.shift(-1,0) + A.shift(1,0) + A.shift(-1,1) + A.shift(1,1)
           + A.shift(-1,2) + A.shift(1,2))
      -     (A.shift(-2,0) + A.shift(2,0) + A.shift(-2,1) + A.shift(2,1)
           + A.shift(-2,2) + A.shift(2,2));
BZ_END_STENCIL_OPERATOR

BZ_DECLARE_STENCIL_OPERATOR1(Laplacian3D4n, A)
    return Laplacian3D4(A) * recip_12;
BZ_END_STENCIL_OPERATOR

/****************************************************************************
 * Derivatives
 ****************************************************************************/

#define BZ_DECLARE_DIFF(name)  \
  template<class T> \
  inline _bz_typename T::T_numtype name(T& A, int dim = firstDim)

#define BZ_DECLARE_MULTIDIFF(name) \
  template<class T> \
  inline _bz_typename multicomponent_traits<_bz_typename     \
     T::T_numtype>::T_element name(T& A, int comp, int dim)

/****************************************************************************
 * Central differences with accuracy O(h^2)
 ****************************************************************************/

BZ_DECLARE_DIFF(central12) {
  return A.shift(1,dim) - A.shift(-1,dim);
}

BZ_DECLARE_DIFF(central22) {
  return A.shift(-1,dim) - 2. * A + A.shift(+1,dim);
}

BZ_DECLARE_DIFF(central32) {
  return -A.shift(-2,dim) + 2.*(A.shift(-1,dim) - A.shift(+1,dim))
    + A.shift(+2,dim);
}

BZ_DECLARE_DIFF(central42) {
  return A.shift(-2,dim) + A.shift(2,dim) -4.*(A.shift(-1,dim)+A.shift(+1,dim)) 
    +6.*A.shift(0,dim);
}

/****************************************************************************
 * Central differences with accuracy O(h^2)  (multicomponent versions)
 ****************************************************************************/

BZ_DECLARE_MULTIDIFF(central12) {
  return A.shift(1,dim)[comp] - A.shift(-1,dim)[comp];
}

BZ_DECLARE_MULTIDIFF(central22) {
  return A.shift(-1,dim)[comp] - 2. * (*A)[comp] + A.shift(+1,dim)[comp];
}

BZ_DECLARE_MULTIDIFF(central32) {
  return -A.shift(-2,dim)[comp] + 2.*A.shift(-1,dim)[comp] 
    -2.*A.shift(+1,dim)[comp] + A.shift(+2,dim)[comp];
}

BZ_DECLARE_MULTIDIFF(central42) {
  return A.shift(-2,dim)[comp] -4.*A.shift(-1,dim)[comp] 
    +6.*A.shift(0,dim)[comp] -4.*A.shift(1,dim)[comp] +A.shift(2,dim)[comp];
}

/****************************************************************************
 * Central differences with accuracy O(h^2)  (normalized versions)
 ****************************************************************************/

BZ_DECLARE_DIFF(central12n) {
  return central12(A,dim) * recip_2;
}

BZ_DECLARE_DIFF(central22n) {
  return central22(A,dim);
}

BZ_DECLARE_DIFF(central32n) {
  return central32(A,dim) * recip_2;
}

BZ_DECLARE_DIFF(central42n) {
  return central42(A,dim);
}

/****************************************************************************
 * Central differences with accuracy O(h^2)  (normalized multicomponent)
 ****************************************************************************/

BZ_DECLARE_MULTIDIFF(central12n) {
  return central12(A,comp,dim) * recip_2;
}

BZ_DECLARE_MULTIDIFF(central22n) {
  return central22(A,comp,dim);
}

BZ_DECLARE_MULTIDIFF(central32n) {
  return central32(A,comp,dim) * recip_2;
}

BZ_DECLARE_MULTIDIFF(central42n) {
  return central42(A,comp,dim);
}

/****************************************************************************
 * Central differences with accuracy O(h^4)  
 ****************************************************************************/

BZ_DECLARE_DIFF(central14) {
  return (A.shift(-2,dim) - A.shift(2,dim)) 
     + 8.*(A.shift(1,dim)-A.shift(-1,dim));
}

BZ_DECLARE_DIFF(central24) {
  return -30.*A + 16.*(A.shift(-1,dim)+A.shift(1,dim))
    - (A.shift(-2,dim)+A.shift(2,dim));
}

BZ_DECLARE_DIFF(central34) {
  return A.shift(-3,dim) - 8.*A.shift(-2,dim) +13.*A.shift(-1,dim) 
     -13.*A.shift(1,dim)+8.*A.shift(2,dim)-A.shift(3,dim);
}

BZ_DECLARE_DIFF(central44) {
  return -1.*A.shift(-3,dim)+12.*A.shift(-2,dim)-39.*A.shift(-1,dim)
    +56.*A-39.*A.shift(1,dim)+12.*A.shift(2,dim)-A.shift(3,dim);
}

/****************************************************************************
 * Central differences with accuracy O(h^4)  (multicomponent versions)
 ****************************************************************************/

BZ_DECLARE_MULTIDIFF(central14) {
  return A.shift(-2,dim)[comp] - 8. * A.shift(-1,dim)[comp] 
    + 8. * A.shift(1,dim)[comp] - A.shift(2,dim)[comp];
}

BZ_DECLARE_MULTIDIFF(central24) {
  return - A.shift(-2,dim)[comp] + 16.*A.shift(-1,dim)[comp] - 30.*(*A)[comp]
    + 16.*A.shift(1,dim)[comp] - A.shift(2,dim)[comp];
}

BZ_DECLARE_MULTIDIFF(central34) {
  return A.shift(-3,dim)[comp] - 8.*A.shift(-2,dim)[comp] 
    +13.*A.shift(-1,dim)[comp] - 13.*A.shift(1,dim)[comp] 
    + 8.*A.shift(2,dim)[comp] - A.shift(3,dim)[comp];
}

BZ_DECLARE_MULTIDIFF(central44) {
  return -1.*A.shift(-3,dim)[comp]+12.*A.shift(-2,dim)[comp]
    -39.*A.shift(-1,dim)[comp] +56.*(*A)[comp]-39.*A.shift(1,dim)[comp]
    +12.*A.shift(2,dim)[comp]-A.shift(3,dim)[comp];
}

/****************************************************************************
 * Central differences with accuracy O(h^4)  (normalized)
 ****************************************************************************/

BZ_DECLARE_DIFF(central14n) {
  return central14(A,dim) * recip_12;
}

BZ_DECLARE_DIFF(central24n) {
  return central24(A,dim) * recip_12;
}

BZ_DECLARE_DIFF(central34n) {
  return central34(A,dim) * recip_8;
}

BZ_DECLARE_DIFF(central44n) {
  return central44(A,dim) * recip_6;
}

/****************************************************************************
 * Central differences with accuracy O(h^4)  (normalized, multicomponent)
 ****************************************************************************/

BZ_DECLARE_MULTIDIFF(central14n) {
  return central14(A,comp,dim) * recip_12;
}

BZ_DECLARE_MULTIDIFF(central24n) {
  return central24(A,comp,dim) * recip_12;
}

BZ_DECLARE_MULTIDIFF(central34n) {
  return central34(A,comp,dim) * recip_8;
}

BZ_DECLARE_MULTIDIFF(central44n) {
  return central44(A,comp,dim) * recip_6;
}

/****************************************************************************
 * Backward differences with accuracy O(h)
 ****************************************************************************/

BZ_DECLARE_DIFF(backward11) {
  return A - A.shift(-1,dim);
}

BZ_DECLARE_DIFF(backward21) {
  return A -2.*A.shift(-1,dim) + A.shift(-2,dim);
}

BZ_DECLARE_DIFF(backward31) {
  return A -3.*A.shift(-1,dim) + 3.*A.shift(-2,dim)-A.shift(-3,dim);
}

BZ_DECLARE_DIFF(backward41) {
  return A - 4.*A.shift(-1,dim) + 6.*A.shift(-2,dim) -4.*A.shift(-3,dim) 
    + A.shift(-4,dim);
}

/****************************************************************************
 * Backward differences with accuracy O(h) (multicomponent versions)
 ****************************************************************************/

BZ_DECLARE_MULTIDIFF(backward11) {
  return (*A)[comp] - A.shift(-1,dim)[comp];
}

BZ_DECLARE_MULTIDIFF(backward21) {
  return (*A)[comp] -2.*A.shift(-1,dim)[comp] + A.shift(-2,dim)[comp];
}

BZ_DECLARE_MULTIDIFF(backward31) {
  return (*A)[comp] -3.*A.shift(-1,dim)[comp] + 3.*A.shift(-2,dim)[comp]
    -A.shift(-3,dim)[comp];
}

BZ_DECLARE_MULTIDIFF(backward41) {
  return (*A)[comp] - 4.*A.shift(-1,dim)[comp] + 6.*A.shift(-2,dim)[comp]
    -4.*A.shift(-3,dim)[comp] + A.shift(-4,dim)[comp];
}

/****************************************************************************
 * Backward differences with accuracy O(h)  (normalized)
 ****************************************************************************/

BZ_DECLARE_DIFF(backward11n) { return backward11(A,dim); }
BZ_DECLARE_DIFF(backward21n) { return backward21(A,dim); }
BZ_DECLARE_DIFF(backward31n) { return backward31(A,dim); }
BZ_DECLARE_DIFF(backward41n) { return backward41(A,dim); }

/****************************************************************************
 * Backward differences with accuracy O(h)  (normalized, multicomponent)
 ****************************************************************************/

BZ_DECLARE_MULTIDIFF(backward11n) { return backward11(A,comp,dim); }
BZ_DECLARE_MULTIDIFF(backward21n) { return backward21(A,comp,dim); }
BZ_DECLARE_MULTIDIFF(backward31n) { return backward31(A,comp,dim); }
BZ_DECLARE_MULTIDIFF(backward41n) { return backward41(A,comp,dim); }

/****************************************************************************
 * Backward differences with accuracy O(h^2)
 ****************************************************************************/

BZ_DECLARE_DIFF(backward12) {
  return 3.*A -4.*A.shift(-1,dim) + A.shift(-2,dim);
}

BZ_DECLARE_DIFF(backward22) {
  return 2.*A -5.*A.shift(-1,dim) + 4.*A.shift(-2,dim) -A.shift(-3,dim);
}

BZ_DECLARE_DIFF(backward32) {
  return 5.*A - 18.*A.shift(-1,dim) + 24.*A.shift(-2,dim) -14.*A.shift(-3,dim) 
    + 3.*A.shift(-4,dim);
}

BZ_DECLARE_DIFF(backward42) {
  return 3.*A -14.*A.shift(-1,dim) + 26.*A.shift(-2,dim) -24.*A.shift(-3,dim) 
    + 11.*A.shift(-4,dim) -2.*A.shift(-5,dim);
}

/****************************************************************************
 * Backward differences with accuracy O(h^2) (multicomponent versions)
 ****************************************************************************/

BZ_DECLARE_MULTIDIFF(backward12) {
  return 3.*(*A)[comp] -4.*A.shift(-1,dim)[comp] + A.shift(-2,dim)[comp];
}

BZ_DECLARE_MULTIDIFF(backward22) {
  return 2.*(*A)[comp] -5.*A.shift(-1,dim)[comp] + 4.*A.shift(-2,dim)[comp]
     -A.shift(-3,dim)[comp];
}

BZ_DECLARE_MULTIDIFF(backward32) {
  return 5.*(*A)[comp] - 18.*A.shift(-1,dim)[comp] + 24.*A.shift(-2,dim)[comp]
     -14.*A.shift(-3,dim)[comp] + 3.*A.shift(-4,dim)[comp];
}

BZ_DECLARE_MULTIDIFF(backward42) {
  return 3.*(*A)[comp] -14.*A.shift(-1,dim)[comp] + 26.*A.shift(-2,dim)[comp]
    -24.*A.shift(-3,dim)[comp] + 11.*A.shift(-4,dim)[comp] 
    -2.*A.shift(-5,dim)[comp];
}

/****************************************************************************
 * Backward differences with accuracy O(h^2)  (normalized)
 ****************************************************************************/

BZ_DECLARE_DIFF(backward12n) { return backward12(A,dim) * recip_2; }
BZ_DECLARE_DIFF(backward22n) { return backward22(A,dim); }
BZ_DECLARE_DIFF(backward32n) { return backward32(A,dim) * recip_2; }
BZ_DECLARE_DIFF(backward42n) { return backward42(A,dim); }

/****************************************************************************
 * Backward differences with accuracy O(h^2)  (normalized, multicomponent)
 ****************************************************************************/

BZ_DECLARE_MULTIDIFF(backward12n) { return backward12(A,comp,dim) * recip_2; }
BZ_DECLARE_MULTIDIFF(backward22n) { return backward22(A,comp,dim); }
BZ_DECLARE_MULTIDIFF(backward32n) { return backward32(A,comp,dim) * recip_2; }
BZ_DECLARE_MULTIDIFF(backward42n) { return backward42(A,comp,dim); }

/****************************************************************************
 * Forward differences with accuracy O(h)  
 ****************************************************************************/

BZ_DECLARE_DIFF(forward11) {
  return A.shift(1,dim) - A;
}

BZ_DECLARE_DIFF(forward21) {
  return A - 2.*A.shift(1,dim) + A.shift(2,dim);
}

BZ_DECLARE_DIFF(forward31) {
  return -A + 3.*A.shift(1,dim) -3.*A.shift(2,dim) + A.shift(3,dim);
}

BZ_DECLARE_DIFF(forward41) {
  return A -4.*A.shift(1,dim) + 6.*A.shift(2,dim) -4.*A.shift(3,dim) 
    + A.shift(4,dim);
}

/****************************************************************************
 * Forward differences with accuracy O(h)  (multicomponent versions)
 ****************************************************************************/

BZ_DECLARE_MULTIDIFF(forward11) {
  return -(*A)[comp]+A.shift(1,dim)[comp];
}

BZ_DECLARE_MULTIDIFF(forward21) {
  return (*A)[comp] - 2.*A.shift(1,dim)[comp] + A.shift(2,dim)[comp];
}

BZ_DECLARE_MULTIDIFF(forward31) {
  return -(*A)[comp] + 3.*A.shift(1,dim)[comp] -3.*A.shift(2,dim)[comp] 
    + A.shift(3,dim)[comp];
}

BZ_DECLARE_MULTIDIFF(forward41) {
  return (*A)[comp] -4.*A.shift(1,dim)[comp] + 6.*A.shift(2,dim)[comp] 
    -4.*A.shift(3,dim)[comp] + A.shift(4,dim)[comp];
}

/****************************************************************************
 * Forward differences with accuracy O(h)     (normalized)
 ****************************************************************************/

BZ_DECLARE_DIFF(forward11n) { return forward11(A,dim); }
BZ_DECLARE_DIFF(forward21n) { return forward21(A,dim); }
BZ_DECLARE_DIFF(forward31n) { return forward31(A,dim); }
BZ_DECLARE_DIFF(forward41n) { return forward41(A,dim); }

/****************************************************************************
 * Forward differences with accuracy O(h)     (multicomponent,normalized)
 ****************************************************************************/

BZ_DECLARE_MULTIDIFF(forward11n) { return forward11(A,comp,dim); }
BZ_DECLARE_MULTIDIFF(forward21n) { return forward21(A,comp,dim); }
BZ_DECLARE_MULTIDIFF(forward31n) { return forward31(A,comp,dim); }
BZ_DECLARE_MULTIDIFF(forward41n) { return forward41(A,comp,dim); }

/****************************************************************************
 * Forward differences with accuracy O(h^2)     
 ****************************************************************************/

BZ_DECLARE_DIFF(forward12) {
  return -3.*A + 4.*A.shift(1,dim) - A.shift(2,dim);
}

BZ_DECLARE_DIFF(forward22) {
  return 2.*A -5.*A.shift(1,dim) + 4.*A.shift(2,dim) -A.shift(3,dim);
}

BZ_DECLARE_DIFF(forward32) {
  return -5.*A + 18.*A.shift(1,dim) -24.*A.shift(2,dim) 
    + 14.*A.shift(3,dim) -3.*A.shift(4,dim);
}

BZ_DECLARE_DIFF(forward42) {
  return 3.*A -14.*A.shift(1,dim) + 26.*A.shift(2,dim) -24.*A.shift(3,dim) 
    +11.*A.shift(4,dim) -2.*A.shift(5,dim);
}

/****************************************************************************
 * Forward differences with accuracy O(h^2)   (multicomponent versions)
 ****************************************************************************/

BZ_DECLARE_MULTIDIFF(forward12) {
  return -3.*(*A)[comp] + 4.*A.shift(1,dim)[comp] - A.shift(2,dim)[comp];
}

BZ_DECLARE_MULTIDIFF(forward22) {
  return 2.*(*A)[comp] -5.*A.shift(1,dim)[comp] + 4.*A.shift(2,dim)[comp] 
    -A.shift(3,dim)[comp];
}

BZ_DECLARE_MULTIDIFF(forward32) {
  return -5.*(*A)[comp] + 18.*A.shift(1,dim)[comp] -24.*A.shift(2,dim)[comp]
    + 14.*A.shift(3,dim)[comp] -3.*A.shift(4,dim)[comp];
}

BZ_DECLARE_MULTIDIFF(forward42) {
  return 3.*(*A)[comp] -14.*A.shift(1,dim)[comp] + 26.*A.shift(2,dim)[comp] 
    -24.*A.shift(3,dim)[comp] +11.*A.shift(4,dim)[comp] 
    + 11.*A.shift(5,dim)[comp];
}


/****************************************************************************
 * Forward differences with accuracy O(h^2)     (normalized)
 ****************************************************************************/

BZ_DECLARE_DIFF(forward12n) { return forward12(A,dim) * recip_2; }
BZ_DECLARE_DIFF(forward22n) { return forward22(A,dim); }
BZ_DECLARE_DIFF(forward32n) { return forward32(A,dim) * recip_2; }
BZ_DECLARE_DIFF(forward42n) { return forward42(A,dim); }

/****************************************************************************
 * Forward differences with accuracy O(h^2)     (normalized)
 ****************************************************************************/

BZ_DECLARE_MULTIDIFF(forward12n) { return forward12(A,comp,dim) * recip_2; }
BZ_DECLARE_MULTIDIFF(forward22n) { return forward22(A,comp,dim); }
BZ_DECLARE_MULTIDIFF(forward32n) { return forward32(A,comp,dim) * recip_2; }
BZ_DECLARE_MULTIDIFF(forward42n) { return forward42(A,comp,dim); }

/****************************************************************************
 * Gradient operators
 ****************************************************************************/

template<class T>
inline TinyVector<_bz_typename T::T_numtype,2> grad2D(T& A) {
  return TinyVector<_bz_typename T::T_numtype,2>(
    central12(A,firstDim),
    central12(A,secondDim));
}

template<class T>
inline TinyVector<_bz_typename T::T_numtype,2> grad2D4(T& A) {
  return TinyVector<_bz_typename T::T_numtype,2>(
    central14(A,firstDim),
    central14(A,secondDim));
}

template<class T>
inline TinyVector<_bz_typename T::T_numtype,3> grad3D(T& A) {
  return TinyVector<_bz_typename T::T_numtype,3>(
    central12(A,firstDim),
    central12(A,secondDim),
    central12(A,thirdDim));
}

template<class T>
inline TinyVector<_bz_typename T::T_numtype,3> grad3D4(T& A) {
  return TinyVector<_bz_typename T::T_numtype,3>(
    central14(A,firstDim),
    central14(A,secondDim),
    central14(A,thirdDim));
}

template<class T>
inline TinyVector<_bz_typename T::T_numtype,2> grad2Dn(T& A) {
  return TinyVector<_bz_typename T::T_numtype,2>(
    central12n(A,firstDim),
    central12n(A,secondDim));
}

template<class T>
inline TinyVector<_bz_typename T::T_numtype,2> grad2D4n(T& A) {
  return TinyVector<_bz_typename T::T_numtype,2>(
    central14n(A,firstDim),
    central14n(A,secondDim));
}

template<class T>
inline TinyVector<_bz_typename T::T_numtype,3> grad3Dn(T& A) {
  return TinyVector<_bz_typename T::T_numtype,3>(
    central12n(A,firstDim),
    central12n(A,secondDim),
    central12n(A,thirdDim));
}

template<class T>
inline TinyVector<_bz_typename T::T_numtype,3> grad3D4n(T& A) {
  return TinyVector<_bz_typename T::T_numtype,3>(
    central14n(A,firstDim),
    central14n(A,secondDim),
    central14n(A,thirdDim));
}

/****************************************************************************
 * Grad-squared operators
 ****************************************************************************/

template<class T>
inline TinyVector<_bz_typename T::T_numtype,2> gradSqr2D(T& A) {
  return TinyVector<_bz_typename T::T_numtype,2>(
    central22(A,firstDim),
    central22(A,secondDim));
}

template<class T>
inline TinyVector<_bz_typename T::T_numtype,2> gradSqr2D4(T& A) {
  return TinyVector<_bz_typename T::T_numtype,2>(
    central24(A,firstDim),
    central24(A,secondDim));
}

template<class T>
inline TinyVector<_bz_typename T::T_numtype,3> gradSqr3D(T& A) {
  return TinyVector<_bz_typename T::T_numtype,3>(
    central22(A,firstDim),
    central22(A,secondDim),
    central22(A,thirdDim));
}

template<class T>
inline TinyVector<_bz_typename T::T_numtype,3> gradSqr3D4(T& A) {
  return TinyVector<_bz_typename T::T_numtype,3>(
    central24(A,firstDim),
    central24(A,secondDim),
    central24(A,thirdDim));
}

/****************************************************************************
 * Grad-squared operators (normalized)
 ****************************************************************************/

template<class T>
inline TinyVector<_bz_typename T::T_numtype,2> gradSqr2Dn(T& A) {
  return gradSqr2D(A);
}

template<class T>
inline TinyVector<_bz_typename T::T_numtype,2> gradSqr2D4n(T& A) {
  return TinyVector<_bz_typename T::T_numtype,2>(
    central24(A,firstDim) * recip_12,
    central24(A,secondDim) * recip_12);
}

template<class T>
inline TinyVector<_bz_typename T::T_numtype,3> gradSqr3Dn(T& A) {
  return gradSqr3D(A);
}

template<class T>
inline TinyVector<_bz_typename T::T_numtype,3> gradSqr3D4n(T& A) {
  return TinyVector<_bz_typename T::T_numtype,3>(
    central24(A,firstDim) * recip_12,
    central24(A,secondDim) * recip_12,
    central24(A,thirdDim) * recip_12);
}

/****************************************************************************
 * Gradient operators on a vector field
 ****************************************************************************/

template<class T>
inline TinyMatrix<_bz_typename multicomponent_traits<_bz_typename 
    T::T_numtype>::T_element, 3, 3>
Jacobian3D(T& A)
{
    const int x=0, y=1, z=2;
    const int u=0, v=1, w=2;

    TinyMatrix<_bz_typename multicomponent_traits<_bz_typename 
        T::T_numtype>::T_element, 3, 3> grad;

    grad(u,x) = central12(A,u,x);
    grad(u,y) = central12(A,u,y);
    grad(u,z) = central12(A,u,z);
    grad(v,x) = central12(A,v,x);
    grad(v,y) = central12(A,v,y);
    grad(v,z) = central12(A,v,z);
    grad(w,x) = central12(A,w,x);
    grad(w,y) = central12(A,w,y);
    grad(w,z) = central12(A,w,z);

    return grad;
}

template<class T>
inline TinyMatrix<_bz_typename multicomponent_traits<_bz_typename 
    T::T_numtype>::T_element, 3, 3>
Jacobian3Dn(T& A)
{
    const int x=0, y=1, z=2;
    const int u=0, v=1, w=2;

    TinyMatrix<_bz_typename multicomponent_traits<_bz_typename 
        T::T_numtype>::T_element, 3, 3> grad;
    
    grad(u,x) = central12n(A,u,x);
    grad(u,y) = central12n(A,u,y);
    grad(u,z) = central12n(A,u,z);
    grad(v,x) = central12n(A,v,x);
    grad(v,y) = central12n(A,v,y);
    grad(v,z) = central12n(A,v,z);
    grad(w,x) = central12n(A,w,x);
    grad(w,y) = central12n(A,w,y);
    grad(w,z) = central12n(A,w,z);

    return grad;
}

template<class T>
inline TinyMatrix<_bz_typename multicomponent_traits<_bz_typename
    T::T_numtype>::T_element, 3, 3>
Jacobian3D4(T& A)
{
    const int x=0, y=1, z=2;
    const int u=0, v=1, w=2;

    TinyMatrix<_bz_typename multicomponent_traits<_bz_typename 
        T::T_numtype>::T_element, 3, 3> grad;
    
    grad(u,x) = central14(A,u,x);
    grad(u,y) = central14(A,u,y);
    grad(u,z) = central14(A,u,z);
    grad(v,x) = central14(A,v,x);
    grad(v,y) = central14(A,v,y);
    grad(v,z) = central14(A,v,z);
    grad(w,x) = central14(A,w,x);
    grad(w,y) = central14(A,w,y);
    grad(w,z) = central14(A,w,z);

    return grad;
}

template<class T>
inline TinyMatrix<_bz_typename multicomponent_traits<_bz_typename
    T::T_numtype>::T_element, 3, 3>
Jacobian3D4n(T& A)
{
    const int x=0, y=1, z=2;
    const int u=0, v=1, w=2;

    TinyMatrix<_bz_typename multicomponent_traits<_bz_typename 
        T::T_numtype>::T_element, 3, 3> grad;
    
    grad(u,x) = central14n(A,u,x);
    grad(u,y) = central14n(A,u,y);
    grad(u,z) = central14n(A,u,z);
    grad(v,x) = central14n(A,v,x);
    grad(v,y) = central14n(A,v,y);
    grad(v,z) = central14n(A,v,z);
    grad(w,x) = central14n(A,w,x);
    grad(w,y) = central14n(A,w,y);
    grad(w,z) = central14n(A,w,z);

    return grad;
}

/****************************************************************************
 * Curl operators
 ****************************************************************************/

// O(h^2) curl, using central difference

template<class T>
inline TinyVector<_bz_typename T::T_numtype,3> 
curl(T& vx, T& vy, T& vz) {
  const int x = firstDim, y = secondDim, z = thirdDim;

  return TinyVector<_bz_typename T::T_numtype,3>(
    central12(vz,y)-central12(vy,z),
    central12(vx,z)-central12(vz,x),
    central12(vy,x)-central12(vx,y));
}

// Normalized O(h^2) curl, using central difference
template<class T>
inline TinyVector<_bz_typename T::T_numtype,3>
curln(T& vx, T& vy, T& vz) {
  const int x = firstDim, y = secondDim, z = thirdDim;

  return TinyVector<_bz_typename T::T_numtype,3>(
    (central12(vz,y)-central12(vy,z)) * recip_2,
    (central12(vx,z)-central12(vz,x)) * recip_2,
    (central12(vy,x)-central12(vx,y)) * recip_2);
}

// Multicomponent curl
template<class T>
inline _bz_typename T::T_numtype curl(T& A) {
  const int x = firstDim, y = secondDim, z = thirdDim;

  return T::T_numtype(
    central12(A,z,y)-central12(A,y,z),
    central12(A,x,z)-central12(A,z,x),
    central12(A,y,x)-central12(A,x,y));
}

// Normalized multicomponent curl
template<class T>
inline _bz_typename T::T_numtype curln(T& A) {
  const int x = firstDim, y = secondDim, z = thirdDim;

  return T::T_numtype(
    (central12(A,z,y)-central12(A,y,z)) * recip_2,
    (central12(A,x,z)-central12(A,z,x)) * recip_2,
    (central12(A,y,x)-central12(A,x,y)) * recip_2);
}

// O(h^4) curl, using 4th order central difference
template<class T>
inline TinyVector<_bz_typename T::T_numtype,3>
curl4(T& vx, T& vy, T& vz) {
  const int x = firstDim, y = secondDim, z = thirdDim;

  return TinyVector<_bz_typename T::T_numtype,3>(
    central14(vz,y)-central14(vy,z),
    central14(vx,z)-central14(vz,x),
    central14(vy,x)-central14(vx,y));
}

// O(h^4) curl, using 4th order central difference (multicomponent version)
template<class T>
inline _bz_typename T::T_numtype
curl4(T& A) {
  const int x = firstDim, y = secondDim, z = thirdDim;

  return T::T_numtype(
    central14(A,z,y)-central14(A,y,z),
    central14(A,x,z)-central14(A,z,x),
    central14(A,y,x)-central14(A,x,y));
}

// Normalized O(h^4) curl, using 4th order central difference
template<class T>
inline TinyVector<_bz_typename T::T_numtype,3>
curl4n(T& vx, T& vy, T& vz) {
  const int x = firstDim, y = secondDim, z = thirdDim;

  return TinyVector<_bz_typename T::T_numtype,3>(
    (central14(vz,y)-central14(vy,z)) * recip_2,
    (central14(vx,z)-central14(vz,x)) * recip_2,
    (central14(vy,x)-central14(vx,y)) * recip_2);
}

// O(h^4) curl, using 4th order central difference (normalized multicomponent)
template<class T>
inline _bz_typename T::T_numtype
curl4n(T& A) {
  const int x = firstDim, y = secondDim, z = thirdDim;

  return T::T_numtype(
    (central14(A,z,y)-central14(A,y,z)) * recip_2,
    (central14(A,x,z)-central14(A,z,x)) * recip_2,
    (central14(A,y,x)-central14(A,x,y)) * recip_2);
}



// Two-dimensional curl

template<class T>
inline _bz_typename T::T_numtype
curl(T& vx, T& vy) {
  const int x = firstDim, y = secondDim;

  return central12(vy,x)-central12(vx,y);
}

template<class T>
inline _bz_typename T::T_numtype
curln(T& vx, T& vy) {
  const int x = firstDim, y = secondDim;

  return (central12(vy,x)-central12(vx,y)) * recip_2;
}

// Multicomponent curl
template<class T>
inline _bz_typename T::T_numtype::T_numtype curl2D(T& A) {
  const int x = firstDim, y = secondDim;
  return central12(A,y,x)-central12(A,x,y);
}

template<class T>
inline _bz_typename T::T_numtype::T_numtype curl2Dn(T& A) {
  const int x = firstDim, y = secondDim;
  return (central12(A,y,x)-central12(A,x,y)) * recip_2;
}


// 4th order versions

template<class T>
inline _bz_typename T::T_numtype
curl4(T& vx, T& vy) {
  const int x = firstDim, y = secondDim;

  return central14(vy,x)-central14(vx,y);
}

template<class T>
inline _bz_typename T::T_numtype
curl4n(T& vx, T& vy) {
  const int x = firstDim, y = secondDim;

  return (central14(vy,x)-central14(vx,y)) * recip_12;
}

// Multicomponent curl
template<class T>
inline _bz_typename T::T_numtype::T_numtype curl2D4(T& A) {
  const int x = firstDim, y = secondDim;
  return central14(A,y,x)-central14(A,x,y);
}

template<class T>
inline _bz_typename T::T_numtype::T_numtype curl2D4n(T& A) {
  const int x = firstDim, y = secondDim;
  return (central14(A,y,x)-central14(A,x,y)) * recip_12;
}

/****************************************************************************
 * Divergence
 ****************************************************************************/


BZ_DECLARE_STENCIL_OPERATOR2(div,vx,vy)
  return central12(vx,firstDim) + central12(vy,secondDim);
BZ_END_STENCIL_OPERATOR

BZ_DECLARE_STENCIL_OPERATOR2(divn,vx,vy)
  return (central12(vx,firstDim) + central12(vy,secondDim))
     * recip_2;
BZ_END_STENCIL_OPERATOR

BZ_DECLARE_STENCIL_OPERATOR2(div4,vx,vy)
  return central14(vx,firstDim) + central14(vy,secondDim);
BZ_END_STENCIL_OPERATOR

BZ_DECLARE_STENCIL_OPERATOR2(div4n,vx,vy)
  return (central14(vx,firstDim) + central14(vy,secondDim))
    * recip_12;
BZ_END_STENCIL_OPERATOR

BZ_DECLARE_STENCIL_OPERATOR3(div,vx,vy,vz)
  return central12(vx,firstDim) + central12(vy,secondDim) 
    + central12(vz,thirdDim);
BZ_END_STENCIL_OPERATOR

BZ_DECLARE_STENCIL_OPERATOR3(divn,vx,vy,vz)
  return (central12(vx,firstDim) + central12(vy,secondDim) 
    + central12(vz,thirdDim)) * recip_2;
BZ_END_STENCIL_OPERATOR

BZ_DECLARE_STENCIL_OPERATOR3(div4,vx,vy,vz)
  return central14(vx,firstDim) + central14(vy,secondDim) 
    + central14(vz,thirdDim);
BZ_END_STENCIL_OPERATOR

BZ_DECLARE_STENCIL_OPERATOR3(div4n,vx,vy,vz)
  return (central14(vx,firstDim) + central14(vy,secondDim)
    + central14(vz,thirdDim)) * recip_12;
BZ_END_STENCIL_OPERATOR

template<class T>
inline _bz_typename T::T_numtype::T_numtype
div2D(T& A) {
  const int x = firstDim, y = secondDim;
  return central12(A,x,x) + central12(A,y,y);
}

template<class T>
inline _bz_typename T::T_numtype::T_numtype
div2D4(T& A) {
  const int x = firstDim, y = secondDim;
  return central14(A,x,x) + central14(A,y,y);
}

template<class T>
inline _bz_typename T::T_numtype::T_numtype
div2Dn(T& A) {
  const int x = firstDim, y = secondDim;
  return (central12(A,x,x) + central12(A,y,y)) * recip_2;
}

template<class T>
inline _bz_typename T::T_numtype::T_numtype
div2D4n(T& A) {
  const int x = firstDim, y = secondDim;
  return (central14(A,x,x) + central14(A,y,y)) * recip_12;
}

template<class T>
inline _bz_typename T::T_numtype::T_numtype
div3D(T& A) {
  const int x = firstDim, y = secondDim, z = thirdDim;
  return central12(A,x,x) + central12(A,y,y) + central12(A,z,z);
}

template<class T>
inline _bz_typename T::T_numtype::T_numtype
div3D4(T& A) {
  const int x = firstDim, y = secondDim, z = thirdDim;
  return central14(A,x,x) + central14(A,y,y) + central14(A,z,z);
}

template<class T>
inline _bz_typename T::T_numtype::T_numtype
div3Dn(T& A) {
  const int x = firstDim, y = secondDim, z = thirdDim;
  return (central12(A,x,x) + central12(A,y,y) + central12(A,z,z))
    * recip_2;
}

template<class T>
inline _bz_typename T::T_numtype::T_numtype
div3D4n(T& A) {
  const int x = firstDim, y = secondDim, z = thirdDim;
  return (central14(A,x,x) + central14(A,y,y) + central14(A,z,z))
    * recip_12;
}

/****************************************************************************
 * Mixed Partial derivatives
 ****************************************************************************/

template<class T>
inline _bz_typename T::T_numtype
mixed22(T& A, int x, int y)
{
    return A.shift(-1,x,-1,y) - A.shift(-1,x,1,y)
        -A.shift(1,x,-1,y) + A.shift(1,x,1,y);
}

template<class T>
inline _bz_typename T::T_numtype
mixed22n(T& A, int x, int y)
{
    return mixed22(A, x, y) * recip_4;
}

template<class T>
inline _bz_typename T::T_numtype
mixed24(T& A, int x, int y)
{
    return 64.*(A.shift(-1,x,-1,y) - A.shift(-1,x,1,y)
        -A.shift(1,x,-1,y) + A.shift(1,x,1,y))
        + (A.shift(-2,x,+1,y) - A.shift(-1,x,2,y)
        - A.shift(1,x,2,y)-A.shift(2,x,1,y)
        + A.shift(2,x,-1,y)+A.shift(1,x,-2,y)
        - A.shift(-1,x,-2,y)+A.shift(-2,x,-1,y))
        + 8.*(A.shift(-1,x,1,y)+A.shift(-1,x,2,y)
        -A.shift(2,x,-2,y) + A.shift(2,x,2,y));
}

template<class T>
inline _bz_typename T::T_numtype
mixed24n(T& A, int x, int y)
{
    return mixed24(A,x,y) * recip_144;
}

/****************************************************************************
 * Smoothers
 ****************************************************************************/

// NEEDS_WORK-- put other stencil operators here:
//   Average5pt2D
//   Average7pt3D
// etc.

/****************************************************************************
 * Stencil operators with geometry (experimental)
 ****************************************************************************/

template<class T>
inline _bz_typename multicomponent_traits<_bz_typename
    T::T_numtype>::T_element div3DVec4(T& A, 
    const UniformCubicGeometry<3>& geom)
{
    const int x = 0, y = 1, z = 2;

    return (central14(A, x, firstDim) + central14(A, y, secondDim)
        + central14(A, z, thirdDim)) * recip_12 * geom.recipSpatialStep();
}

template<class T>
inline _bz_typename T::T_numtype Laplacian3D4(T& A, 
    const UniformCubicGeometry<3>& geom)
{
    return Laplacian3D4n(A) * geom.recipSpatialStepPow2();
}

template<class T>
inline _bz_typename T::T_numtype Laplacian3DVec4(T& A,
    const UniformCubicGeometry<3>& geom)
{
    typedef _bz_typename T::T_numtype vector3d;
    typedef _bz_typename multicomponent_traits<vector3d>::T_element 
        T_element;
    const int u = 0, v = 1, w = 2;
    const int x = 0, y = 1, z = 2;

    // central24 is a 5-point stencil
    // This is a 9*5 = 45 point stencil

    T_element t1 = (central24(A,u,x) + central24(A,u,y) + central24(A,u,z))
        * recip_12 * geom.recipSpatialStepPow2();

    T_element t2 = (central24(A,v,x) + central24(A,v,y) + central24(A,v,z))
        * recip_12 * geom.recipSpatialStepPow2();

    T_element t3 = (central24(A,w,x) + central24(A,w,y) + central24(A,w,z))
        * recip_12 * geom.recipSpatialStepPow2();

    return vector3d(t1,t2,t3);
}

template<class T>
inline TinyMatrix<_bz_typename multicomponent_traits<_bz_typename
    T::T_numtype>::T_element, 3, 3>
grad3DVec4(T& A, const UniformCubicGeometry<3>& geom)
{
    const int x=0, y=1, z=2;
    const int u=0, v=1, w=2;

    TinyMatrix<_bz_typename multicomponent_traits<_bz_typename
        T::T_numtype>::T_element, 3, 3> grad;

    // This is a 9*4 = 36 point stencil
    grad(u,x) = central14n(A,u,x) * geom.recipSpatialStep();
    grad(u,y) = central14n(A,u,y) * geom.recipSpatialStep();
    grad(u,z) = central14n(A,u,z) * geom.recipSpatialStep();
    grad(v,x) = central14n(A,v,x) * geom.recipSpatialStep();
    grad(v,y) = central14n(A,v,y) * geom.recipSpatialStep();
    grad(v,z) = central14n(A,v,z) * geom.recipSpatialStep();
    grad(w,x) = central14n(A,w,x) * geom.recipSpatialStep();
    grad(w,y) = central14n(A,w,y) * geom.recipSpatialStep();
    grad(w,z) = central14n(A,w,z) * geom.recipSpatialStep();

    return grad;
}

template<class T>
inline TinyVector<_bz_typename T::T_numtype,3> grad3D4(T& A,
    const UniformCubicGeometry<3>& geom) {
  return TinyVector<_bz_typename T::T_numtype,3>(
    central14(A,firstDim) * recip_12 * geom.recipSpatialStep(),
    central14(A,secondDim) * recip_12 * geom.recipSpatialStep(),
    central14(A,thirdDim) * recip_12 * geom.recipSpatialStep());
}

BZ_NAMESPACE_END

#endif // BZ_ARRAYSTENCILOPS_H

