// Generated: genmathfunc.cpp Feb  3 1999 09:08:50

#ifndef BZ_MATHFUNC_H
#define BZ_MATHFUNC_H

#ifndef BZ_APPLICS_H
 #include <blitz/applics.h>
#endif


#ifndef BZ_PRETTYPRINT_H
 #include <blitz/prettyprint.h>
#endif

BZ_NAMESPACE(blitz)

// abs(P_numtype1)    Absolute value
template<class P_numtype1>
class _bz_abs : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype1 T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(abs)(x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "abs(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// abs(long)
template<>
class _bz_abs<long> : public OneOperandApplicativeTemplatesBase {
public:
    typedef long T_numtype1;
    typedef long T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(labs)((long)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "labs(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// abs(float)
template<>
class _bz_abs<float> : public OneOperandApplicativeTemplatesBase {
public:
    typedef float T_numtype1;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(fabs)((float)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "fabs(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// abs(double)
template<>
class _bz_abs<double> : public OneOperandApplicativeTemplatesBase {
public:
    typedef double T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(fabs)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "fabs(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// abs(long double)
template<>
class _bz_abs<long double> : public OneOperandApplicativeTemplatesBase {
public:
    typedef long double T_numtype1;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(fabs)((long double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "fabs(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// abs(complex<float> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_abs<complex<float> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<float>  T_numtype1;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(abs)((complex<float> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "abs(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// abs(complex<double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_abs<complex<double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<double>  T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(abs)((complex<double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "abs(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// abs(complex<long double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_abs<complex<long double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<long double>  T_numtype1;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(abs)((complex<long double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "abs(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// acos(P_numtype1)    Inverse cosine
template<class P_numtype1>
class _bz_acos : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(acos)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "acos(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// acos(float)
template<>
class _bz_acos<float> : public OneOperandApplicativeTemplatesBase {
public:
    typedef float T_numtype1;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(acos)((float)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "acos(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// acos(long double)
template<>
class _bz_acos<long double> : public OneOperandApplicativeTemplatesBase {
public:
    typedef long double T_numtype1;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(acos)((long double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "acos(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// acosh(P_numtype1)    Inverse hyperbolic cosine
#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
class _bz_acosh : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(acosh)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "acosh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// asin(P_numtype1)    Inverse sine
template<class P_numtype1>
class _bz_asin : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(asin)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "asin(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// asin(float)
template<>
class _bz_asin<float> : public OneOperandApplicativeTemplatesBase {
public:
    typedef float T_numtype1;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(asin)((float)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "asin(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// asin(long double)
template<>
class _bz_asin<long double> : public OneOperandApplicativeTemplatesBase {
public:
    typedef long double T_numtype1;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(asin)((long double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "asin(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// asinh(P_numtype1)    Inverse hyperbolic sine
#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
class _bz_asinh : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(asinh)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "asinh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// arg(P_numtype1)
#ifdef BZ_HAVE_COMPLEX_MATH
template<class P_numtype1>
class _bz_arg : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype1 T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return 0; }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "0(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// arg(complex<float> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_arg<complex<float> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<float>  T_numtype1;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(arg)((complex<float> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "arg(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// arg(complex<double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_arg<complex<double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<double>  T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(arg)((complex<double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "arg(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// arg(complex<long double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_arg<complex<long double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<long double>  T_numtype1;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(arg)((complex<long double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "arg(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// atan(P_numtype1)    Inverse tangent
template<class P_numtype1>
class _bz_atan : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(atan)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "atan(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// atan(float)
template<>
class _bz_atan<float> : public OneOperandApplicativeTemplatesBase {
public:
    typedef float T_numtype1;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(atan)((float)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "atan(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// atan(long double)
template<>
class _bz_atan<long double> : public OneOperandApplicativeTemplatesBase {
public:
    typedef long double T_numtype1;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(atan)((long double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "atan(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// atanh(P_numtype1)    Inverse hyperbolic tangent
#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
class _bz_atanh : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(atanh)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "atanh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// atan2(P_numtype1, P_numtype2)    Inverse tangent
template<class P_numtype1, class P_numtype2>
class _bz_atan2 : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)
    { return BZ_MATHFN_SCOPE(atan2)((double)x,(double)y); }

    template<class T1, class T2>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a, const T2& b)
    {
        str += "atan2(";
        a.prettyPrint(str,format);
        str += ",";
        b.prettyPrint(str,format);
        str += ")";
    }
};

// atan2(float, float)
template<>
class _bz_atan2<float, float > : public TwoOperandApplicativeTemplatesBase {
public:
    typedef float T_numtype1;
    typedef float T_numtype2;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)
    { return BZ_MATHFN_SCOPE(atan2)((float)x,(float)y); }

    template<class T1, class T2>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a, const T2& b)
    {
        str += "atan2(";
        a.prettyPrint(str,format);
        str += ",";
        b.prettyPrint(str,format);
        str += ")";
    }
};

// atan2(long double, long double)
template<>
class _bz_atan2<long double, long double > : public TwoOperandApplicativeTemplatesBase {
public:
    typedef long double T_numtype1;
    typedef long double T_numtype2;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)
    { return BZ_MATHFN_SCOPE(atan2)((long double)x,(long double)y); }

    template<class T1, class T2>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a, const T2& b)
    {
        str += "atan2(";
        a.prettyPrint(str,format);
        str += ",";
        b.prettyPrint(str,format);
        str += ")";
    }
};

// _class(P_numtype1)    Classification of float-point value (FP_xxx)
#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1>
class _bz__class : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef int T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(_class)(x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "_class(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// cbrt(P_numtype1)    Cube root
#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
class _bz_cbrt : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(cbrt)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "cbrt(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// ceil(P_numtype1)    Ceiling
template<class P_numtype1>
class _bz_ceil : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(ceil)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "ceil(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// ceil(float)
template<>
class _bz_ceil<float> : public OneOperandApplicativeTemplatesBase {
public:
    typedef float T_numtype1;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(ceil)((float)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "ceil(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// ceil(long double)
template<>
class _bz_ceil<long double> : public OneOperandApplicativeTemplatesBase {
public:
    typedef long double T_numtype1;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(ceil)((long double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "ceil(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// conj(P_numtype1)
#ifdef BZ_HAVE_COMPLEX_MATH
template<class P_numtype1>
class _bz_conj : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype1 T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(conj)(x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "conj(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// cos(P_numtype1)    Cosine
template<class P_numtype1>
class _bz_cos : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(cos)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "cos(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// cos(float)
template<>
class _bz_cos<float> : public OneOperandApplicativeTemplatesBase {
public:
    typedef float T_numtype1;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(cos)((float)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "cos(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// cos(long double)
template<>
class _bz_cos<long double> : public OneOperandApplicativeTemplatesBase {
public:
    typedef long double T_numtype1;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(cos)((long double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "cos(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// cos(complex<float> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_cos<complex<float> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<float>  T_numtype1;
    typedef complex<float> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(cos)((complex<float> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "cos(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// cos(complex<double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_cos<complex<double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<double>  T_numtype1;
    typedef complex<double> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(cos)((complex<double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "cos(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// cos(complex<long double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_cos<complex<long double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<long double>  T_numtype1;
    typedef complex<long double> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(cos)((complex<long double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "cos(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// copysign(P_numtype1, P_numtype2)
#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1, class P_numtype2>
class _bz_copysign : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)
    { return BZ_IEEEMATHFN_SCOPE(copysign)((double)x,(double)y); }

    template<class T1, class T2>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a, const T2& b)
    {
        str += "copysign(";
        a.prettyPrint(str,format);
        str += ",";
        b.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// cosh(P_numtype1)    Hyperbolic cosine
template<class P_numtype1>
class _bz_cosh : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(cosh)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "cosh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// cosh(float)
template<>
class _bz_cosh<float> : public OneOperandApplicativeTemplatesBase {
public:
    typedef float T_numtype1;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(cosh)((float)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "cosh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// cosh(long double)
template<>
class _bz_cosh<long double> : public OneOperandApplicativeTemplatesBase {
public:
    typedef long double T_numtype1;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(cosh)((long double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "cosh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// cosh(complex<float> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_cosh<complex<float> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<float>  T_numtype1;
    typedef complex<float> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(cosh)((complex<float> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "cosh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// cosh(complex<double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_cosh<complex<double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<double>  T_numtype1;
    typedef complex<double> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(cosh)((complex<double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "cosh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// cosh(complex<long double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_cosh<complex<long double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<long double>  T_numtype1;
    typedef complex<long double> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(cosh)((complex<long double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "cosh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// drem(P_numtype1, P_numtype2)    Remainder
#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1, class P_numtype2>
class _bz_drem : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)
    { return BZ_IEEEMATHFN_SCOPE(drem)((double)x,(double)y); }

    template<class T1, class T2>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a, const T2& b)
    {
        str += "drem(";
        a.prettyPrint(str,format);
        str += ",";
        b.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// exp(P_numtype1)    Exponential
template<class P_numtype1>
class _bz_exp : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(exp)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "exp(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// exp(float)
template<>
class _bz_exp<float> : public OneOperandApplicativeTemplatesBase {
public:
    typedef float T_numtype1;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(exp)((float)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "exp(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// exp(long double)
template<>
class _bz_exp<long double> : public OneOperandApplicativeTemplatesBase {
public:
    typedef long double T_numtype1;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(exp)((long double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "exp(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// exp(complex<float> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_exp<complex<float> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<float>  T_numtype1;
    typedef complex<float> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(exp)((complex<float> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "exp(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// exp(complex<double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_exp<complex<double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<double>  T_numtype1;
    typedef complex<double> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(exp)((complex<double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "exp(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// exp(complex<long double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_exp<complex<long double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<long double>  T_numtype1;
    typedef complex<long double> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(exp)((complex<long double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "exp(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// expm1(P_numtype1)    Exp(x)-1
#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
class _bz_expm1 : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(expm1)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "expm1(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// erf(P_numtype1)    Error function
#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
class _bz_erf : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(erf)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "erf(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// erfc(P_numtype1)    Complementary error function
#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
class _bz_erfc : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(erfc)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "erfc(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// floor(P_numtype1)    Floor function
template<class P_numtype1>
class _bz_floor : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(floor)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "floor(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// floor(float)
template<>
class _bz_floor<float> : public OneOperandApplicativeTemplatesBase {
public:
    typedef float T_numtype1;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(floor)((float)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "floor(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// floor(long double)
template<>
class _bz_floor<long double> : public OneOperandApplicativeTemplatesBase {
public:
    typedef long double T_numtype1;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(floor)((long double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "floor(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// fmod(P_numtype1, P_numtype2)    Modulo remainder
template<class P_numtype1, class P_numtype2>
class _bz_fmod : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)
    { return BZ_IEEEMATHFN_SCOPE(fmod)((double)x,(double)y); }

    template<class T1, class T2>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a, const T2& b)
    {
        str += "fmod(";
        a.prettyPrint(str,format);
        str += ",";
        b.prettyPrint(str,format);
        str += ")";
    }
};

// hypot(P_numtype1, P_numtype2)    sqrt(x*x+y*y)
#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1, class P_numtype2>
class _bz_hypot : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)
    { return BZ_IEEEMATHFN_SCOPE(hypot)((double)x,(double)y); }

    template<class T1, class T2>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a, const T2& b)
    {
        str += "hypot(";
        a.prettyPrint(str,format);
        str += ",";
        b.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// ilogb(P_numtype1)    Integer unbiased exponent
#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
class _bz_ilogb : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef int T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(ilogb)(x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "ilogb(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// blitz_isnan(P_numtype1)    Nonzero if NaNS or NaNQ
#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
class _bz_blitz_isnan : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef int T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { 
#ifdef isnan
        // Some platforms define isnan as a macro, which causes the
        // BZ_IEEEMATHFN_SCOPE macro to break.
        return isnan(x); 
#else
        return BZ_IEEEMATHFN_SCOPE(isnan)(x);
#endif
    }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "blitz_isnan(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// itrunc(P_numtype1)    Truncate and convert to integer
#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1>
class _bz_itrunc : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef int T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(itrunc)(x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "itrunc(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// j0(P_numtype1)    Bessel function first kind, order 0
#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
class _bz_j0 : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(j0)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "j0(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// j1(P_numtype1)    Bessel function first kind, order 1
#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
class _bz_j1 : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(j1)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "j1(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// lgamma(P_numtype1)    Log absolute gamma
#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
class _bz_lgamma : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(lgamma)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "lgamma(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// log(P_numtype1)    Natural logarithm
template<class P_numtype1>
class _bz_log : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(log)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "log(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// log(float)
template<>
class _bz_log<float> : public OneOperandApplicativeTemplatesBase {
public:
    typedef float T_numtype1;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(log)((float)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "log(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// log(long double)
template<>
class _bz_log<long double> : public OneOperandApplicativeTemplatesBase {
public:
    typedef long double T_numtype1;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(log)((long double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "log(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// log(complex<float> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_log<complex<float> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<float>  T_numtype1;
    typedef complex<float> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(log)((complex<float> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "log(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// log(complex<double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_log<complex<double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<double>  T_numtype1;
    typedef complex<double> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(log)((complex<double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "log(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// log(complex<long double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_log<complex<long double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<long double>  T_numtype1;
    typedef complex<long double> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(log)((complex<long double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "log(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// logb(P_numtype1)    Unbiased exponent (IEEE)
#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
class _bz_logb : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(logb)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "logb(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// log1p(P_numtype1)    Compute log(1 + x)
#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
class _bz_log1p : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(log1p)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "log1p(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// log10(P_numtype1)    Logarithm base 10
template<class P_numtype1>
class _bz_log10 : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(log10)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "log10(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// log10(float)
template<>
class _bz_log10<float> : public OneOperandApplicativeTemplatesBase {
public:
    typedef float T_numtype1;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(log10)((float)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "log10(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// log10(long double)
template<>
class _bz_log10<long double> : public OneOperandApplicativeTemplatesBase {
public:
    typedef long double T_numtype1;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(log10)((long double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "log10(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// nearest(P_numtype1)    Nearest floating point integer
#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1>
class _bz_nearest : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(nearest)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "nearest(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// nextafter(P_numtype1, P_numtype2)    Next representable number after x towards y
#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1, class P_numtype2>
class _bz_nextafter : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)
    { return BZ_IEEEMATHFN_SCOPE(nextafter)((double)x,(double)y); }

    template<class T1, class T2>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a, const T2& b)
    {
        str += "nextafter(";
        a.prettyPrint(str,format);
        str += ",";
        b.prettyPrint(str,format);
        str += ")";
    }
};
#endif

template<class P_numtype>
class _bz_negate : public OneOperandApplicativeTemplatesBase {
public:
    typedef BZ_SIGNEDTYPE(P_numtype) T_numtype;

    static inline T_numtype apply(T_numtype x)
    { return -x; }

		template<class T1>
		static void prettyPrint(string& str, prettyPrintFormat& format, const T1& a)
		{
		    str += "-(";
			  a.prettyPrint(str,format);
			  str += ")";
		}
};

// norm(P_numtype1)
#ifdef BZ_HAVE_COMPLEX_MATH
template<class P_numtype1>
class _bz_norm : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype1 T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(norm)(x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "norm(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// polar(P_numtype1, P_numtype2)
#ifdef BZ_HAVE_COMPLEX_MATH
template<class P_numtype1, class P_numtype2>
class _bz_polar : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef complex<T_numtype1> T_numtype;

    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)
    { return BZ_CMATHFN_SCOPE(polar)(x,y); }

    template<class T1, class T2>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a, const T2& b)
    {
        str += "polar(";
        a.prettyPrint(str,format);
        str += ",";
        b.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// pow(P_numtype1, P_numtype2)    Power
template<class P_numtype1, class P_numtype2>
class _bz_pow : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)
    { return BZ_MATHFN_SCOPE(pow)((double)x,(double)y); }

    template<class T1, class T2>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a, const T2& b)
    {
        str += "pow(";
        a.prettyPrint(str,format);
        str += ",";
        b.prettyPrint(str,format);
        str += ")";
    }
};

// pow(float, float)
template<>
class _bz_pow<float, float > : public TwoOperandApplicativeTemplatesBase {
public:
    typedef float T_numtype1;
    typedef float T_numtype2;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)
    { return BZ_MATHFN_SCOPE(pow)((float)x,(float)y); }

    template<class T1, class T2>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a, const T2& b)
    {
        str += "pow(";
        a.prettyPrint(str,format);
        str += ",";
        b.prettyPrint(str,format);
        str += ")";
    }
};

// pow(long double, long double)
template<>
class _bz_pow<long double, long double > : public TwoOperandApplicativeTemplatesBase {
public:
    typedef long double T_numtype1;
    typedef long double T_numtype2;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)
    { return BZ_MATHFN_SCOPE(pow)((long double)x,(long double)y); }

    template<class T1, class T2>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a, const T2& b)
    {
        str += "pow(";
        a.prettyPrint(str,format);
        str += ",";
        b.prettyPrint(str,format);
        str += ")";
    }
};

// pow(complex<float>, complex<float>)
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_pow<complex<float>, complex<float> > : public TwoOperandApplicativeTemplatesBase {
public:
    typedef complex<float> T_numtype1;
    typedef complex<float> T_numtype2;
    typedef complex<float> T_numtype;

    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)
    { return BZ_CMATHFN_SCOPE(pow)((complex<float>)x,(complex<float>)y); }

    template<class T1, class T2>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a, const T2& b)
    {
        str += "pow(";
        a.prettyPrint(str,format);
        str += ",";
        b.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// pow(complex<double>, complex<double>)
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_pow<complex<double>, complex<double> > : public TwoOperandApplicativeTemplatesBase {
public:
    typedef complex<double> T_numtype1;
    typedef complex<double> T_numtype2;
    typedef complex<double> T_numtype;

    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)
    { return BZ_CMATHFN_SCOPE(pow)((complex<double>)x,(complex<double>)y); }

    template<class T1, class T2>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a, const T2& b)
    {
        str += "pow(";
        a.prettyPrint(str,format);
        str += ",";
        b.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// pow(complex<long double>, complex<long double>)
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_pow<complex<long double>, complex<long double> > : public TwoOperandApplicativeTemplatesBase {
public:
    typedef complex<long double> T_numtype1;
    typedef complex<long double> T_numtype2;
    typedef complex<long double> T_numtype;

    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)
    { return BZ_CMATHFN_SCOPE(pow)((complex<long double>)x,(complex<long double>)y); }

    template<class T1, class T2>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a, const T2& b)
    {
        str += "pow(";
        a.prettyPrint(str,format);
        str += ",";
        b.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// remainder(P_numtype1, P_numtype2)    Remainder
#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1, class P_numtype2>
class _bz_remainder : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)
    { return BZ_IEEEMATHFN_SCOPE(remainder)((double)x,(double)y); }

    template<class T1, class T2>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a, const T2& b)
    {
        str += "remainder(";
        a.prettyPrint(str,format);
        str += ",";
        b.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// rint(P_numtype1)    Round to floating point integer
#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
class _bz_rint : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(rint)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "rint(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// rsqrt(P_numtype1)    Reciprocal square root
#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1>
class _bz_rsqrt : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(rsqrt)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "rsqrt(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// scalb(P_numtype1, P_numtype2)    x * (2**y)
#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1, class P_numtype2>
class _bz_scalb : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)
    { return BZ_IEEEMATHFN_SCOPE(scalb)((double)x,(double)y); }

    template<class T1, class T2>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a, const T2& b)
    {
        str += "scalb(";
        a.prettyPrint(str,format);
        str += ",";
        b.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// sin(P_numtype1)    Sine
template<class P_numtype1>
class _bz_sin : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(sin)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sin(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// sin(float)
template<>
class _bz_sin<float> : public OneOperandApplicativeTemplatesBase {
public:
    typedef float T_numtype1;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(sin)((float)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sin(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// sin(long double)
template<>
class _bz_sin<long double> : public OneOperandApplicativeTemplatesBase {
public:
    typedef long double T_numtype1;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(sin)((long double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sin(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// sin(complex<float> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_sin<complex<float> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<float>  T_numtype1;
    typedef complex<float> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(sin)((complex<float> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sin(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// sin(complex<double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_sin<complex<double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<double>  T_numtype1;
    typedef complex<double> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(sin)((complex<double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sin(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// sin(complex<long double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_sin<complex<long double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<long double>  T_numtype1;
    typedef complex<long double> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(sin)((complex<long double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sin(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// sinh(P_numtype1)    Hyperbolic sine
template<class P_numtype1>
class _bz_sinh : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(sinh)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sinh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// sinh(float)
template<>
class _bz_sinh<float> : public OneOperandApplicativeTemplatesBase {
public:
    typedef float T_numtype1;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(sinh)((float)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sinh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// sinh(long double)
template<>
class _bz_sinh<long double> : public OneOperandApplicativeTemplatesBase {
public:
    typedef long double T_numtype1;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(sinh)((long double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sinh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// sinh(complex<float> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_sinh<complex<float> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<float>  T_numtype1;
    typedef complex<float> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(sinh)((complex<float> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sinh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// sinh(complex<double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_sinh<complex<double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<double>  T_numtype1;
    typedef complex<double> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(sinh)((complex<double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sinh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// sinh(complex<long double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_sinh<complex<long double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<long double>  T_numtype1;
    typedef complex<long double> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(sinh)((complex<long double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sinh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

template<class P_numtype>
class _bz_sqr : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype T_numtype;

    static inline T_numtype apply(T_numtype x)
    { return x*x; }
    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sqr(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

#ifdef BZ_HAVE_COMPLEX_MATH
// Specialization of _bz_sqr for complex<T>
template<class T>
class _bz_sqr<complex<T> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<T> T_numtype;

    static inline T_numtype apply(T_numtype x)
    {
        T r = x.real();  T i = x.imag();
        return T_numtype(r*r-i*i, 2*r*i);
    }
    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sqr(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// sqrt(P_numtype1)    Square root
template<class P_numtype1>
class _bz_sqrt : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(sqrt)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sqrt(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// sqrt(float)
template<>
class _bz_sqrt<float> : public OneOperandApplicativeTemplatesBase {
public:
    typedef float T_numtype1;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(sqrt)((float)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sqrt(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// sqrt(long double)
template<>
class _bz_sqrt<long double> : public OneOperandApplicativeTemplatesBase {
public:
    typedef long double T_numtype1;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(sqrt)((long double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sqrt(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// sqrt(complex<float> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_sqrt<complex<float> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<float>  T_numtype1;
    typedef complex<float> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(sqrt)((complex<float> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sqrt(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// sqrt(complex<double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_sqrt<complex<double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<double>  T_numtype1;
    typedef complex<double> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(sqrt)((complex<double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sqrt(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// sqrt(complex<long double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_sqrt<complex<long double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<long double>  T_numtype1;
    typedef complex<long double> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(sqrt)((complex<long double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "sqrt(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// tan(P_numtype1)    Tangent
template<class P_numtype1>
class _bz_tan : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(tan)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "tan(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// tan(float)
template<>
class _bz_tan<float> : public OneOperandApplicativeTemplatesBase {
public:
    typedef float T_numtype1;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(tan)((float)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "tan(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// tan(long double)
template<>
class _bz_tan<long double> : public OneOperandApplicativeTemplatesBase {
public:
    typedef long double T_numtype1;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(tan)((long double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "tan(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// tan(complex<float> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_tan<complex<float> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<float>  T_numtype1;
    typedef complex<float> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(tan)((complex<float> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "tan(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// tan(complex<double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_tan<complex<double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<double>  T_numtype1;
    typedef complex<double> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(tan)((complex<double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "tan(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// tan(complex<long double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_tan<complex<long double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<long double>  T_numtype1;
    typedef complex<long double> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(tan)((complex<long double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "tan(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// tanh(P_numtype1)    Hyperbolic tangent
template<class P_numtype1>
class _bz_tanh : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(tanh)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "tanh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// tanh(float)
template<>
class _bz_tanh<float> : public OneOperandApplicativeTemplatesBase {
public:
    typedef float T_numtype1;
    typedef float T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(tanh)((float)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "tanh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// tanh(long double)
template<>
class _bz_tanh<long double> : public OneOperandApplicativeTemplatesBase {
public:
    typedef long double T_numtype1;
    typedef long double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_MATHFN_SCOPE(tanh)((long double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "tanh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};

// tanh(complex<float> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_tanh<complex<float> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<float>  T_numtype1;
    typedef complex<float> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(tanh)((complex<float> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "tanh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// tanh(complex<double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_tanh<complex<double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<double>  T_numtype1;
    typedef complex<double> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(tanh)((complex<double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "tanh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// tanh(complex<long double> )
#ifdef BZ_HAVE_COMPLEX_MATH
template<>
class _bz_tanh<complex<long double> > : public OneOperandApplicativeTemplatesBase {
public:
    typedef complex<long double>  T_numtype1;
    typedef complex<long double> T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_CMATHFN_SCOPE(tanh)((complex<long double> )x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "tanh(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// uitrunc(P_numtype1)    Truncate and convert to unsigned
#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1>
class _bz_uitrunc : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef unsigned T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(uitrunc)((unsigned)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "uitrunc(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// unordered(P_numtype1, P_numtype2)    True if a comparison of x and y would be unordered
#ifdef BZ_HAVE_SYSTEM_V_MATH
template<class P_numtype1, class P_numtype2>
class _bz_unordered : public TwoOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef P_numtype2 T_numtype2;
    typedef int T_numtype;

    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)
    { return BZ_IEEEMATHFN_SCOPE(unordered)(x,y); }

    template<class T1, class T2>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a, const T2& b)
    {
        str += "unordered(";
        a.prettyPrint(str,format);
        str += ",";
        b.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// y0(P_numtype1)    Bessel function of the second kind, order zero
#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
class _bz_y0 : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(y0)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "y0(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif

// y1(P_numtype1)    Bessel function of the second kind, order one
#ifdef BZ_HAVE_IEEE_MATH
template<class P_numtype1>
class _bz_y1 : public OneOperandApplicativeTemplatesBase {
public:
    typedef P_numtype1 T_numtype1;
    typedef double T_numtype;

    static inline T_numtype apply(T_numtype1 x)
    { return BZ_IEEEMATHFN_SCOPE(y1)((double)x); }

    template<class T1>
    static void prettyPrint(string& str, prettyPrintFormat& format,
        const T1& a)
    {
        str += "y1(";
        a.prettyPrint(str,format);
        str += ")";
    }
};
#endif



BZ_NAMESPACE_END

#endif // BZ_MATHFUNC_H
