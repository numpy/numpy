#include <iostream.h>
#include <fstream.h> 
#include <iomanip.h>

using namespace std;

// abs(i), labs(l)                     Absolute value
// acos(d), acols(ld)                  Inverse cosine
// acosh(d)                            Inverse hyperbolic cosine
// asin(d), asinl(ld)                  Inverse sine
// asinh(d)                            Inverse hyperbolic sine
// atan(d), atanl(ld)                  Inverse tangent
// atan2(d,d), atan2l(ld,ld)           Inverse tangent
// atanh(d)                            Inverse hyperbolic tangent
// _class(d)                           Classification of floating-point values
// cbrt(x)                             Cube root
// ceil(d), ceill(ld)                  Smallest f-int not less than x
// cos(d), cosl(ld)                    Cosine
// cosh(d), coshl(ld)                  Hyperbolic cosine
// copysign(d,d)                       Return 1st arg with same sign as 2nd
// drem(x,x)                           IEEE remainder
// exp(d), expl(ld)                    Exponential
// expm1(d)                            Exp(x)-1     
// erf(d), erfl(ld)                    Error function
// erfc(d), erfcl(ld)                  Complementary error function
// fabs(d), fabsl(ld)                  Floating point absolute value
// int finite(d)                       Nonzero if finite
// floor(d), floor(ld)                 Largest f-int not greater than x
// fmod(d,d), fmodl(ld,ld)             Floating point remainder
// frexp(d, int* e)                    Break into mantissa/exponent  (*)
// frexpl(ld, int* e)                  Break into mantissa/exponent  (*)
// gammaFunc(d)                        Gamma function (** needs special 
//                                     implementation using lgamma)
// hypot(d,d)                          Hypotenuse: sqrt(x*x+y*y)
// int ilogb(d)                        Integer unbiased exponent
// int isnan(d)                        Nonzero if NaNS or NaNQ
// int itrunc(d)                       Truncate and convert to integer
// j0(d)                               Bessel function first kind, order 0
// j1(d)                               Bessel function first kind, order 1
// jn(int, double)                     Bessel function first kind, order i
// ldexp(d,i), ldexpl(ld,i)            Compute d * 2^i
// lgamma(d), lgammald(ld)             Log absolute gamma
// log(d), logl(ld)                    Natural logarithm
// logb(d)                             Unbiased exponent (IEEE)
// log1p(d)                            Compute log(1 + x)
// log10(d), log10l(ld)                Logarithm base 10
// modf(d, int* i), modfl(ld, int* i)  Break into integral/fractional part
// double nearest(double)              Nearest floating point integer
// nextafter(d, d)                     Next representable neighbor of 1st
//                                     in direction of 2nd
// pow(d,d), pow(ld,ld)                Computes x ^ y
// d remainder(d,d)                    IEEE remainder
// d rint(d)                           Round to f-integer (depends on mode)
// d rsqrt(d)                          Reciprocal square root
// d scalb(d,d)                        Return x * (2^y)
// sin(d), sinl(ld)                    Sine 
// sinh(d), sinhl(ld)                  Hyperbolic sine
// sqr(x)                              Return x * x
// sqrt(d), sqrtl(ld)                  Square root
// tan(d), tanl(ld)                    Tangent
// tanh(d), tanhl(ld)                  Hyperbolic tangent
// trunc(d)                            Nearest f-int in the direction of 0
// unsigned uitrunc(d)                 Truncate and convert to unsigned
// int unordered(d,d)                  Nonzero if comparison is unordered
// y0(d)                               Bessel function 2nd kind, order 0
// y1(d)                               Bessel function 2nd kind, order 1
// yn(i,d)                             Bessel function 2nd kind, order d

ofstream ofs;

const int ldflag = 1;
const int cflag = 2;
const int ieeeflag = 3;
const int bsdflag = 4;
const int cflag2 = 5;
const int nofuncflag = 6;

void one(char* applicName, char* specialization, char* funcName,
    char* returnType, char* comment, int flag=0, int noCastFlag=0)
{
    if (specialization != 0 && !strlen(specialization))
        specialization = 0;
    if (returnType != 0 && !strlen(returnType))
        returnType = 0;
    if (comment != 0 && !strlen(comment))
        comment = 0;

    ofs << "// " << applicName << "(";
    if (specialization)
        ofs << specialization;
    else
        ofs << "P_numtype1";
    ofs << ")";
    if (comment)
        ofs << "    " << comment;
    ofs << endl;

    if (flag == cflag)
        ofs << "#ifdef BZ_HAVE_COMPLEX_MATH" << endl;
    else if (flag == cflag2)
        ofs << "#ifdef BZ_HAVE_COMPLEX_MATH2" << endl;
    else if (flag == ieeeflag)
        ofs << "#ifdef BZ_HAVE_IEEE_MATH" << endl;
    else if (flag == bsdflag)
        ofs << "#ifdef BZ_HAVE_SYSTEM_V_MATH" << endl;
//    else if (flag == ldflag)
//        ofs << "#ifdef BZ_LONGDOUBLE128" << endl;

    if (!specialization)
    {
        ofs << "template<class P_numtype1>" << endl;
    }
    else {
        ofs << "template<>" << endl;
    }
    ofs << "class _bz_" << applicName;     
    if (specialization)
        ofs << "<" << specialization << ">";

    ofs << " : public OneOperandApplicativeTemplatesBase {" << endl;

    ofs << "public:" << endl;
    ofs << "    typedef ";
    if (specialization)
        ofs << specialization;
    else 
        ofs << "P_numtype1";
    ofs << " T_numtype1;" << endl;

    ofs << "    typedef ";
    if (returnType)
        ofs << returnType;
    else if (specialization)
        ofs << specialization;
    else
        ofs << "P_numtype1";
    ofs << " T_numtype;" << endl;

    ofs << endl << "    static inline T_numtype apply(T_numtype1 x)"
        << endl << "    { return ";

    if (noCastFlag == nofuncflag)
    {
        ofs << funcName;
    }
    else {
    if ((flag == cflag) || (flag == cflag2))
        ofs << "BZ_CMATHFN_SCOPE(";
    else if ((flag == ieeeflag) || (flag == bsdflag))
        ofs << "BZ_IEEEMATHFN_SCOPE(";
    else 
        ofs << "BZ_MATHFN_SCOPE(";
    
    ofs << funcName << ")(";
    if (specialization != 0)
        ofs << "(" << specialization << ")";
    else if ((returnType)&&(!noCastFlag))
        ofs << "(" << returnType << ")";

    ofs << "x)";
    }
    ofs << "; }" << endl;

    ofs << endl << "    template<class T1>" << endl
        << "    static void prettyPrint(string& str, prettyPrintFormat& format,"
        << endl
        << "        const T1& a)" << endl
        << "    {" << endl
        << "        str += \"" << funcName;
      ofs  << "(\";" << endl
        << "        a.prettyPrint(str,format);" << endl
        << "        str += \")\";" << endl
        << "    }" << endl
        << "};" << endl;

   if ((flag != ldflag) && (flag != 0))
        ofs << "#endif" << endl;

    ofs << endl;
}

void two(char* applicName, char* specialization, char* funcName,
    char* returnType, char* comment, int flag=0, int noCastFlag=0)
{
    if (specialization != 0 && !strlen(specialization))
        specialization = 0;
    if (returnType != 0 && !strlen(returnType))
        returnType = 0;
    if (comment != 0 && !strlen(comment))
        comment = 0;

    ofs << "// " << applicName << "(";
    if (specialization)
        ofs << specialization << ", " << specialization;
    else
        ofs << "P_numtype1, P_numtype2";
    ofs << ")";
    if (comment)
        ofs << "    " << comment;
    ofs << endl;

    if (flag == cflag)
        ofs << "#ifdef BZ_HAVE_COMPLEX_MATH" << endl;
    else if (flag == cflag2)
        ofs << "#ifdef BZ_HAVE_COMPLEX_MATH2" << endl;
    else if (flag == ieeeflag)
        ofs << "#ifdef BZ_HAVE_IEEE_MATH" << endl;
    else if (flag == bsdflag)
        ofs << "#ifdef BZ_HAVE_SYSTEM_V_MATH" << endl;
//    else if (flag == ldflag)
//        ofs << "#ifdef BZ_LONGDOUBLE128" << endl;

    if (!specialization)
    {
        ofs << "template<class P_numtype1, class P_numtype2>" << endl;
    }
    else {
        ofs << "template<>" << endl;
    }
    ofs << "class _bz_" << applicName;
    if (specialization)
        ofs << "<" << specialization  << ", " << specialization << " >";
    ofs << " : public TwoOperandApplicativeTemplatesBase {" << endl;

    ofs << "public:" << endl;
    ofs << "    typedef ";
    if (specialization)
        ofs << specialization;
    else
        ofs << "P_numtype1";
    ofs << " T_numtype1;" << endl;

    ofs << "    typedef ";
    if (specialization)
        ofs << specialization;
    else
        ofs << "P_numtype2";
    ofs << " T_numtype2;" << endl;

    ofs << "    typedef ";
    if (returnType)
        ofs << returnType;
    else if (specialization)
        ofs << specialization;
    else
        ofs << "BZ_PROMOTE(T_numtype1, T_numtype2)";
    ofs << " T_numtype;" << endl;

    ofs << endl << "    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)"
        << endl << "    { return ";

    if ((flag == cflag) || (flag == cflag2))
        ofs << "BZ_CMATHFN_SCOPE(";
    else if ((flag == ieeeflag) || (flag == bsdflag))
        ofs << "BZ_IEEEMATHFN_SCOPE(";
    else
        ofs << "BZ_MATHFN_SCOPE(";

    ofs << funcName << ")(";

    if (specialization != 0)
        ofs << "(" << specialization << ")";
    else if ((returnType) && (!noCastFlag))
        ofs << "(" << returnType << ")";

    ofs << "x,";
    if (specialization != 0)
        ofs << "(" << specialization << ")";
    else if ((returnType) && (!noCastFlag))
        ofs << "(" << returnType << ")";
    ofs << "y); }" << endl;

    ofs << endl << "    template<class T1, class T2>" << endl
        << "    static void prettyPrint(string& str, prettyPrintFormat& format,"
        << endl
        << "        const T1& a, const T2& b)" << endl
        << "    {" << endl
        << "        str += \"" << funcName;
      ofs  << "(\";" << endl
        << "        a.prettyPrint(str,format);" << endl
        << "        str += \",\";" << endl
        << "        b.prettyPrint(str,format);" << endl
        << "        str += \")\";" << endl
        << "    }" << endl;

    ofs << "};" << endl;

    if ((flag != ldflag) && (flag != 0))
        ofs << "#endif" << endl;

    ofs << endl;
}

int main()
{
    cout << "Generating <mathfunc.h>" << endl;

    ofs.open("mathfunc.h");

    ofs <<  
"// Generated: " << __FILE__ << " " << __DATE__ << " " << __TIME__ 
                 << endl << endl <<
"#ifndef BZ_MATHFUNC_H\n"
"#define BZ_MATHFUNC_H\n"
"\n"
"#ifndef BZ_APPLICS_H\n"
" #error <blitz/mathfunc.h> should be included via <blitz/applics.h>\n"
"#endif\n\n"
"\n"
"#ifndef BZ_PRETTYPRINT_H\n"
" #include <blitz/prettyprint.h>\n"
"#endif\n\n"
"BZ_NAMESPACE(blitz)\n\n";

    one("abs", 0, "abs", 0, "Absolute value");
    one("abs","long","labs","long", 0);
    one("abs","float"       ,"fabs",    "float",       0);

one("abs"    ,"double"      ,"fabs"    ,"double"       ,"");
one("abs"    ,"long double" ,"fabs"   ,"long double"  ,"", ldflag);
one("abs"    ,"complex<float> ", "abs","float", "", cflag);
one("abs"    ,"complex<double> ", "abs", "double", "", cflag);
one("abs"    ,"complex<long double> ", "abs", "long double", "", cflag);
one("acos"   ,""            ,"acos"    ,"double"       ,"Inverse cosine");
one("acos"   ,"float"       ,"acos"  ,"float"         ,"");
one("acos"   ,"long double" ,"acos"   ,"long double"  ,"", ldflag);
// one("acos"   ,"complex<float> ", "acos", "complex<float>", "", cflag2);
// one("acos"   ,"complex<double> ", "acos", "complex<double>", "", cflag2);
// one("acos", "complex<long double> ", "acos", "complex<long double>", "", cflag2);
one("acosh"  ,""            ,"acosh"   ,"double"       ,"Inverse hyperbolic cosine", ieeeflag);
one("asin"   ,""            ,"asin"    ,"double"       ,"Inverse sine");
one("asin",   "float",       "asin",    "float", "");
one("asin"   ,"long double" ,"asin"   ,"long double"  ,"", ldflag);
// one("asin"   ,"complex<float> ", "asin", "complex<float>", "", cflag2);
// one("asin"   ,"complex<double> ", "asin", "complex<double>", "", cflag2);
// one("asin", "complex<long double> ", "asin", "complex<long double>", "", cflag2);
one("asinh"  ,""            ,"asinh"   ,"double"       ,"Inverse hyperbolic sine", ieeeflag);
one("arg",   ""            ,"0"    ,0             ,"", cflag, nofuncflag);
one("arg",   "complex<float> ", "arg", "float", "", cflag, 0);
one("arg",   "complex<double> ", "arg", "double", "", cflag, 0);
one("arg",   "complex<long double> ", "arg", "long double", "", cflag, 0);
one("atan"   ,""            ,"atan"    ,"double"       ,"Inverse tangent");
one("atan",   "float",       "atan",    "float",        "");
one("atan"   ,"long double" ,"atan"   ,"long double"  ,"", ldflag);
// one("atan"   ,"complex<float> ", "atan", "complex<float>", "", cflag2);
// one("atan"   ,"complex<double> ", "atan", "complex<double>", "", cflag2);
// one("atan", "complex<long double> ", "atan", "complex<long double>", "", cflag2);
one("atanh"  ,""            ,"atanh"   ,"double"       ,"Inverse hyperbolic tangent", ieeeflag);
two("atan2"  ,""            ,"atan2"   ,"double"       ,"Inverse tangent");
two("atan2"  ,"float"       ,"atan2"   ,"float"        ,"");
two("atan2"  ,"long double" ,"atan2"   ,"long double"  ,"");
one("_class" ,""            ,"_class"  ,"int"          ,"Classification of float-point value (FP_xxx)", bsdflag,1);
one("cbrt"   ,""            ,"cbrt"    ,"double"       ,"Cube root", ieeeflag);
one("ceil"   ,""            ,"ceil"    ,"double"       ,"Ceiling");
one("ceil",   "float",       "ceil",    "float",       "");
one("ceil"   ,"long double" ,"ceil"   ,"long double"  ,"", ldflag);
one("conj",   ""            ,"conj"    ,0             ,"", cflag);
one("cos"    ,""            ,"cos"     ,"double"       ,"Cosine");
one("cos",    "float",       "cos",     "float",       "");
one("cos"    ,"long double" ,"cos"    ,"long double"  ,"", ldflag);
one("cos"   ,"complex<float> ", "cos", "complex<float>", "", cflag);
one("cos"   ,"complex<double> ", "cos", "complex<double>", "", cflag);
one("cos", "complex<long double> ", "cos", "complex<long double>", "", cflag);
two("copysign", ""          ,"copysign","double"       ,"", bsdflag);
one("cosh"   ,""            ,"cosh"    ,"double"       ,"Hyperbolic cosine");
one("cosh",   "float",       "cosh",    "float", "");
one("cosh"   ,"long double" ,"cosh"   ,"long double"  ,"", ldflag);
one("cosh"   ,"complex<float> ", "cosh", "complex<float>", "", cflag);
one("cosh"   ,"complex<double> ", "cosh", "complex<double>", "", cflag);
one("cosh", "complex<long double> ", "cosh", "complex<long double>", "", cflag);
two("drem"   ,""            ,"drem"    ,"double"       ,"Remainder", bsdflag);
one("exp"    ,""            ,"exp"     ,"double"       ,"Exponential");
one("exp",    "float",       "exp",     "float",       "");
one("exp"    ,"long double" ,"exp"    ,"long double"  ,"", ldflag      );
one("exp"   ,"complex<float> ", "exp", "complex<float>", "", cflag);
one("exp"   ,"complex<double> ", "exp", "complex<double>", "", cflag);
one("exp", "complex<long double> ", "exp", "complex<long double>", "", cflag);
one("expm1"  ,""            ,"expm1"   ,"double"       ,"Exp(x)-1", ieeeflag);
one("erf"    ,""            ,"erf"     ,"double"       ,"Error function", ieeeflag);
one("erfc"   ,""            ,"erfc"    ,"double"       ,"Complementary error function", ieeeflag);

// blitz-bugs/archive/0189.html
// one("finite" ,""            ,"finite"  ,"int"          ,"Nonzero if finite", ieeeflag,1);

one("floor"  ,""            ,"floor"   ,"double"       ,"Floor function");
one("floor",  "float",       "floor",   "float",        "");
one("floor"  ,"long double" ,"floor"   ,"long double"  ,"");
two("fmod"   ,""            ,"fmod"    ,"double"       ,"Modulo remainder",bsdflag);
two("hypot"  ,""            ,"hypot"   ,"double"       ,"sqrt(x*x+y*y)",bsdflag);
one("ilogb"  ,""            ,"ilogb"   ,"int"          ,"Integer unbiased exponent", bsdflag,1);
one("isnan"  ,""            ,"isnan"   ,"int"          ,"Nonzero if NaNS or NaNQ", ieeeflag,1);
one("itrunc" ,""            ,"itrunc"  ,"int"          ,"Truncate and convert to integer", bsdflag,1);
one("j0"     ,""            ,"j0"      ,"double"       ,"Bessel function first kind, order 0", ieeeflag);
one("j1"     ,""            ,"j1"      ,"double"       ,"Bessel function first kind, order 1", ieeeflag);
one("lgamma" ,""            ,"lgamma"  ,"double"       ,"Log absolute gamma", ieeeflag);
one("log"    ,""            ,"log"     ,"double"       ,"Natural logarithm");
one("log",    "float",       "log",     "float",        "");
one("log"    ,"long double" ,"log"     ,"long double"  ,"", ldflag);
one("log"   ,"complex<float> ", "log", "complex<float>", "", cflag);
one("log"   ,"complex<double> ", "log", "complex<double>", "", cflag);
one("log", "complex<long double> ", "log", "complex<long double>", "", cflag);
one("logb"   ,""            ,"logb"    ,"double"       ,"Unbiased exponent (IEEE)", ieeeflag);
one("log1p"  ,""            ,"log1p"   ,"double"       ,"Compute log(1 + x)", ieeeflag);
one("log10"  ,""            ,"log10"   ,"double"       ,"Logarithm base 10");
one("log10",  "float",       "log10",   "float",        "");
one("log10"  ,"long double" ,"log10"  ,"long double"  ,"", ldflag);
// one("log10"   ,"complex<float> ", "log10", "complex<float>", "", cflag2);
// one("log10"   ,"complex<double> ", "log10", "complex<double>", "", cflag2);
// one("log10", "complex<long double> ", "log10", "complex<long double>", "", cflag2);
one("nearest", ""           ,"nearest" ,"double"       ,"Nearest floating point integer", bsdflag);
two("nextafter", "",         "nextafter", "double",     "Next representable number after x towards y", bsdflag);

ofs <<
"template<class P_numtype>\n"
"class _bz_negate : public OneOperandApplicativeTemplatesBase {\n"
"public:\n"
"    typedef BZ_SIGNEDTYPE(P_numtype) T_numtype;\n\n"
"    static inline T_numtype apply(T_numtype x)\n"
"    { return -x; }\n"
"};\n\n";

one("norm",   ""            ,"norm"    ,0             ,"", cflag);

two("polar"  ,""            ,"polar"   ,"complex<T_numtype1>", "", cflag, 1);
two("pow"    ,""            ,"pow"     ,"double"       ,"Power");
two("pow"    ,"float"       ,"pow"     ,"float"        ,"");
two("pow"    ,"long double" ,"pow"     ,"long double"  ,"");
two("pow"    ,"complex<float>","pow"   ,"complex<float>" ,"",cflag);
two("pow"    ,"complex<double>","pow"  ,"complex<double>","",cflag);
two("pow"    ,"complex<long double>","pow","complex<long double>","",cflag);
two("remainder", "",         "remainder", "double",     "Remainder", bsdflag);

one("rint"   ,""            ,"rint"    ,"double"       ,"Round to floating point integer", ieeeflag);
one("rsqrt"  ,""            ,"rsqrt"   ,"double"       ,"Reciprocal square root", bsdflag);
two("scalb"  ,""            ,"scalb"   ,"double"       ,"x * (2**y)", bsdflag);
one("sin"    ,""            ,"sin"     ,"double"       ,"Sine");
one("sin",    "float",       "sin",     "float",       "");
one("sin"    ,"long double" ,"sin"    ,"long double"  ,"", ldflag);
one("sin"   ,"complex<float> ", "sin", "complex<float>", "", cflag);
one("sin"   ,"complex<double> ", "sin", "complex<double>", "", cflag);
one("sin", "complex<long double> ", "sin", "complex<long double>", "", cflag);
one("sinh"   ,""            ,"sinh"    ,"double"       ,"Hyperbolic sine");
one("sinh",   "float",       "sinh",    "float",        "");
one("sinh"   ,"long double" ,"sinh"   ,"long double"  ,"", ldflag);
one("sinh"   ,"complex<float> ", "sinh", "complex<float>", "", cflag);
one("sinh"   ,"complex<double> ", "sinh", "complex<double>", "", cflag);
one("sinh", "complex<long double> ", "sinh", "complex<long double>", "", cflag);

ofs << 
"template<class P_numtype>\n"
"class _bz_sqr : public OneOperandApplicativeTemplatesBase {\n"
"public:\n"
"    typedef P_numtype T_numtype;\n\n"
"    static inline T_numtype apply(T_numtype x)\n"
"    { return x*x; }\n"
"    template<class T1>\n"
"    static void prettyPrint(string& str, prettyPrintFormat& format,\n"
"        const T1& a)\n"
"    {\n"
"        str += \"sqr(\";\n"
"        a.prettyPrint(str,format);\n"
"        str += \")\";\n"
"    }\n"
"};\n\n"
"// Specialization of _bz_sqr for complex<T>\n"
"template<class T>\n"
"class _bz_sqr<complex<T> > : public OneOperandApplicativeTemplatesBase {\n"
"public:\n"
"    typedef complex<T> T_numtype;\n\n"
"    static inline T_numtype apply(T_numtype x)\n"
"    {\n"
"        T r = x.real();  T i = x.imag();\n"
"        return T_numtype(r*r-i*i, 2*r*i);\n"
"    }\n"
"    template<class T1>\n"
"    static void prettyPrint(string& str, prettyPrintFormat& format,\n"
"        const T1& a)\n"
"    {\n"
"        str += \"sqr(\";\n"
"        a.prettyPrint(str,format);\n"
"        str += \")\";\n"
"    }\n"
"};\n\n"
;

one("sqrt"   ,""            ,"sqrt"    ,"double"       ,"Square root");
one("sqrt",   "float",       "sqrt",    "float",        "");
one("sqrt"   ,"long double" ,"sqrt"   ,"long double"  ,"", ldflag);
one("sqrt"   ,"complex<float> ", "sqrt", "complex<float>", "", cflag);
one("sqrt"   ,"complex<double> ", "sqrt", "complex<double>", "", cflag);
one("sqrt", "complex<long double> ", "sqrt", "complex<long double>", "", cflag);
one("tan"    ,""            ,"tan"     ,"double"       ,"Tangent");
one("tan",    "float",       "tan",    "float",         "");
one("tan"    ,"long double" ,"tan"    ,"long double"  ,"");
one("tan"   ,"complex<float> ", "tan", "complex<float>", "", cflag);
one("tan"   ,"complex<double> ", "tan", "complex<double>", "", cflag);
one("tan", "complex<long double> ", "tan", "complex<long double>", "", cflag);
one("tanh"   ,""            ,"tanh"    ,"double"       ,"Hyperbolic tangent");
one("tanh",   "float",       "tanh",    "float",        "");
one("tanh"   ,"long double" ,"tanh"   ,"long double"  ,"", ldflag);
one("tanh"   ,"complex<float> ", "tanh", "complex<float>", "", cflag);
one("tanh"   ,"complex<double> ", "tanh", "complex<double>", "", cflag);
one("tanh", "complex<long double> ", "tanh", "complex<long double>", "", cflag);

// blitz-bugs/archive/0189.html
// one("trunc"  ,""            ,"trunc"   ,"double"       ,"Nearest floating integer in the direction of zero", ieeeflag);

one("uitrunc", ""           ,"uitrunc" ,"unsigned"     ,"Truncate and convert to unsigned", bsdflag);
two("unordered", "",         "unordered", "int",       "True if a comparison of x and y would be unordered", bsdflag,1);
one("y0"     ,""            ,"y0"      ,"double"       ,"Bessel function of the second kind, order zero", ieeeflag);
one("y1"     ,""            ,"y1"      ,"double"       ,"Bessel function of the second kind, order one", ieeeflag);

    ofs << endl << endl <<
"BZ_NAMESPACE_END\n\n"
"#endif // BZ_MATHFUNC_H\n";

    return 0;
}

