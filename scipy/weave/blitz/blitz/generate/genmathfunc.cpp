#include <iostream>
#include <string>
#include <fstream> 
#include <iomanip>

using namespace std;

// abs(i), labs(l)                     Absolute value
// acos(d), acosl(ld)                  Inverse cosine
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
// lgamma(d), lgammal(ld)              Log absolute gamma
// log(d), logl(ld)                    Natural logarithm
// logb(d)                             Unbiased exponent (IEEE)
// log1p(d)                            Compute log(1 + x)
// log10(d), log10l(ld)                Logarithm base 10
// modf(d, int* i), modfl(ld, int* i)  Break into integral/fractional part
// double nearest(double)              Nearest floating point integer
// nextafter(d, d)                     Next representable neighbor of 1st
//                                     in direction of 2nd
// pow(d,d), powl(ld,ld)               Computes x ^ y
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
const int cflag1 = 5;
const int cflag2 = 6;
const int nofuncflag = 7;

void one(const char* applicName, const char* specialization, const char* funcName,
    const char* returnType, const char* comment, int flag=0, int noCastFlag=0)
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
    ofs << std::endl;

    if (flag == cflag)
        ofs << "#ifdef BZ_HAVE_COMPLEX_FCNS" << std::endl;
    else if (flag == cflag1)
        ofs << "#ifdef BZ_HAVE_COMPLEX_MATH1" << std::endl;
    else if (flag == cflag2)
        ofs << "#ifdef BZ_HAVE_COMPLEX_MATH2" << std::endl;
    else if (flag == ieeeflag)
        ofs << "#ifdef BZ_HAVE_IEEE_MATH" << std::endl;
    else if (flag == bsdflag)
        ofs << "#ifdef BZ_HAVE_SYSTEM_V_MATH" << std::endl;
//    else if (flag == ldflag)
//        ofs << "#ifdef BZ_LONGDOUBLE128" << std::endl;

    if (!specialization)
    {
        ofs << "template<typename P_numtype1>" << std::endl;
    }
    else {
        ofs << "template<>" << std::endl;
    }
    ofs << "class _bz_" << applicName;     
    if (specialization)
        ofs << "<" << specialization << ">";

    ofs << " : public OneOperandApplicativeTemplatesBase {" << std::endl;

    ofs << "public:" << std::endl;
    ofs << "    typedef ";
    if (specialization)
        ofs << specialization;
    else 
        ofs << "P_numtype1";
    ofs << " T_numtype1;" << std::endl;

    ofs << "    typedef ";
    if (returnType)
        ofs << returnType;
    else if (specialization)
        ofs << specialization;
    else
        ofs << "P_numtype1";
    ofs << " T_numtype;" << std::endl;

    if (strcmp(applicName,"blitz_isnan") == 0) // Special case nan
    {
        ofs << std::endl << "    static inline T_numtype apply(T_numtype1 x)"
            << std::endl << "    {" << std::endl;
        ofs << "#ifdef isnan" << std::endl;
        ofs << "        "
            << "// Some platforms define isnan as a macro, which causes the"
            << std::endl << "        "
            << "// BZ_IEEEMATHFN_SCOPE macro to break." << std::endl;
        ofs << "        return isnan(x);" << std::endl;
        ofs << "#else" << std::endl;
        ofs << "        return BZ_IEEEMATHFN_SCOPE(isnan)(x);" << std::endl;
        ofs << "#endif" << std::endl << "    }" << std::endl;
    }
    else 
    {
        ofs << std::endl << "    static inline T_numtype apply(T_numtype1 x)"
            << std::endl << "    { return ";

        if (noCastFlag == nofuncflag)
        {
            ofs << funcName;
        }
        else {
        if ((flag == cflag) || (flag == cflag1) || (flag == cflag2))
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
        ofs << "; }" << std::endl;
    }

    ofs << std::endl << "    template<typename T1>" << std::endl
        << "    static void prettyPrint(BZ_STD_SCOPE(string) &str, prettyPrintFormat& format,"
        << std::endl
        << "        const T1& a)" << std::endl
        << "    {" << std::endl
        << "        str += \"" << funcName;
      ofs  << "(\";" << std::endl
        << "        a.prettyPrint(str,format);" << std::endl
        << "        str += \")\";" << std::endl
        << "    }" << std::endl
        << "};" << std::endl;

   if ((flag != ldflag) && (flag != 0))
        ofs << "#endif" << std::endl;

    ofs << std::endl;
}

void two(const char* applicName, const char* specialization, const char* funcName,
    const char* returnType, const char* comment, int flag=0, int noCastFlag=0)
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
    ofs << std::endl;

    if (flag == cflag)
        ofs << "#ifdef BZ_HAVE_COMPLEX_FCNS" << std::endl;
    else if (flag == cflag1)
        ofs << "#ifdef BZ_HAVE_COMPLEX_MATH1" << std::endl;
    else if (flag == cflag2)
        ofs << "#ifdef BZ_HAVE_COMPLEX_MATH2" << std::endl;
    else if (flag == ieeeflag)
        ofs << "#ifdef BZ_HAVE_IEEE_MATH" << std::endl;
    else if (flag == bsdflag)
        ofs << "#ifdef BZ_HAVE_SYSTEM_V_MATH" << std::endl;
//    else if (flag == ldflag)
//        ofs << "#ifdef BZ_LONGDOUBLE128" << std::endl;

    if (!specialization)
    {
        ofs << "template<typename P_numtype1, typename P_numtype2>" << std::endl;
    }
    else {
        ofs << "template<>" << std::endl;
    }
    ofs << "class _bz_" << applicName;
    if (specialization)
        ofs << "<" << specialization  << ", " << specialization << " >";
    ofs << " : public TwoOperandApplicativeTemplatesBase {" << std::endl;

    ofs << "public:" << std::endl;
    ofs << "    typedef ";
    if (specialization)
        ofs << specialization;
    else
        ofs << "P_numtype1";
    ofs << " T_numtype1;" << std::endl;

    ofs << "    typedef ";
    if (specialization)
        ofs << specialization;
    else
        ofs << "P_numtype2";
    ofs << " T_numtype2;" << std::endl;

    ofs << "    typedef ";
    if (returnType)
        ofs << returnType;
    else if (specialization)
        ofs << specialization;
    else
        ofs << "BZ_PROMOTE(T_numtype1, T_numtype2)";
    ofs << " T_numtype;" << std::endl;

    ofs << std::endl << "    static inline T_numtype apply(T_numtype1 x, T_numtype2 y)"
        << std::endl << "    { return ";

    if ((flag == cflag) || (flag == cflag1) || (flag == cflag2))
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
    ofs << "y); }" << std::endl;

    ofs << std::endl << "    template<typename T1, typename T2>" << std::endl
        << "    static void prettyPrint(BZ_STD_SCOPE(string) &str, prettyPrintFormat& format,"
        << std::endl
        << "        const T1& a, const T2& b)" << std::endl
        << "    {" << std::endl
        << "        str += \"" << funcName;
      ofs  << "(\";" << std::endl
        << "        a.prettyPrint(str,format);" << std::endl
        << "        str += \",\";" << std::endl
        << "        b.prettyPrint(str,format);" << std::endl
        << "        str += \")\";" << std::endl
        << "    }" << std::endl;

    ofs << "};" << std::endl;

    if ((flag != ldflag) && (flag != 0))
        ofs << "#endif" << std::endl;

    ofs << std::endl;
}

int main()
{
    std::cout << "Generating <mathfunc.h>" << std::endl;

    ofs.open("../mathfunc.h");

    ofs <<  
"// Generated: " << __FILE__ << " " << __DATE__ << " " << __TIME__ 
                 << std::endl << std::endl <<
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
one("acos"   ,"complex<float> ", "acos", "complex<float>", "", cflag2);
one("acos"   ,"complex<double> ", "acos", "complex<double>", "", cflag2);
one("acos", "complex<long double> ", "acos", "complex<long double>", "", cflag2);
one("acosh"  ,""            ,"acosh"   ,"double"       ,"Inverse hyperbolic cosine", ieeeflag);
one("asin"   ,""            ,"asin"    ,"double"       ,"Inverse sine");
one("asin",   "float",       "asin",    "float", "");
one("asin"   ,"long double" ,"asin"   ,"long double"  ,"", ldflag);
one("asin"   ,"complex<float> ", "asin", "complex<float>", "", cflag2);
one("asin"   ,"complex<double> ", "asin", "complex<double>", "", cflag2);
one("asin", "complex<long double> ", "asin", "complex<long double>", "", cflag2);
one("asinh"  ,""            ,"asinh"   ,"double"       ,"Inverse hyperbolic sine", ieeeflag);
one("arg",   ""            ,"0"    ,0             ,"", cflag, nofuncflag);
one("arg",   "complex<float> ", "arg", "float", "", cflag, 0);
one("arg",   "complex<double> ", "arg", "double", "", cflag, 0);
one("arg",   "complex<long double> ", "arg", "long double", "", cflag, 0);
one("atan"   ,""            ,"atan"    ,"double"       ,"Inverse tangent");
one("atan",   "float",       "atan",    "float",        "");
one("atan"   ,"long double" ,"atan"   ,"long double"  ,"", ldflag);
one("atan"   ,"complex<float> ", "atan", "complex<float>", "", cflag2);
one("atan"   ,"complex<double> ", "atan", "complex<double>", "", cflag2);
one("atan", "complex<long double> ", "atan", "complex<long double>", "", cflag2);
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
one("cos"   ,"complex<float> ", "cos", "complex<float>", "", cflag1);
one("cos"   ,"complex<double> ", "cos", "complex<double>", "", cflag1);
 ofs << "#ifndef __PGI\n";
one("cos", "complex<long double> ", "cos", "complex<long double>", "", cflag1);
 ofs << "#endif\n";
two("copysign", ""          ,"copysign","double"       ,"", bsdflag);
one("cosh"   ,""            ,"cosh"    ,"double"       ,"Hyperbolic cosine");
one("cosh",   "float",       "cosh",    "float", "");
one("cosh"   ,"long double" ,"cosh"   ,"long double"  ,"", ldflag);
one("cosh"   ,"complex<float> ", "cosh", "complex<float>", "", cflag1);
one("cosh"   ,"complex<double> ", "cosh", "complex<double>", "", cflag1);
 ofs << "#ifndef __PGI\n";
one("cosh", "complex<long double> ", "cosh", "complex<long double>", "", cflag1);
 ofs << "#endif\n";
two("drem"   ,""            ,"drem"    ,"double"       ,"Remainder", bsdflag);
one("exp"    ,""            ,"exp"     ,"double"       ,"Exponential");
one("exp",    "float",       "exp",     "float",       "");
one("exp"    ,"long double" ,"exp"    ,"long double"  ,"", ldflag      );
one("exp"   ,"complex<float> ", "exp", "complex<float>", "", cflag1);
one("exp"   ,"complex<double> ", "exp", "complex<double>", "", cflag1);
 ofs << "#ifndef __PGI\n";
one("exp", "complex<long double> ", "exp", "complex<long double>", "", cflag1);
 ofs << "#endif\n";
one("expm1"  ,""            ,"expm1"   ,"double"       ,"Exp(x)-1", ieeeflag);
one("erf"    ,""            ,"erf"     ,"double"       ,"Error function", ieeeflag);
one("erfc"   ,""            ,"erfc"    ,"double"       ,"Complementary error function", ieeeflag);

// blitz-bugs/archive/0189.html
// one("finite" ,""            ,"finite"  ,"int"          ,"Nonzero if finite", ieeeflag,1);

one("floor"  ,""            ,"floor"   ,"double"       ,"Floor function");
one("floor",  "float",       "floor",   "float",        "");
one("floor"  ,"long double" ,"floor"   ,"long double"  ,"");
two("fmod"   ,""            ,"fmod"    ,"double"       ,"Modulo remainder");
two("hypot"  ,""            ,"hypot"   ,"double"       ,"sqrt(x*x+y*y)",bsdflag);
one("ilogb"  ,""            ,"ilogb"   ,"int"          ,"Integer unbiased exponent", ieeeflag,1);
one("blitz_isnan"  ,""            ,"blitz_isnan"   ,"int"          ,"Nonzero if NaNS or NaNQ", ieeeflag,nofuncflag);
one("itrunc" ,""            ,"itrunc"  ,"int"          ,"Truncate and convert to integer", bsdflag,1);
one("j0"     ,""            ,"j0"      ,"double"       ,"Bessel function first kind, order 0", ieeeflag);
one("j1"     ,""            ,"j1"      ,"double"       ,"Bessel function first kind, order 1", ieeeflag);
one("lgamma" ,""            ,"lgamma"  ,"double"       ,"Log absolute gamma", ieeeflag);
one("log"    ,""            ,"log"     ,"double"       ,"Natural logarithm");
one("log",    "float",       "log",     "float",        "");
one("log"    ,"long double" ,"log"     ,"long double"  ,"", ldflag);
one("log"   ,"complex<float> ", "log", "complex<float>", "", cflag1);
one("log"   ,"complex<double> ", "log", "complex<double>", "", cflag1);
 ofs << "#ifndef __PGI\n";
one("log", "complex<long double> ", "log", "complex<long double>", "", cflag1);
 ofs << "#endif\n";
one("logb"   ,""            ,"logb"    ,"double"       ,"Unbiased exponent (IEEE)", ieeeflag);
one("log1p"  ,""            ,"log1p"   ,"double"       ,"Compute log(1 + x)", ieeeflag);
one("log10"  ,""            ,"log10"   ,"double"       ,"Logarithm base 10");
one("log10",  "float",       "log10",   "float",        "");
one("log10"  ,"long double" ,"log10"  ,"long double"  ,"", ldflag);
one("log10"   ,"complex<float> ", "log10", "complex<float>", "", cflag2);
one("log10"   ,"complex<double> ", "log10", "complex<double>", "", cflag2);
one("log10", "complex<long double> ", "log10", "complex<long double>", "", cflag2);
one("nearest", ""           ,"nearest" ,"double"       ,"Nearest floating point integer", bsdflag);
two("nextafter", "",         "nextafter", "double",     "Next representable number after x towards y", bsdflag);

ofs <<
"template<typename P_numtype>\n"
"class _bz_negate : public OneOperandApplicativeTemplatesBase {\n"
"public:\n"
"    typedef BZ_SIGNEDTYPE(P_numtype) T_numtype;\n\n"
"    static inline T_numtype apply(T_numtype x)\n"
"    { return -x; }\n\n"
"        template<typename T1>\n"
"        "
"static void prettyPrint(BZ_STD_SCOPE(string) &str, prettyPrintFormat& format, const T1& a)\n"
"        {\n"
"                str += \"-(\";\n"
"                       a.prettyPrint(str,format);\n"
"                       str += \")\";\n"
"        }\n"
"};\n\n"
;

one("norm",   ""            ,"norm"    ,0             ,"", cflag);

two("polar"  ,""            ,"polar"   ,"complex<T_numtype1>", "", cflag, 1);
two("pow"    ,""            ,"pow"     ,"double"       ,"Power");
 ofs << "#ifndef __PGI\n";
two("pow"    ,"float"       ,"pow"     ,"float"        ,"");
 ofs << "#endif\n";
two("pow"    ,"long double" ,"pow"     ,"long double"  ,"");
two("pow"    ,"complex<float>","pow"   ,"complex<float>" ,"",cflag1);
two("pow"    ,"complex<double>","pow"  ,"complex<double>","",cflag1);
 ofs << "#ifndef __PGI\n";
two("pow"    ,"complex<long double>","pow","complex<long double>","",cflag1);
 ofs << "#endif\n";
two("remainder", "",         "remainder", "double",     "Remainder", bsdflag);

one("rint"   ,""            ,"rint"    ,"double"       ,"Round to floating point integer", ieeeflag);
one("rsqrt"  ,""            ,"rsqrt"   ,"double"       ,"Reciprocal square root", bsdflag);
two("scalb"  ,""            ,"scalb"   ,"double"       ,"x * (2**y)", bsdflag);
one("sin"    ,""            ,"sin"     ,"double"       ,"Sine");
one("sin",    "float",       "sin",     "float",       "");
one("sin"    ,"long double" ,"sin"    ,"long double"  ,"", ldflag);
one("sin"   ,"complex<float> ", "sin", "complex<float>", "", cflag1);
one("sin"   ,"complex<double> ", "sin", "complex<double>", "", cflag1);
 ofs << "#ifndef __PGI\n";
one("sin", "complex<long double> ", "sin", "complex<long double>", "", cflag1);
 ofs << "#endif\n";
one("sinh"   ,""            ,"sinh"    ,"double"       ,"Hyperbolic sine");
one("sinh",   "float",       "sinh",    "float",        "");
one("sinh"   ,"long double" ,"sinh"   ,"long double"  ,"", ldflag);
one("sinh"   ,"complex<float> ", "sinh", "complex<float>", "", cflag1);
one("sinh"   ,"complex<double> ", "sinh", "complex<double>", "", cflag1);
 ofs << "#ifndef __PGI\n";
one("sinh", "complex<long double> ", "sinh", "complex<long double>", "", cflag1);
 ofs << "#endif\n";

ofs << 
"template<typename P_numtype>\n"
"class _bz_sqr : public OneOperandApplicativeTemplatesBase {\n"
"public:\n"
"    typedef P_numtype T_numtype;\n\n"
"    static inline T_numtype apply(T_numtype x)\n"
"    { return x*x; }\n"
"    template<typename T1>\n"
"    static void prettyPrint(BZ_STD_SCOPE(string) &str, prettyPrintFormat& format,\n"
"        const T1& a)\n"
"    {\n"
"        str += \"sqr(\";\n"
"        a.prettyPrint(str,format);\n"
"        str += \")\";\n"
"    }\n"
"};\n\n"
"#ifdef BZ_HAVE_COMPLEX\n"
"// Specialization of _bz_sqr for complex<T>\n"
"template<typename T>\n"
"class _bz_sqr<complex<T> > : public OneOperandApplicativeTemplatesBase {\n"
"public:\n"
"    typedef complex<T> T_numtype;\n\n"
"    static inline T_numtype apply(T_numtype x)\n"
"    {\n"
"        T r = x.real();  T i = x.imag();\n"
"        return T_numtype(r*r-i*i, 2*r*i);\n"
"    }\n"
"    template<typename T1>\n"
"    static void prettyPrint(BZ_STD_SCOPE(string) &str, prettyPrintFormat& format,\n"
"        const T1& a)\n"
"    {\n"
"        str += \"sqr(\";\n"
"        a.prettyPrint(str,format);\n"
"        str += \")\";\n"
"    }\n"
"};\n"
"#endif\n\n"
;

one("sqrt"   ,""            ,"sqrt"    ,"double"       ,"Square root");
one("sqrt",   "float",       "sqrt",    "float",        "");
one("sqrt"   ,"long double" ,"sqrt"   ,"long double"  ,"", ldflag);
one("sqrt"   ,"complex<float> ", "sqrt", "complex<float>", "", cflag1);
one("sqrt"   ,"complex<double> ", "sqrt", "complex<double>", "", cflag1);
 ofs << "#ifndef __PGI\n";
one("sqrt", "complex<long double> ", "sqrt", "complex<long double>", "", cflag1);
 ofs << "#endif\n";
one("tan"    ,""            ,"tan"     ,"double"       ,"Tangent");
one("tan",    "float",       "tan",    "float",         "");
one("tan"    ,"long double" ,"tan"    ,"long double"  ,"");
one("tan"   ,"complex<float> ", "tan", "complex<float>", "", cflag1);
one("tan"   ,"complex<double> ", "tan", "complex<double>", "", cflag1);
 ofs << "#ifndef __PGI\n";
one("tan", "complex<long double> ", "tan", "complex<long double>", "", cflag1);
 ofs << "#endif\n";
one("tanh"   ,""            ,"tanh"    ,"double"       ,"Hyperbolic tangent");
one("tanh",   "float",       "tanh",    "float",        "");
one("tanh"   ,"long double" ,"tanh"   ,"long double"  ,"", ldflag);
one("tanh"   ,"complex<float> ", "tanh", "complex<float>", "", cflag1);
one("tanh"   ,"complex<double> ", "tanh", "complex<double>", "", cflag1);
 ofs << "#ifndef __PGI\n";
one("tanh", "complex<long double> ", "tanh", "complex<long double>", "", cflag1);
 ofs << "#endif\n";

// blitz-bugs/archive/0189.html
// one("trunc"  ,""            ,"trunc"   ,"double"       ,"Nearest floating integer in the direction of zero", ieeeflag);

one("uitrunc", ""           ,"uitrunc" ,"unsigned"     ,"Truncate and convert to unsigned", bsdflag);
two("unordered", "",         "unordered", "int",       "True if a comparison of x and y would be unordered", bsdflag,1);
one("y0"     ,""            ,"y0"      ,"double"       ,"Bessel function of the second kind, order zero", ieeeflag);
one("y1"     ,""            ,"y1"      ,"double"       ,"Bessel function of the second kind, order one", ieeeflag);

    ofs << std::endl << std::endl <<
"BZ_NAMESPACE_END\n\n"
"#endif // BZ_MATHFUNC_H\n";

    return 0;
}

