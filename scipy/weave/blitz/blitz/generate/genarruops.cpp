#include "bzfstream.h"
#include "arroptuple.h"

bzofstream ofs("../array/uops.cc", 
    "Expression templates for arrays, unary functions", __FILE__,
    "BZ_ARRAYUOPS_CC");

const int ieeeflag = 1, bsdflag = 2;

void one(const char* fname, int flag=0, const char* apname = 0)
{
    if (apname == 0)
        apname = fname;

    ofs << "/****************************************************************************" << std::endl
        << " * " << fname << std::endl
        << " ****************************************************************************/" << std::endl << std::endl;

    if (flag == ieeeflag)
        ofs << "#ifdef BZ_HAVE_IEEE_MATH" << std::endl;
    else if (flag == bsdflag)
        ofs << "#ifdef BZ_HAVE_SYSTEM_V_MATH" << std::endl;

    OperandTuple operands(1);

    do {
        if (operands.anyComplex())
            ofs << "#ifdef BZ_HAVE_COMPLEX" << std::endl;

        operands.printTemplates(ofs);
        ofs << std::endl << "inline" << std::endl
            << "_bz_ArrayExpr<_bz_ArrayExprUnaryOp<";
        operands.printIterators(ofs);
        ofs << "," << std::endl << "    _bz_" << apname << "<";
        operands[0].printNumtype(ofs);
        ofs << "> > >" << std::endl
            << fname << "(";
        operands.printArgumentList(ofs);
        ofs << ")" << std::endl
            << "{" << std::endl;

        ofs << "    return _bz_ArrayExprUnaryOp<";
        operands.printIterators(ofs);
        ofs << "," << std::endl << "    _bz_" << apname << "<";
        operands[0].printNumtype(ofs);
        ofs << "> >(";
        operands.printInitializationList(ofs);
        ofs << ");" << std::endl
            << "}" << std::endl;

        if (operands.anyComplex())
            ofs << "#endif // BZ_HAVE_COMPLEX" << std::endl;

        ofs << std::endl;

    } while (++operands);
    
    if (flag != 0)
        ofs << "#endif" << std::endl;

    ofs << std::endl;
}

void two(const char* fname, int flag=0, const char* apname = 0)
{
    if (apname == 0)
        apname = fname;

    ofs << "/****************************************************************************" << std::endl
        << " * " << fname << std::endl
        << " ****************************************************************************/" << std::endl << std::endl;

    if (flag == ieeeflag)
        ofs << "#ifdef BZ_HAVE_IEEE_MATH" << std::endl;
    else if (flag == bsdflag)
        ofs << "#ifdef BZ_HAVE_SYSTEM_V_MATH" << std::endl;

    OperandTuple operands(2);

    do {
        if (operands[0].isScalar() && operands[1].isScalar())
            continue;

        if ((operands[0].isScalar() && operands[0].isInteger())
         || (operands[1].isScalar() && operands[1].isInteger()))
            continue;

        if (operands.anyComplex())
            ofs << "#ifdef BZ_HAVE_COMPLEX" << std::endl;

        operands.printTemplates(ofs);
        ofs << std::endl << "inline" << std::endl
            << "_bz_ArrayExpr<_bz_ArrayExprBinaryOp<";
        operands.printIterators(ofs);
        ofs << "," << std::endl << "    _bz_" << apname << "<";
        operands[0].printNumtype(ofs);
        ofs << ",";
        operands[1].printNumtype(ofs);
        ofs << "> > >" << std::endl
            << fname << "(";
        operands.printArgumentList(ofs);
        ofs << ")" << std::endl
            << "{" << std::endl;

        ofs << "    return _bz_ArrayExprBinaryOp<";
        operands.printIterators(ofs);
        ofs << "," << std::endl << "    _bz_" << apname << "<";
        operands[0].printNumtype(ofs);
        ofs << ",";
        operands[1].printNumtype(ofs);
        ofs << "> >(";
        operands.printInitializationList(ofs);
        ofs << ");" << std::endl
            << "}" << std::endl << std::endl;

        if (operands.anyComplex())
            ofs << "#endif // BZ_HAVE_COMPLEX" << std::endl << std::endl;

    } while (++operands);

    if (flag != 0)
        ofs << "#endif" << std::endl;

    ofs << std::endl;
}

int main()
{
    std::cout << "Generating <array/uops.cc>" << std::endl;

ofs << 
"#ifndef BZ_ARRAYEXPR_H\n"
" #error <blitz/array/uops.cc> must be included after <blitz/arrayexpr.h>\n"
"#endif // BZ_ARRAYEXPR_H\n\n";

    ofs.beginNamespace();

    one("abs");
    one("acos");
    one("acosh", ieeeflag);
    one("asin");
    one("asinh", ieeeflag);
    one("atan");
    one("atanh", ieeeflag);
    two("atan2");
    one("_class", bsdflag);
    one("cbrt", ieeeflag);
    one("ceil");
    one("cexp");
    one("cos");
    one("cosh");
    two("copysign", bsdflag);
    one("csqrt");
    two("drem", bsdflag);
    one("exp");
    one("expm1", ieeeflag);
    one("erf", ieeeflag);
    one("erfc", ieeeflag);
    one("fabs", 0, "abs");
//    one("finite", ieeeflag);
    one("floor");
    two("fmod", bsdflag);
    two("hypot", bsdflag);
    one("ilogb", bsdflag);
    one("isnan", ieeeflag);
    one("itrunc", bsdflag);
    one("j0", ieeeflag);
    one("j1", ieeeflag);
    one("lgamma", ieeeflag);
    one("log");
    one("logb", ieeeflag);
    one("log1p", ieeeflag);
    one("log10");
    one("nearest", bsdflag);
    two("nextafter", bsdflag);
    two("pow");
    one("pow2");
    one("pow3");
    one("pow4");
    one("pow5");
    one("pow6");
    one("pow7");
    one("pow8");
    two("remainder", bsdflag);
    one("rint", ieeeflag);
    one("rsqrt", bsdflag);
    two("scalb", bsdflag);
    one("sin");
    one("sinh");
    one("sqr");
    one("sqrt");
    one("tan");
    one("tanh");
//    one("trunc", ieeeflag);
    one("uitrunc", bsdflag);
    two("unordered", bsdflag);
    one("y0", ieeeflag);
    one("y1", ieeeflag);

    return 0;
}

