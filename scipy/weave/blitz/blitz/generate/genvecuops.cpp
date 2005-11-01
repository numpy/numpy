#include <fstream>
#include <iostream>
#include <iomanip>
#include "bzfstream.h"
#include "optuple.h"

const int ieeeflag = 1, bsdflag = 2, cmplxflag = 3;

bzofstream ofs("../vecuops.cc", 
    "Expression templates for vectors, unary functions",
    __FILE__,
    "BZ_VECUOPS_CC");

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
    else if (flag == cmplxflag)
        ofs << "#ifdef BZ_HAVE_COMPLEX_FCNS" << std::endl;

    OperandTuple operands(1);

    do {
        operands.printTemplates(ofs);
        ofs << std::endl << "inline" << std::endl
            << "_bz_VecExpr<_bz_VecExprUnaryOp<";
        operands.printIterators(ofs);
        ofs << "," << std::endl << "    _bz_" << apname << "<";
        operands[0].printNumtype(ofs);
        ofs << "> > >" << std::endl
            << fname << "(";
        operands.printArgumentList(ofs);
        ofs << ")" << std::endl
            << "{" << std::endl
            << "    typedef _bz_VecExprUnaryOp<";
        operands.printIterators(ofs);
        ofs << "," << std::endl << "        _bz_" << apname << "<";
        operands[0].printNumtype(ofs);
        ofs << "> > T_expr;" << std::endl << std::endl
            << "    return _bz_VecExpr<T_expr>(T_expr(";
        operands.printInitializationList(ofs);
        ofs << "));" << std::endl
            << "}" << std::endl << std::endl;

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
    else if (flag == cmplxflag)
        ofs << "#ifdef BZ_HAVE_COMPLEX_FCNS" << std::endl;

    OperandTuple operands(2);

    do {
        operands.printTemplates(ofs);
        ofs << std::endl << "inline" << std::endl
            << "_bz_VecExpr<_bz_VecExprOp<";
        operands.printIterators(ofs);
        ofs << "," << std::endl << "    _bz_" << apname << "<";
        operands[0].printNumtype(ofs);
        ofs << ",";
        operands[1].printNumtype(ofs);
        ofs << "> > >" << std::endl
            << fname << "(";
        operands.printArgumentList(ofs);
        ofs << ")" << std::endl
            << "{" << std::endl
            << "    typedef _bz_VecExprOp<";
        operands.printIterators(ofs);
        ofs << "," << std::endl << "        _bz_" << apname << "<";
        operands[0].printNumtype(ofs);
        ofs << ",";
        operands[1].printNumtype(ofs);
        ofs << "> > T_expr;" << std::endl << std::endl
            << "    return _bz_VecExpr<T_expr>(T_expr(";
        operands.printInitializationList(ofs);
        ofs << "));" << std::endl
            << "}" << std::endl << std::endl;
    } while (++operands);

    if (flag != 0)
        ofs << "#endif" << std::endl;

    ofs << std::endl;
}

int main()
{
    std::cout << "Generating <vecuops.cc>" << std::endl;

ofs << 
"#ifndef BZ_VECEXPR_H\n"
" #error <blitz/vecuops.cc> must be included via <blitz/vecexpr.h>\n"
"#endif // BZ_VECEXPR_H\n\n";

    ofs.beginNamespace();

    one("abs");
    one("acos");
    one("acosh", ieeeflag);
    one("asin");
    one("asinh", ieeeflag);
    one("atan");
    two("atan2");
    one("atanh", ieeeflag);
    one("_class", bsdflag);
    one("cbrt", ieeeflag);
    one("ceil");
    one("conj", cmplxflag);
    two("copysign", bsdflag);
    one("cos");
    one("cosh");
    two("drem", bsdflag);
    one("exp");
    one("expm1", ieeeflag);
    one("erf", ieeeflag);
    one("erfc", ieeeflag);
    one("fabs", 0, "abs");
//    one("finite", ieeeflag);
    one("floor");
    two("fmod");
    two("hypot", bsdflag);
    one("ilogb", ieeeflag);
    one("blitz_isnan", ieeeflag);
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

