/*
 * $Id$
 *
 * $Log$
 * Revision 1.1  2002/09/12 07:04:27  eric
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
 * Revision 1.2  2002/06/28 23:59:49  jcumming
 * Files for generating Matrix operators and math functions.
 *
 *
 */

#include <iostream.h>
#include <fstream.h>
#include "optuple2.h"

int main()
{
    cout << "Generating <matbops.h>" << endl;

    OperandTuple2 operands(2);

    ofstream ofs("matbops.h");
    ofs << "// Generated source file.  Do not edit." << endl;
    ofs << "// Created by: " << __FILE__ << " " << __DATE__ 
        << " " << __TIME__ << endl << endl;

    ofs << "#ifndef BZ_MATBOPS_H" << endl
        << "#define BZ_MATBOPS_H" << endl
        << endl << "BZ_NAMESPACE(blitz)" << endl << endl
        << "#ifndef BZ_MATEXPR_H" << endl
        << " #error <blitz/matbops.h> must be included via <blitz/matexpr.h>" 
        << endl << "#endif" << endl << endl;

    struct {
        const char* opSymbol;
        bool        nonIntOperands;
        bool        complexOperands;
        const char* opApplicName;
        const char* comment;
    } ops[] = {
     { "+",  true,  true,  "_bz_Add",            "Addition Operators" },
     { "-",  true,  true,  "_bz_Subtract",       "Subtraction Operators" },
     { "*",  true,  true,  "_bz_Multiply",       "Multiplication Operators" },
     { "/",  true,  true,  "_bz_Divide",         "Division Operators" },
     { "%",  false, false, "_bz_Mod",            "Modulus Operators" },
     { "^",  false, false, "_bz_BitwiseXOR",     "Bitwise XOR Operators" },
     { "&",  false, false, "_bz_BitwiseAnd",     "Bitwise And Operators" },
     { "|",  false, false, "_bz_BitwiseOr",      "Bitwise Or Operators" },
     { ">>", false, false, "_bz_ShiftRight",     "Shift right Operators" },
     { "<<", false, false, "_bz_ShiftLeft",      "Shift left Operators" },
     { ">",  true,  false, "_bz_Greater",        "Greater-than Operators" },
     { "<",  true,  false, "_bz_Less",           "Less-than Operators" },
     { ">=", true,  false, "_bz_GreaterOrEqual", "Greater or equal (>=) operators" },
     { "<=", true,  false, "_bz_LessOrEqual",    "Less or equal (<=) operators" },
     { "==", true,  true,  "_bz_Equal",          "Equality operators" },
     { "!=", true,  true,  "_bz_NotEqual",       "Not-equal operators" },
     { "&&", false, false, "_bz_LogicalAnd",     "Logical AND operators" },
     { "||", false, false, "_bz_LogicalOr",      "Logical OR operators" }
    };

    for (int i=0; i < 18; ++i)
    {
    ofs << "/****************************************************************************" << endl
        << " * " << ops[i].comment << endl
        << " ****************************************************************************/" << endl;

    operands.reset();

    do {
        if (ops[i].nonIntOperands == false)
        {
            if ((operands[0].isScalar() && !operands[0].isInteger())
             || (operands[1].isScalar() && !operands[1].isInteger()))
                continue;
        }

        // Matrix<P_numtype1> + _bz_MatExpr<P_expr2>

        if (operands.anyComplex())
            ofs << "#ifdef BZ_HAVE_COMPLEX" << endl;

        ofs << endl << "// ";
        operands[0].printName(ofs);
        ofs << " " << ops[i].opSymbol << " ";
        operands[1].printName(ofs);
        ofs << endl;

        operands.printTemplates(ofs);
        ofs << endl << "inline" << endl;

        // _bz_MatExpr<_bz_MatExprOp<MatRef<T_numtype1>,
        //     _bz_MatExpr<T_expr2>, _bz_Add<T_numtype1, _bz_typename T_expr2::T_numtype> > >
        ofs << "_bz_MatExpr<_bz_MatExprOp<";
        operands.printIterators(ofs, 1);
        ofs << "," << endl << "      " << ops[i].opApplicName << "<";
        operands[0].printNumtype(ofs);
        ofs << ", ";    
        operands[1].printNumtype(ofs);
        ofs << " > > >" << endl;
     
        // operator+(const Matrix<T_numtype1>& d1, _bz_MatExpr<T_expr2> d2)
        ofs << "operator" << ops[i].opSymbol << "(";
        operands.printArgumentList(ofs, 1);
        ofs << ")" << endl << "{" << endl;

        // typedef _bz_MatExprOp<MatRef<T_numtype1>,
        // _bz_MatExpr<T_expr2>, _bz_Add<T_numtype1, _bz_typename T_expr2::T_numtype> > T_expr;
        ofs << "    typedef _bz_MatExprOp<";
        operands.printIterators(ofs, 1);
        ofs << ", " << endl << "      " << ops[i].opApplicName << "<";
        operands[0].printNumtype(ofs);
        ofs << ", ";
        operands[1].printNumtype(ofs);
        ofs << "> > T_expr;" << endl << endl;

        // return _bz_MatExpr<T_expr>(T_expr(a.begin(), b));
        ofs << "    return _bz_MatExpr<T_expr>(T_expr(";
        operands.printInitializationList(ofs,1);
        ofs << "));" << endl << "}" << endl;

        if (operands.anyComplex())
            ofs << "#endif // BZ_HAVE_COMPLEX" << endl << endl;

    } while (++operands);
    
    }

    ofs << endl << "BZ_NAMESPACE_END" << endl << endl
        << "#endif // BZ_MATBOPS_H" << endl;

   cout << operands.numSpecializations() << " operators written." << endl;
}

