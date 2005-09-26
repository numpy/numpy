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
 * Revision 1.1  2000/06/19 13:02:48  tveldhui
 * Initial source check-in; added files not usually released in the
 * distribution.
 *
 *
 */

#include <iostream.h>
#include <fstream.h>
#include "bzfstream.h"
#include "arroptuple.h"

#ifdef NO_BOOL
 #define bool int
 #define true 1
 #define false 0
#endif

int main()
{
    cout << "Generating <arraybops.cc>" << endl;

    OperandTuple operands(2);

    bzofstream ofs("arraybops.cc", "Array expression templates (2 operands)",
        __FILE__, "BZ_ARRAYBOPS_CC");

    ofs << "#ifndef BZ_ARRAYEXPR_H" << endl
        << " #error <blitz/arraybops.cc> must be included after <blitz/arrayexpr.h>" 
        << endl << "#endif" << endl << endl;

    ofs.beginNamespace();

    struct {
        const char* opSymbol;
        bool        nonIntOperands;
        bool        complexOperands;
        const char* opApplicName;
        const char* comment;
    } ops[] = {
     { "+",  true,  true,  "Add",            "Addition Operators" },
     { "-",  true,  true,  "Subtract",       "Subtraction Operators" },
     { "*",  true,  true,  "Multiply",       "Multiplication Operators" },
     { "/",  true,  true,  "Divide",         "Division Operators" },
     { "%",  false, false, "Modulo",         "Modulus Operators" },
     { ">",  true,  false, "Greater",        "Greater-than Operators" },
     { "<",  true,  false, "Less",           "Less-than Operators" },
     { ">=", true,  false, "GreaterOrEqual", "Greater or equal (>=) operators" },
     { "<=", true,  false, "LessOrEqual",    "Less or equal (<=) operators" },
     { "==", true,  true,  "Equal",          "Equality operators" },
     { "!=", true,  true,  "NotEqual",       "Not-equal operators" },
     { "&&", false, false, "LogicalAnd",     "Logical AND operators" },
     { "||", false, false, "LogicalOr",      "Logical OR operators" },


     { "^",  false, false, "BitwiseXor",     "Bitwise XOR Operators" },
     { "&",  false, false, "BitwiseAnd",     "Bitwise And Operators" },
     { "|",  false, false, "BitwiseOr",      "Bitwise Or Operators" },
     { ">>", false, false, "ShiftRight",     "Shift right Operators" },
     { "<<", false, false, "ShiftLeft",      "Shift left Operators" }
    };

    const int numOperators = 18;   // Should be 18

    for (int i=0; i < numOperators; ++i)
    {
    ofs << "/****************************************************************************" << endl
        << " * " << ops[i].comment << endl
        << " ****************************************************************************/" << endl;

    operands.reset();

    do {
        // Can't declare operator+(int,Range) or operator+(Range,int)
        // because these would conflict with the versions defined
        // in range.h.  Also, the versions in range.h will be
        // much faster.
        if (operands[0].isScalar() && operands[0].isInteger()
            && operands[1].isRange())
                continue;
        if (operands[1].isScalar() && operands[1].isInteger()
            && operands[0].isRange())
                continue;

        if (ops[i].nonIntOperands == false)
        {
            if ((operands[0].isScalar() && !operands[0].isInteger())
             || (operands[1].isScalar() && !operands[1].isInteger()))
                continue;
        }

        ofs << endl;

        if (operands.anyComplex())
            ofs << "#ifdef BZ_HAVE_COMPLEX" << endl;

        ofs << "// ";
        operands[0].printName(ofs);
        ofs << " " << ops[i].opSymbol << " ";
        operands[1].printName(ofs);
        ofs << endl;

        operands.printTemplates(ofs);
        ofs << endl << "inline" << endl;

        ofs << "_bz_ArrayExpr<_bz_ArrayExprOp<";
        operands.printIterators(ofs, 1);
        ofs << "," << endl << "      " << ops[i].opApplicName << "<";
        operands[0].printNumtype(ofs);
        ofs << ", ";    
        operands[1].printNumtype(ofs);
        ofs << " > > >" << endl;
     
        // operator+(const Vector<T_numtype1>& d1, _bz_VecExpr<T_expr2> d2)
        ofs << "operator" << ops[i].opSymbol << "(";
        operands.printArgumentList(ofs, 1);
        ofs << ")" << endl << "{" << endl;

        ofs << "    return _bz_ArrayExprOp<";
        operands.printIterators(ofs, 1);
        ofs << ", " << endl << "      " << ops[i].opApplicName << "<";
        operands[0].printNumtype(ofs);
        ofs << ", ";
        operands[1].printNumtype(ofs);
        ofs << "> >" << endl;
        ofs << "      (";
        operands.printInitializationList(ofs,1);
        ofs << ");" << endl << "}" << endl;

        if (operands.anyComplex())
            ofs << "#endif // BZ_HAVE_COMPLEX" << endl;

    } while (++operands);

   }

   cout << operands.numSpecializations() << " operators written." << endl;
}

