
#include <iostream>
#include <fstream>
#include "bzfstream.h"
#include "arroptuple.h"

#ifdef NO_BOOL
 #define bool int
 #define true 1
 #define false 0
#endif

int main()
{
    std::cout << "Generating <array/bops.cc>" << std::endl;

    OperandTuple operands(2);

    bzofstream ofs("../array/bops.cc", "Array expression templates (2 operands)",
        __FILE__, "BZ_ARRAYBOPS_CC");

    ofs << "#ifndef BZ_ARRAYEXPR_H" << std::endl
        << " #error <blitz/array/bops.cc> must be included after <blitz/arrayexpr.h>" 
        << std::endl << "#endif" << std::endl << std::endl;

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
     { "<<", false, false, "ShiftLeft",      "Shift left Operators" },
		 { "min", false, false, "_bz_Min",      "Minimum Operators" },
		 { "max", false, false, "_bz_Max",      "Maximum Operators" }
    };

    const int numOperators = 20;   // Should be 20

    for (int i=0; i < numOperators; ++i)
    {
    ofs << "/****************************************************************************" << std::endl
        << " * " << ops[i].comment << std::endl
        << " ****************************************************************************/" << std::endl;

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

        ofs << std::endl;

        if (operands.anyComplex())
            ofs << "#ifdef BZ_HAVE_COMPLEX" << std::endl;

        ofs << "// ";
        operands[0].printName(ofs);
        ofs << " " << ops[i].opSymbol << " ";
        operands[1].printName(ofs);
        ofs << std::endl;

        operands.printTemplates(ofs);
        ofs << std::endl << "inline" << std::endl;

        ofs << "_bz_ArrayExpr<_bz_ArrayExprBinaryOp<";
        operands.printIterators(ofs, 1);
        ofs << "," << std::endl << "      " << ops[i].opApplicName << "<";
        operands[0].printNumtype(ofs);
        ofs << ", ";    
        operands[1].printNumtype(ofs);
        ofs << " > > >" << std::endl;
     
        if (ops[i].opSymbol[0] == 'm')
            ofs << ops[i].opSymbol << "(";
        else
            ofs << "operator" << ops[i].opSymbol << "(";
        operands.printArgumentList(ofs, 1);
        ofs << ")" << std::endl << "{" << std::endl;

        ofs << "    return _bz_ArrayExprBinaryOp<";
        operands.printIterators(ofs, 1);
        ofs << ", " << std::endl << "      " << ops[i].opApplicName << "<";
        operands[0].printNumtype(ofs);
        ofs << ", ";
        operands[1].printNumtype(ofs);
        ofs << "> >" << std::endl;
        ofs << "      (";
        operands.printInitializationList(ofs,1);
        ofs << ");" << std::endl << "}" << std::endl;

        if (operands.anyComplex())
            ofs << "#endif // BZ_HAVE_COMPLEX" << std::endl;

    } while (++operands);

   }

   std::cout << operands.numSpecializations() << " operators written." << std::endl;
}

