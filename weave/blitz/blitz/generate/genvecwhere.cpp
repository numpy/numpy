#include "bzfstream.h"
#include "optuple.h"

int main()
{
    cout << "Generating <vecwhere.cc>" << endl;

    bzofstream ofs("vecwhere.cc", "where(X,Y,Z) function for vectors",
        __FILE__, "BZ_VECWHERE_CC");

    ofs.beginNamespace();

    OperandTuple ops(3);

    do {
        if (ops[0].isScalar())
            continue;

        if (ops[1].isScalar() && ops[2].isScalar())
        {
            if (ops.operandIndex(1) != ops.operandIndex(2))
                continue;
        }

        ofs << "// where(";
        ops.printTypes(ofs);
        ofs << ")" << endl;

        int complexFlag = 0;
        if (ops.anyComplex())
        {
            ofs << "#ifdef BZ_HAVE_COMPLEX" << endl;
            complexFlag = 1;
        }

        ops.printTemplates(ofs);
        ofs << endl << "inline" << endl;
        ofs << "_bz_VecExpr<_bz_VecWhere<";
        ops.printIterators(ofs, 1);
        ofs << " > >" << endl;

        ofs << "where(";
        ops.printArgumentList(ofs, 1);
        ofs << ")" << endl
            << "{ " << endl;

        ofs << "    typedef _bz_VecWhere<";
        ops.printIterators(ofs, 1);
        ofs << " > T_expr;" << endl << endl;

        ofs << "    return _bz_VecExpr<T_expr>(T_expr(";
        ops.printInitializationList(ofs, 1);
        ofs << "));" << endl
            << "}" << endl;
      
        if (complexFlag)
            ofs << "#endif // BZ_HAVE_COMPLEX" << endl;

        ofs << endl;
    } while (++ops);

    cout << ops.numSpecializations() << " specializations written." << endl;
}

