#include <fstream.h>

struct {
    int priority;
    int promotion;
    const char* name;
} types[] = {
    { 9,  4, "char" },
    { 9,  4, "unsigned char" },
    { 9,  4, "short int" },
    { 9,  5, "short unsigned int" },
    { 9, -1, "int" },                 /* 4 */
    { 8, -1, "unsigned int" },        /* 5 */
    { 7, -1, "long" },
    { 6, -1, "unsigned long" },
    { 5, -1, "float" },
    { 4, -1, "double" },
    { 3, -1, "long double" },
    { 2, -1, "complex<float> " },
    { 1, -1, "complex<double> " },
    { 0, -1, "complex<long double> " }
};

int nTypes = 14;

const char *className = "promote_trait";
const char *typeName = "T_promote";

void generate()
{
    cout << "Generating <promote.h>" << endl;

    ofstream ofs("promote.h");

    ofs << "/***********************************************************************\n"
" * promote.h   Arithmetic type promotion trait class\n"
" * Author: Todd Veldhuizen         (tveldhui@seurat.uwaterloo.ca)\n"
" *\n"
" * This program may be distributed in an unmodified form.  It may not be\n"
" * sold or used in a commercial product.\n"
" *\n"
" * For more information on these template techniques, please see the\n"
" * Blitz++ Numerical Library Project, at URL http://monet.uwaterloo.ca/blitz/\n"
" */\n"
"\n"
<< "// Generated: " << __FILE__ << " " << __DATE__ << " " << __TIME__ << endl <<endl <<
"#ifndef BZ_PROMOTE_H\n"
"#define BZ_PROMOTE_H\n\n"
"#include <blitz/blitz.h>\n"
"#include <complex>\n\n"
"BZ_NAMESPACE(blitz)\n\n"
"#ifdef BZ_TEMPLATE_QUALIFIED_RETURN_TYPE\n"
"    #define BZ_PROMOTE(A,B) _bz_typename promote_trait<A,B>::T_promote\n"
"#else\n"
"    #define BZ_PROMOTE(A,B) A\n"
"#endif\n";

ofs <<
"// NEEDS_WORK: once partial specialization is supported, the default\n"
"// behaviour of promote_trait should be changed to promote to whichever\n"
"// type has a bigger sizeof().\n"
"\n"
"template<class A, class B>\n"
"class promote_trait {\n"
"public:\n"
"        typedef A   T_promote;\n"
"};\n\n\n";

    for (int i=0; i < nTypes; ++i)
    {
        for (int j=0; j < nTypes; ++j)
        {
            int promote;

            if ((i > 7) || (j > 7))
            {
                // One of them is float
                if (types[i].priority < types[j].priority)
                    promote = i;
                else
                    promote = j;
            }
            else {
                int ni = i, nj = j;
                if (types[i].promotion != -1)
                    ni = types[i].promotion;
                if (types[j].promotion != -1)
                    nj = types[j].promotion;

                if (types[ni].priority < types[nj].priority)
                    promote = ni;
                else
                    promote = nj;
            }


            if ((i >= 11) || (j >= 11))
                ofs << "#ifdef BZ_HAVE_COMPLEX" << endl;

            ofs << "template<>" << endl
                << "class " << className << "<" << types[i].name << ", "
                << types[j].name << "> {\npublic:\n"
                << "\ttypedef " << types[promote].name << " "
                << typeName << ";\n};\n";

            if ((i >= 11) || (j >= 11))
                ofs << "#endif" << endl;

            ofs << endl;
        }
    }

    ofs << endl << "BZ_NAMESPACE_END" << endl << endl
        << "#endif // BZ_PROMOTE_H" << endl;
}

int main()
{
    generate();
    return 1;
}
