#include <iostream>
#include <fstream>

struct TypePromotions {
    int priority;
    int promotion;
    const char* name;
};

TypePromotions types[] = {
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
    std::cout << "Generating <promote-old.h>" << std::endl;

    std::ofstream ofs("../promote-old.h");

    ofs << "/***********************************************************************\n"
" * promote.h   Arithmetic type promotion trait class\n"
" * Author: Todd Veldhuizen         (tveldhui@oonumerics.org)\n"
" *\n"
" * Copyright (C) 1997-2001 Todd Veldhuizen <tveldhui@oonumerics.org>\n"
" *\n"
" * This program is free software; you can redistribute it and/or\n"
" * modify it under the terms of the GNU General Public License\n"
" * as published by the Free Software Foundation; either version 2\n"
" * of the License, or (at your option) any later version.\n"
" *\n"
" * This program is distributed in the hope that it will be useful,\n"
" * but WITHOUT ANY WARRANTY; without even the implied warranty of\n"
" * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n"
" * GNU General Public License for more details.\n"
" *\n"
" * Suggestions:          blitz-dev@oonumerics.org\n"
" * Bugs:                 blitz-bugs@oonumerics.org\n"
" *\n"
" * For more information, please see the Blitz++ Home Page:\n"
" *    http://oonumerics.org/blitz/\n"
" *\n"
" ***************************************************************************\n"
" */\n"
"\n"
        << "// Generated: " << __FILE__ << " " << __DATE__ << " " << __TIME__ << std::endl;

ofs <<
"template<typename A, typename B>\n"
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
                ofs << "#ifdef BZ_HAVE_COMPLEX" << std::endl;

            ofs << "template<>" << std::endl
                << "class " << className << "<" << types[i].name << ", "
                << types[j].name << "> {\npublic:\n"
                << "\ttypedef " << types[promote].name << " "
                << typeName << ";\n};\n";

            if ((i >= 11) || (j >= 11))
                ofs << "#endif" << std::endl;

            ofs << std::endl;
        }
    }

}

int main()
{
    generate();
    return 0;
}
