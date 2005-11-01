/***************************************************************************
 * blitz/generate/bzfstream.h    Definition of the bzofstream class
 *
 * $Id$
 *
 * Copyright (C) 1997-2001 Todd Veldhuizen <tveldhui@oonumerics.org>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * Suggestions:          blitz-dev@oonumerics.org
 * Bugs:                 blitz-bugs@oonumerics.org
 *
 * For more information, please see the Blitz++ Home Page:
 *    http://oonumerics.org/blitz/
 *
 ***************************************************************************/


#include <fstream>
#include <iomanip>
#include <iostream>

class bzofstream : public std::ofstream {

public:
    bzofstream(const char* filename, const char* description,
        const char* sourceFile, const char* mnemonic)
        : std::ofstream(filename)
    {
        (*this) << 
"/***************************************************************************\n"
" * blitz/" << filename << "\t" << description << std::endl <<
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
" * Suggestions:          blitz-suggest@cybervision.com\n"
" * Bugs:                 blitz-bugs@cybervision.com\n"
" *\n"
" * For more information, please see the Blitz++ Home Page:\n"
" *    http://seurat.uwaterloo.ca/blitz/\n"
" *\n"
" ***************************************************************************\n"
" *\n"
" */ " 
       << std::endl << std::endl
       << "// Generated source file.  Do not edit. " << std::endl
       << "// " << sourceFile << " " << __DATE__ << " " << __TIME__ 
       << std::endl << std::endl
       << "#ifndef " << mnemonic << std::endl
       << "#define " << mnemonic << std::endl << std::endl;
    }

    void include(const char* filename)
    {
        (*this) << "#include <blitz/" << filename << ">" << std::endl;
    }

    void beginNamespace()
    {
        (*this) << "BZ_NAMESPACE(blitz)" << std::endl << std::endl;
    }

    ~bzofstream()
    {
        (*this) << "BZ_NAMESPACE_END" << std::endl << std::endl
                << "#endif" << std::endl;
    }

};

