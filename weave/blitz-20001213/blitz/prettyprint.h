/***************************************************************************
 * blitz/prettyprint.h      Format object for pretty-printing of
 *                          array expressions
 *
 * $Id$
 *
 * Copyright (C) 1997-1999 Todd Veldhuizen <tveldhui@oonumerics.org>
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
 ***************************************************************************
 * $Log$
 * Revision 1.1  2002/01/03 19:50:34  eric
 * renaming compiler to weave
 *
 * Revision 1.1  2001/04/27 17:22:04  ej
 * first attempt to include needed pieces of blitz
 *
 * Revision 1.1.1.1  2000/06/19 12:26:12  tveldhui
 * Imported sources
 *
 * Revision 1.2  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 */

#ifndef BZ_PRETTYPRINT_H
#define BZ_PRETTYPRINT_H

BZ_NAMESPACE(blitz)

class prettyPrintFormat {

public:
    prettyPrintFormat(_bz_bool terse = _bz_false)
      : tersePrintingSelected_(terse)
    {
        arrayOperandCounter_ = 0;
        scalarOperandCounter_ = 0;
        dumpArrayShapes_ = _bz_false;
    }

    void setDumpArrayShapesMode()
    {
        dumpArrayShapes_ = _bz_true;
    }

    char nextArrayOperandSymbol()
    {
        return 'A' + ((arrayOperandCounter_++) % 26);
    }

    char nextScalarOperandSymbol()
    {
        return 's' + ((scalarOperandCounter_++) % 26);
    }

    _bz_bool tersePrintingSelected() const
    { 
        return tersePrintingSelected_;
    }

    _bz_bool dumpArrayShapesMode() const
    {
        return dumpArrayShapes_;
    }

private:
    _bz_bool tersePrintingSelected_;
    _bz_bool dumpArrayShapes_;
    int arrayOperandCounter_;
    int scalarOperandCounter_;
};

BZ_NAMESPACE_END

#endif // BZ_PRETTYPRINT_H
