/*
 * $Id$
 *
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.  Please see <blitz/blitz.h> for terms and
 * conditions of use.
 *
 * $Log$
 * Revision 1.1  2002/01/03 19:50:34  eric
 * renaming compiler to weave
 *
 * Revision 1.1  2001/04/27 17:25:28  ej
 * Looks like I need all the .cc files for blitz also
 *
 * Revision 1.1.1.1  2000/06/19 12:26:08  tveldhui
 * Imported sources
 *
 * Revision 1.3  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.2  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 */

#ifndef BZ_VECGLOBS_CC
#define BZ_VECGLOBS_CC

#ifndef BZ_VECGLOBS_H
 #include <blitz/vecglobs.h>
#endif

#include <blitz/vecaccum.cc>    // accumulate()
#include <blitz/vecdelta.cc>    // delta()
#include <blitz/vecmin.cc>      // min(), minValue(), minIndex()
#include <blitz/vecmax.cc>      // max(), maxValue(), maxIndex()
#include <blitz/vecsum.cc>      // sum(), mean()
#include <blitz/vecdot.cc>      // dot()
#include <blitz/vecnorm.cc>     // norm()
#include <blitz/vecnorm1.cc>    // norm1()
#include <blitz/vecany.cc>      // any()
#include <blitz/vecall.cc>      // all()
#include <blitz/veccount.cc>    // count()

#endif // BZ_VECGLOBS_CC
