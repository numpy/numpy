/*
 * $Id$
 *
 * Copyright (C) 1997 Todd Veldhuizen <tveldhui@oonumerics.org>
 * All rights reserved.  Please see <blitz/blitz.h> for terms and
 * conditions of use.
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
