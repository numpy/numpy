// -*- C++ -*-
/***************************************************************************
 * blitz/etbase.h    Declaration of the ETBase<T> class
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

#ifndef BZ_ETBASE_H
#define BZ_ETBASE_H

BZ_NAMESPACE(blitz)

template<typename T>
class ETBase { 
public:
    ETBase() 
    { }

    ETBase(const ETBase<T>&)
    { }
    
    T& unwrap() { return static_cast<T&>(*this); }
    
    const T& unwrap() const { return static_cast<const T&>(*this); }
};

BZ_NAMESPACE_END

#endif // BZ_ETBASE_H

