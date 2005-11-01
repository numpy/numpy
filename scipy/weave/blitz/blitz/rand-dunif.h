/***************************************************************************
 * blitz/rand-dunif.h    Discrete uniform generator
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

#ifndef BZ_RAND_DUNIF_H
#define BZ_RAND_DUNIF_H

#ifndef BZ_RANDOM_H
 #include <blitz/random.h>
#endif

#ifndef BZ_RAND_UNIFORM_H
 #include <blitz/rand-uniform.h>
#endif

#include <math.h>

BZ_NAMESPACE(blitz)

template<typename P_uniform BZ_TEMPLATE_DEFAULT(Uniform)>
class DiscreteUniform {

public:
    typedef int T_numtype;
    typedef P_uniform T_uniform;

    DiscreteUniform(int low, int high, double=0)
        : low_(low), range_(high-low+1)
    { 
    }

    void randomize() 
    { 
        uniform_.randomize();
    }
  
    int random()
    { 
        return int(uniform_.random() * range_ + low_);
    } 

private:
    int low_, range_;
    T_uniform uniform_;
};

BZ_NAMESPACE_END

#endif // BZ_RAND_DUNIF_H

