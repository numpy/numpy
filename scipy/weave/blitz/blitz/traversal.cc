/***************************************************************************
 * blitz/traversal.cc  Space-filling curve based traversal orders
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

#ifndef BZ_TRAVERSAL_CC
#define BZ_TRAVERSAL_CC

#ifndef BZ_TRAVERSAL_H
 #error <blitz/traversal.cc> must be included via <blitz/traversal.h>
#endif

BZ_NAMESPACE(blitz)

// Next line is a workaround for Intel C++ V4.0 oddity, due
// to Allan Stokes.
static set<TraversalOrder<2> > *_bz_intel_kludge;

//template<int N_dimensions>
//_bz_typename TraversalOrderCollection<N_dimensions>::T_set
//    TraversalOrderCollection<N_dimensions>::traversals_;

template<int N>
void makeHilbert(Vector<TinyVector<int,N> >& coord, 
    int x0, int y0, int xis, int xjs,
    int yis, int yjs, int n, int& i)
{
    // N != 2 is not yet implemented.
    BZPRECONDITION(N == 2);

    if (!n)
    {
        if (i > coord.length())
        {
            cerr << "makeHilbert: vector not long enough" << endl;
            exit(1);
        }
        coord[i][0] = x0 + (xis+yis)/2;
        coord[i][1] = y0 + (xjs+yjs)/2;
        ++i;
    }
    else {
        makeHilbert(coord,x0,y0,yis/2, yjs/2, xis/2, xjs/2, n-1, i);
        makeHilbert(coord,x0+xis/2,y0+xjs/2,xis/2,xjs/2,yis/2,yjs/2,n-1,i);
        makeHilbert(coord,x0+xis/2+yis/2,y0+xjs/2+yjs/2,xis/2,xjs/2,yis/2,
            yjs/2,n-1,i);
        makeHilbert(coord,x0+xis/2+yis, y0+xjs/2+yjs, -yis/2,-yjs/2,-xis/2,
            -xjs/2,n-1,i);
    }
}

template<int N_dimensions>
void MakeHilbertTraversal(Vector<TinyVector<int,N_dimensions> >& coord, 
    int length)
{
    // N_dimensions != 2 not yet implemented
    BZPRECONDITION(N_dimensions == 2);

    // The constant on the next line is ln(2)
    int d = (int)(::ceil(::log((double)length) / 0.693147180559945309417));  

    int N = 1 << d;
    const int Npoints = N*N;
    Vector<TinyVector<int,2> > coord2(Npoints);

    int i=0;
    makeHilbert(coord2,0,0,32768,0,0,32768,d,i);

    int xp0 = coord2[0][0];
    int yp0 = coord2[0][1];

    int j=0;

    coord.resize(length * length);

    for (int i=0; i < Npoints; ++i)
    {
        coord2[i][0] = (coord2[i][0]-xp0)/(2*xp0);
        coord2[i][1] = (coord2[i][1]-yp0)/(2*yp0);

        if ((coord2[i][0] < length) && (coord2[i][1] < length) 
            && (coord2[i][0] >= 0) && (coord2[i][1] >= 0))
        {
            coord[j][0] = coord2[i][0];
            coord[j][1] = coord2[i][1];
            ++j;
        }
    }
}

template<int N_dimensions>
void generateFastTraversalOrder(const TinyVector<int,N_dimensions>& size)
{
    BZPRECONDITION(N_dimensions == 2);
    BZPRECONDITION(size[0] == size[1]);

    TraversalOrderCollection<2> travCol;
    if (travCol.find(size))
        return;

    Vector<TinyVector<int,2> > ordering(size[0]);

    MakeHilbertTraversal(ordering, size[0]);
    travCol.insert(TraversalOrder<2>(size, ordering));
}

BZ_NAMESPACE_END

#endif // BZ_TRAVERSAL_CC
