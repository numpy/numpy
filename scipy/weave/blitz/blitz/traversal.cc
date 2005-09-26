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
 ***************************************************************************
 * $Log$
 * Revision 1.2  2002/09/12 07:04:04  eric
 * major rewrite of weave.
 *
 * 0.
 * The underlying library code is significantly re-factored and simpler. There used to be a xxx_spec.py and xxx_info.py file for every group of type conversion classes.  The spec file held the python code that handled the conversion and the info file had most of the C code templates that were generated.  This proved pretty confusing in practice, so the two files have mostly been merged into the spec file.
 *
 * Also, there was quite a bit of code duplication running around.  The re-factoring was able to trim the standard conversion code base (excluding blitz and accelerate stuff) by about 40%.  This should be a huge maintainability and extensibility win.
 *
 * 1.
 * With multiple months of using Numeric arrays, I've found some of weave's "magic variable" names unwieldy and want to change them.  The following are the old declarations for an array x of Float32 type:
 *
 *         PyArrayObject* x = convert_to_numpy(...);
 *         float* x_data = (float*) x->data;
 *         int*   _Nx = x->dimensions;
 *         int*   _Sx = x->strides;
 *         int    _Dx = x->nd;
 *
 * The new declaration looks like this:
 *
 *         PyArrayObject* x_array = convert_to_numpy(...);
 *         float* x = (float*) x->data;
 *         int*   Nx = x->dimensions;
 *         int*   Sx = x->strides;
 *         int    Dx = x->nd;
 *
 * This is obviously not backward compatible, and will break some code (including a lot of mine).  It also makes inline() code more readable and natural to write.
 *
 * 2.
 * I've switched from CXX to Gordon McMillan's SCXX for list, tuples, and dictionaries.  I like CXX pretty well, but its use of advanced C++ (templates, etc.) caused some portability problems.  The SCXX library is similar to CXX but doesn't use templates at all.  This, like (1) is not an
 * API compatible change and requires repairing existing code.
 *
 * I have also thought about boost python, but it also makes heavy use of templates.  Moving to SCXX gets rid of almost all template usage for the standard type converters which should help portability.  std::complex and std::string from the STL are the only templates left.  Of course blitz still uses templates in a major way so weave.blitz will continue to be hard on compilers.
 *
 * I've actually considered scrapping the C++ classes for list, tuples, and
 * dictionaries, and just fall back to the standard Python C API because the classes are waaay slower than the raw API in many cases.  They are also more convenient and less error prone in many cases, so I've decided to stick with them.  The PyObject variable will always be made available for variable "x" under the name "py_x" for more speedy operations.  You'll definitely want to use these for anything that needs to be speedy.
 *
 * 3.
 * strings are converted to std::string now.  I found this to be the most useful type in for strings in my code.  Py::String was used previously.
 *
 * 4.
 * There are a number of reference count "errors" in some of the less tested conversion codes such as instance, module, etc.  I've cleaned most of these up.  I put errors in quotes here because I'm actually not positive that objects passed into "inline" really need reference counting applied to them.  The dictionaries passed in by inline() hold references to these objects so it doesn't seem that they could ever be garbage collected inadvertently.  Variables used by ext_tools, though, definitely need the reference counting done.  I don't think this is a major cost in speed, so it probably isn't worth getting rid of the ref count code.
 *
 * 5.
 * Unicode objects are now supported.  This was necessary to support rendering Unicode strings in the freetype wrappers for Chaco.
 *
 * 6.
 * blitz++ was upgraded to the latest CVS.  It compiles about twice as fast as the old blitz and looks like it supports a large number of compilers (though only gcc 2.95.3 is tested).  Compile times now take about 9 seconds on my 850 MHz PIII laptop.
 *
 * Revision 1.3  2002/03/06 17:18:11  patricg
 *
 * template declaration
 * template<int N_dimensions>
 * _bz_typename TraversalOrderCollection<N_dimensions>::T_set
 * 	TraversalOrderCollection<N_dimensions>::traversals_;
 * in blitz/transversal.cc moved before template specialisation
 * template<>
 * class TraversalOrderCollection<0> {}
 * in blitz/transversal.h
 *
 * Revision 1.2  2001/01/25 00:25:55  tveldhui
 * Ensured that source files have cvs logs.
 *
 */

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
