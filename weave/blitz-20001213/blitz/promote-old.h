/***********************************************************************
 * promote.h   Arithmetic type promotion trait class
 * Author: Todd Veldhuizen         (tveldhui@oonumerics.org)
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
 */

// Generated: genpromote.cpp Aug  7 1997 14:59:32

template<class A, class B>
class promote_trait {
public:
        typedef A   T_promote;
};


template<>
class promote_trait<char, char> {
public:
	typedef int T_promote;
};

template<>
class promote_trait<char, unsigned char> {
public:
	typedef int T_promote;
};

template<>
class promote_trait<char, short int> {
public:
	typedef int T_promote;
};

template<>
class promote_trait<char, short unsigned int> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<char, int> {
public:
	typedef int T_promote;
};

template<>
class promote_trait<char, unsigned int> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<char, long> {
public:
	typedef long T_promote;
};

template<>
class promote_trait<char, unsigned long> {
public:
	typedef unsigned long T_promote;
};

template<>
class promote_trait<char, float> {
public:
	typedef float T_promote;
};

template<>
class promote_trait<char, double> {
public:
	typedef double T_promote;
};

template<>
class promote_trait<char, long double> {
public:
	typedef long double T_promote;
};

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<char, complex<float> > {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<char, complex<double> > {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<char, complex<long double> > {
public:
	typedef complex<long double>  T_promote;
};
#endif

template<>
class promote_trait<unsigned char, char> {
public:
	typedef int T_promote;
};

template<>
class promote_trait<unsigned char, unsigned char> {
public:
	typedef int T_promote;
};

template<>
class promote_trait<unsigned char, short int> {
public:
	typedef int T_promote;
};

template<>
class promote_trait<unsigned char, short unsigned int> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<unsigned char, int> {
public:
	typedef int T_promote;
};

template<>
class promote_trait<unsigned char, unsigned int> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<unsigned char, long> {
public:
	typedef long T_promote;
};

template<>
class promote_trait<unsigned char, unsigned long> {
public:
	typedef unsigned long T_promote;
};

template<>
class promote_trait<unsigned char, float> {
public:
	typedef float T_promote;
};

template<>
class promote_trait<unsigned char, double> {
public:
	typedef double T_promote;
};

template<>
class promote_trait<unsigned char, long double> {
public:
	typedef long double T_promote;
};

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<unsigned char, complex<float> > {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<unsigned char, complex<double> > {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<unsigned char, complex<long double> > {
public:
	typedef complex<long double>  T_promote;
};
#endif

template<>
class promote_trait<short int, char> {
public:
	typedef int T_promote;
};

template<>
class promote_trait<short int, unsigned char> {
public:
	typedef int T_promote;
};

template<>
class promote_trait<short int, short int> {
public:
	typedef int T_promote;
};

template<>
class promote_trait<short int, short unsigned int> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<short int, int> {
public:
	typedef int T_promote;
};

template<>
class promote_trait<short int, unsigned int> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<short int, long> {
public:
	typedef long T_promote;
};

template<>
class promote_trait<short int, unsigned long> {
public:
	typedef unsigned long T_promote;
};

template<>
class promote_trait<short int, float> {
public:
	typedef float T_promote;
};

template<>
class promote_trait<short int, double> {
public:
	typedef double T_promote;
};

template<>
class promote_trait<short int, long double> {
public:
	typedef long double T_promote;
};

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<short int, complex<float> > {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<short int, complex<double> > {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<short int, complex<long double> > {
public:
	typedef complex<long double>  T_promote;
};
#endif

template<>
class promote_trait<short unsigned int, char> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<short unsigned int, unsigned char> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<short unsigned int, short int> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<short unsigned int, short unsigned int> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<short unsigned int, int> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<short unsigned int, unsigned int> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<short unsigned int, long> {
public:
	typedef long T_promote;
};

template<>
class promote_trait<short unsigned int, unsigned long> {
public:
	typedef unsigned long T_promote;
};

template<>
class promote_trait<short unsigned int, float> {
public:
	typedef float T_promote;
};

template<>
class promote_trait<short unsigned int, double> {
public:
	typedef double T_promote;
};

template<>
class promote_trait<short unsigned int, long double> {
public:
	typedef long double T_promote;
};

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<short unsigned int, complex<float> > {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<short unsigned int, complex<double> > {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<short unsigned int, complex<long double> > {
public:
	typedef complex<long double>  T_promote;
};
#endif

template<>
class promote_trait<int, char> {
public:
	typedef int T_promote;
};

template<>
class promote_trait<int, unsigned char> {
public:
	typedef int T_promote;
};

template<>
class promote_trait<int, short int> {
public:
	typedef int T_promote;
};

template<>
class promote_trait<int, short unsigned int> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<int, int> {
public:
	typedef int T_promote;
};

template<>
class promote_trait<int, unsigned int> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<int, long> {
public:
	typedef long T_promote;
};

template<>
class promote_trait<int, unsigned long> {
public:
	typedef unsigned long T_promote;
};

template<>
class promote_trait<int, float> {
public:
	typedef float T_promote;
};

template<>
class promote_trait<int, double> {
public:
	typedef double T_promote;
};

template<>
class promote_trait<int, long double> {
public:
	typedef long double T_promote;
};

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<int, complex<float> > {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<int, complex<double> > {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<int, complex<long double> > {
public:
	typedef complex<long double>  T_promote;
};
#endif

template<>
class promote_trait<unsigned int, char> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<unsigned int, unsigned char> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<unsigned int, short int> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<unsigned int, short unsigned int> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<unsigned int, int> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<unsigned int, unsigned int> {
public:
	typedef unsigned int T_promote;
};

template<>
class promote_trait<unsigned int, long> {
public:
	typedef long T_promote;
};

template<>
class promote_trait<unsigned int, unsigned long> {
public:
	typedef unsigned long T_promote;
};

template<>
class promote_trait<unsigned int, float> {
public:
	typedef float T_promote;
};

template<>
class promote_trait<unsigned int, double> {
public:
	typedef double T_promote;
};

template<>
class promote_trait<unsigned int, long double> {
public:
	typedef long double T_promote;
};

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<unsigned int, complex<float> > {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<unsigned int, complex<double> > {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<unsigned int, complex<long double> > {
public:
	typedef complex<long double>  T_promote;
};
#endif

template<>
class promote_trait<long, char> {
public:
	typedef long T_promote;
};

template<>
class promote_trait<long, unsigned char> {
public:
	typedef long T_promote;
};

template<>
class promote_trait<long, short int> {
public:
	typedef long T_promote;
};

template<>
class promote_trait<long, short unsigned int> {
public:
	typedef long T_promote;
};

template<>
class promote_trait<long, int> {
public:
	typedef long T_promote;
};

template<>
class promote_trait<long, unsigned int> {
public:
	typedef long T_promote;
};

template<>
class promote_trait<long, long> {
public:
	typedef long T_promote;
};

template<>
class promote_trait<long, unsigned long> {
public:
	typedef unsigned long T_promote;
};

template<>
class promote_trait<long, float> {
public:
	typedef float T_promote;
};

template<>
class promote_trait<long, double> {
public:
	typedef double T_promote;
};

template<>
class promote_trait<long, long double> {
public:
	typedef long double T_promote;
};

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<long, complex<float> > {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<long, complex<double> > {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<long, complex<long double> > {
public:
	typedef complex<long double>  T_promote;
};
#endif

template<>
class promote_trait<unsigned long, char> {
public:
	typedef unsigned long T_promote;
};

template<>
class promote_trait<unsigned long, unsigned char> {
public:
	typedef unsigned long T_promote;
};

template<>
class promote_trait<unsigned long, short int> {
public:
	typedef unsigned long T_promote;
};

template<>
class promote_trait<unsigned long, short unsigned int> {
public:
	typedef unsigned long T_promote;
};

template<>
class promote_trait<unsigned long, int> {
public:
	typedef unsigned long T_promote;
};

template<>
class promote_trait<unsigned long, unsigned int> {
public:
	typedef unsigned long T_promote;
};

template<>
class promote_trait<unsigned long, long> {
public:
	typedef unsigned long T_promote;
};

template<>
class promote_trait<unsigned long, unsigned long> {
public:
	typedef unsigned long T_promote;
};

template<>
class promote_trait<unsigned long, float> {
public:
	typedef float T_promote;
};

template<>
class promote_trait<unsigned long, double> {
public:
	typedef double T_promote;
};

template<>
class promote_trait<unsigned long, long double> {
public:
	typedef long double T_promote;
};

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<unsigned long, complex<float> > {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<unsigned long, complex<double> > {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<unsigned long, complex<long double> > {
public:
	typedef complex<long double>  T_promote;
};
#endif

template<>
class promote_trait<float, char> {
public:
	typedef float T_promote;
};

template<>
class promote_trait<float, unsigned char> {
public:
	typedef float T_promote;
};

template<>
class promote_trait<float, short int> {
public:
	typedef float T_promote;
};

template<>
class promote_trait<float, short unsigned int> {
public:
	typedef float T_promote;
};

template<>
class promote_trait<float, int> {
public:
	typedef float T_promote;
};

template<>
class promote_trait<float, unsigned int> {
public:
	typedef float T_promote;
};

template<>
class promote_trait<float, long> {
public:
	typedef float T_promote;
};

template<>
class promote_trait<float, unsigned long> {
public:
	typedef float T_promote;
};

template<>
class promote_trait<float, float> {
public:
	typedef float T_promote;
};

template<>
class promote_trait<float, double> {
public:
	typedef double T_promote;
};

template<>
class promote_trait<float, long double> {
public:
	typedef long double T_promote;
};

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<float, complex<float> > {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<float, complex<double> > {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<float, complex<long double> > {
public:
	typedef complex<long double>  T_promote;
};
#endif

template<>
class promote_trait<double, char> {
public:
	typedef double T_promote;
};

template<>
class promote_trait<double, unsigned char> {
public:
	typedef double T_promote;
};

template<>
class promote_trait<double, short int> {
public:
	typedef double T_promote;
};

template<>
class promote_trait<double, short unsigned int> {
public:
	typedef double T_promote;
};

template<>
class promote_trait<double, int> {
public:
	typedef double T_promote;
};

template<>
class promote_trait<double, unsigned int> {
public:
	typedef double T_promote;
};

template<>
class promote_trait<double, long> {
public:
	typedef double T_promote;
};

template<>
class promote_trait<double, unsigned long> {
public:
	typedef double T_promote;
};

template<>
class promote_trait<double, float> {
public:
	typedef double T_promote;
};

template<>
class promote_trait<double, double> {
public:
	typedef double T_promote;
};

template<>
class promote_trait<double, long double> {
public:
	typedef long double T_promote;
};

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<double, complex<float> > {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<double, complex<double> > {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<double, complex<long double> > {
public:
	typedef complex<long double>  T_promote;
};
#endif

template<>
class promote_trait<long double, char> {
public:
	typedef long double T_promote;
};

template<>
class promote_trait<long double, unsigned char> {
public:
	typedef long double T_promote;
};

template<>
class promote_trait<long double, short int> {
public:
	typedef long double T_promote;
};

template<>
class promote_trait<long double, short unsigned int> {
public:
	typedef long double T_promote;
};

template<>
class promote_trait<long double, int> {
public:
	typedef long double T_promote;
};

template<>
class promote_trait<long double, unsigned int> {
public:
	typedef long double T_promote;
};

template<>
class promote_trait<long double, long> {
public:
	typedef long double T_promote;
};

template<>
class promote_trait<long double, unsigned long> {
public:
	typedef long double T_promote;
};

template<>
class promote_trait<long double, float> {
public:
	typedef long double T_promote;
};

template<>
class promote_trait<long double, double> {
public:
	typedef long double T_promote;
};

template<>
class promote_trait<long double, long double> {
public:
	typedef long double T_promote;
};

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<long double, complex<float> > {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<long double, complex<double> > {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<long double, complex<long double> > {
public:
	typedef complex<long double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<float> , char> {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<float> , unsigned char> {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<float> , short int> {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<float> , short unsigned int> {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<float> , int> {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<float> , unsigned int> {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<float> , long> {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<float> , unsigned long> {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<float> , float> {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<float> , double> {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<float> , long double> {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<float> , complex<float> > {
public:
	typedef complex<float>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<float> , complex<double> > {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<float> , complex<long double> > {
public:
	typedef complex<long double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<double> , char> {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<double> , unsigned char> {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<double> , short int> {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<double> , short unsigned int> {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<double> , int> {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<double> , unsigned int> {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<double> , long> {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<double> , unsigned long> {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<double> , float> {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<double> , double> {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<double> , long double> {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<double> , complex<float> > {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<double> , complex<double> > {
public:
	typedef complex<double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<double> , complex<long double> > {
public:
	typedef complex<long double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<long double> , char> {
public:
	typedef complex<long double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<long double> , unsigned char> {
public:
	typedef complex<long double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<long double> , short int> {
public:
	typedef complex<long double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<long double> , short unsigned int> {
public:
	typedef complex<long double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<long double> , int> {
public:
	typedef complex<long double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<long double> , unsigned int> {
public:
	typedef complex<long double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<long double> , long> {
public:
	typedef complex<long double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<long double> , unsigned long> {
public:
	typedef complex<long double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<long double> , float> {
public:
	typedef complex<long double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<long double> , double> {
public:
	typedef complex<long double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<long double> , long double> {
public:
	typedef complex<long double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<long double> , complex<float> > {
public:
	typedef complex<long double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<long double> , complex<double> > {
public:
	typedef complex<long double>  T_promote;
};
#endif

#ifdef BZ_HAVE_COMPLEX
template<>
class promote_trait<complex<long double> , complex<long double> > {
public:
	typedef complex<long double>  T_promote;
};
#endif

