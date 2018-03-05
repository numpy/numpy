#pragma once
#ifndef DSFMT_CALC_JUMP_HPP
#define DSFMT_CALC_JUMP_HPP
/**
 * @file dSFMT-calc-jump.hpp
 *
 * @brief functions for calculating jump polynomial.
 *
 * @author Mutsuo Saito (Hiroshima University)
 * @author Makoto Matsumoto (The University of Tokyo)
 *
 * Copyright (C) 2012 Mutsuo Saito, Makoto Matsumoto,
 * Hiroshima University and The University of Tokyo.
 * All rights reserved.
 *
 * The 3-clause BSD License is applied to this software, see
 * LICENSE.txt
 */
#include <iostream>
#include <iomanip>
#include <sstream>
#include <NTL/GF2X.h>

namespace dsfmt {
/**
 * converts polynomial to string for convenient use in C language.
 * @param x output string
 * @param polynomial input polynomial
 */
    static inline void polytostring(std::string& x, NTL::GF2X& polynomial)
    {
	using namespace NTL;
	using namespace std;

	long degree = deg(polynomial);
	int buff;
	stringstream ss;
	for (int i = 0; i <= degree; i+=4) {
	    buff = 0;
	    for (int j = 0; j < 4; j++) {
		if (IsOne(coeff(polynomial, i + j))) {
		    buff |= 1 << j;
		} else {
		    buff &= (0x0f ^ (1 << j));
		}
	    }
	    ss << hex << buff;
	}
	ss << flush;
	x = ss.str();
    }

/**
 * converts string to polynomial
 * @param str string
 * @param poly output polynomial
 */
    static inline void stringtopoly(NTL::GF2X& poly, std::string& str)
    {
	using namespace NTL;
	using namespace std;

	stringstream ss(str);
	char c;
	long p = 0;
	clear(poly);
	while(ss) {
	    ss >> c;
	    if (!ss) {
		break;
	    }
	    if (c >= 'a') {
		c = c - 'a' + 10;
	    } else {
		c = c - '0';
	    }
	    for (int j = 0; j < 4; j++) {
		if (c & (1 << j)) {
		    SetCoeff(poly, p, 1);
		} else {
		    SetCoeff(poly, p, 0);
		}
		p++;
	    }
	}
    }

/**
 * calculate the jump polynomial.
 * SFMT generates 4 32-bit integers from one internal state.
 * @param jump_str output string which represents jump polynomial.
 * @param step jump step of internal state
 * @param characteristic polynomial
 */
    static inline void calc_jump(std::string& jump_str,
				 NTL::ZZ& step,
				 NTL::GF2X& characteristic)
    {
	using namespace NTL;
	using namespace std;
	GF2X jump;
	PowerXMod(jump, step, characteristic);
	polytostring(jump_str, jump);
    }
}
#endif
