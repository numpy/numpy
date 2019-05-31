/**
 * @file calc-jump.cpp
 *
 * @brief calc jump function.
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
 *
 * Compile:
 * g++ calc-jump.cpp -o calc-jump -lntl
 *
 * Compute polynomial for 2^128 steps:
 * ./calc-jump 340282366920938463463374607431768211456 poly.19937.txt
 *
 */
#include <iostream>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <inttypes.h>
#include <stdint.h>
#include <time.h>
#include <NTL/GF2X.h>
#include <NTL/vec_GF2.h>
#include <NTL/ZZ.h>
#include "dSFMT-calc-jump.hpp"

using namespace NTL;
using namespace std;
using namespace dsfmt;

static void read_file(GF2X& lcmpoly, long line_no, const string& file);

int main(int argc, char * argv[]) {
    if (argc <= 2) {
	cout << argv[0] << " jump-step poly-file" << endl;
	cout << "    jump-step: a number between zero and 2^{DSFMT_MEXP}-1.\n"
	     << "               large decimal number is allowed." << endl;
	cout << "    poly-file: one of poly.{MEXP}.txt "
	     << "file" << endl;
	return -1;
    }
    string step_string = argv[1];
    string filename = argv[2];
    long no = 0;
    GF2X lcmpoly;
    read_file(lcmpoly, no, filename);
    ZZ step;
    stringstream ss(step_string);
    ss >> step;
    string jump_str;
    calc_jump(jump_str, step, lcmpoly);
    cout << "jump polynomial:" << endl;
    cout << jump_str << endl;
    return 0;
}


static void read_file(GF2X& lcmpoly, long line_no, const string& file)
{
    ifstream ifs(file.c_str());
    string line;
    for (int i = 0; i < line_no; i++) {
	ifs >> line;
	ifs >> line;
    }
    if (ifs) {
	ifs >> line;
	line = "";
	ifs >> line;
    }
    stringtopoly(lcmpoly, line);
}
