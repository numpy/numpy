/***************************************************************************
 * blitz/bzdebug.h      Debugging macros
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
 * Revision 1.1.1.1  2000/06/19 12:26:08  tveldhui
 * Imported sources
 *
 * Revision 1.7  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.6  1998/02/26 18:09:36  tveldhui
 * Added testsuite support for precondition fail checking
 *
 * Revision 1.5  1997/07/16 14:51:20  tveldhui
 * Update: Alpha release 0.2 (Arrays)
 *
 * Revision 1.4  1997/01/24 14:42:00  tveldhui
 * Periodic RCS update
 *
 * Revision 1.3  1997/01/13 22:19:58  tveldhui
 * Periodic RCS update
 *
 * Revision 1.2  1996/11/11 17:29:13  tveldhui
 * Periodic RCS update
 *
 * Revision 1.1  1996/04/14  12:36:45  todd
 * Initial revision
 *
 *
 */

#ifndef BZ_DEBUG_H
#define BZ_DEBUG_H

#include <stdlib.h>
#include <assert.h>

#ifdef BZ_RTTI
 #include <typeinfo>
#endif

BZ_NAMESPACE(blitz)

/*
 * These globals are used by the Blitz++ testsuite.  The _bz_global 
 * modifier ensures that they will reside in libblitz.a, but appear
 * "extern" elsewhere.
 */

_bz_global _bz_bool assertFailMode     BZ_GLOBAL_INIT(_bz_false);
_bz_global int      assertFailCount    BZ_GLOBAL_INIT(0);
_bz_global int      assertSuccessCount BZ_GLOBAL_INIT(0);


#ifdef BZ_TESTSUITE
  /*
   * In testsuite mode, these routines allow a test suite to check
   * that precondition checking is being done properly.  A typical
   * use looks like this:
   *
   * beginCheckAssert();
   *   // Some operation which should cause an assert to fail
   * endCheckAssert();
   *
   * The routine beginCheckAssert() sets a flag which results in
   * failed asserts being silently tallied.  If no asserts have
   * failed by the time endCheckAssert() is invoked, the program
   * halts and issues an error code.
   *
   * In normal operation (i.e. when beginCheckAssert() has not
   * been called), failed preconditions will cause the program
   * to halt and issue an error code.   -- TV 980226
   */

  inline void checkAssert(_bz_bool condition, const char* where=0, 
    int line=0)
  {
    if (assertFailMode == _bz_true)
    {
      if (condition == _bz_true)
        ++assertSuccessCount;
      else
        ++assertFailCount;
    }
    else {
      if (!condition)
      {
        cerr << "Unexpected assert failure!" << endl;
        if (where)
            cerr << where << ":" << line << endl;
        cerr.flush();
        assert(0);
      }
    }
  }

  inline void beginCheckAssert()
  {
    assertFailMode = _bz_true;
    assertSuccessCount = 0;
    assertFailCount = 0;
  }

  inline void endCheckAssert()
  {
    assert(assertFailMode == _bz_true);
    assertFailMode = _bz_false;
    if (assertFailCount == 0)
    {
      cerr << "Assert check failed!" << endl;
      assert(0);
    }
  }

    #define BZASSERT(X)        checkAssert(X, __FILE__, __LINE__)
    #define BZPRECONDITION(X)  checkAssert(X, __FILE__, __LINE__)
    #define BZPOSTCONDITION(X) checkAssert(X, __FILE__, __LINE__)
    #define BZSTATECHECK(X,Y)  checkAssert(X == Y, __FILE__, __LINE__)
    #define BZPRECHECK(X,Y)                                    \
        {                                                      \
            if ((assertFailMode == _bz_false) && (!(X)))       \
                cerr << Y << endl;                             \
            checkAssert(X, __FILE__, __LINE__);                \
        }

    #define BZ_DEBUG_MESSAGE(X) \
        { if (assertFailMode == _bz_false) { cout << __FILE__ << ":" << __LINE__ << " " << X << endl; } }

    #define BZ_PRE_FAIL        checkAssert(0)
#else 
#ifdef BZ_DEBUG
    #define BZASSERT(X)        assert(X)
    #define BZPRECONDITION(X)  assert(X)
    #define BZPOSTCONDITION(X) assert(X)
    #define BZSTATECHECK(X,Y)  assert(X == Y)
    #define BZPRECHECK(X,Y)        \
        { if (!(X))                                                           \
          { cerr << "[Blitz++] Precondition failure: Module " << __FILE__   \
               << " line " << __LINE__ << endl                              \
               << Y << endl;                                                \
            cerr.flush();                                                   \
            assert(0);                                                      \
          }                                                                 \
        }

    #define BZ_PRE_FAIL      assert(0)

    #define BZ_DEBUG_MESSAGE(X) \
        { cout << __FILE__ << ":" << __LINE__ << " " << X << endl; }

    void _bz_debug_marker();
    #define BZ_ASM_DEBUG_MARKER   _bz_debug_marker();
#else   // !BZ_DEBUG
    #define BZASSERT(X)
    #define BZPRECONDITION(X)
    #define BZPOSTCONDITION(X)
    #define BZSTATECHECK(X,Y)
    #define BZPRECHECK(X,Y)
    #define BZ_PRE_FAIL
    #define BZ_DEBUG_MESSAGE(X)
#endif  // !BZ_DEBUG
#endif  // !BZ_TESTSUITE

// This routine doesn't exist anywhere; it's used to mark a
// position of interest in assembler (.s) files
void _bz_debug_marker();

#define BZ_NOT_IMPLEMENTED()   { cerr << "[Blitz++] Not implemented: module " \
    << __FILE__ << " line " << __LINE__ << endl;                \
    exit(1); }

#ifdef BZ_RTTI
#define BZ_DEBUG_TEMPLATE_AS_STRING_LITERAL(X)  typeid(X).name()
#else

template<class T>
class _bz_stringLiteralForNumericType {
public:
    static const char* string()
    { return "unknown"; }
};

#define BZ_DECL_SLFNT(X,Y) \
 template<>                 \
 class _bz_stringLiteralForNumericType< X > {  \
 public:                                       \
     static const char* string()               \
     { return Y; }                             \
 }

#ifdef BZ_BOOL
BZ_DECL_SLFNT(bool, "bool");
#endif

BZ_DECL_SLFNT(char, "char");
BZ_DECL_SLFNT(unsigned char, "unsigned char");
BZ_DECL_SLFNT(short int, "short int");
BZ_DECL_SLFNT(short unsigned int, "short unsigned int");
BZ_DECL_SLFNT(int, "int");
BZ_DECL_SLFNT(unsigned int, "unsigned int");
BZ_DECL_SLFNT(long, "long");
BZ_DECL_SLFNT(unsigned long, "unsigned long");
BZ_DECL_SLFNT(float, "float");
BZ_DECL_SLFNT(double, "double");
BZ_DECL_SLFNT(long double, "long double");

#ifdef BZ_HAVE_COMPLEX
BZ_DECL_SLFNT(complex<float>, "complex<float>");
BZ_DECL_SLFNT(complex<double>, "complex<double>");
BZ_DECL_SLFNT(complex<long double>, "complex<long double>");
#endif

#define BZ_DEBUG_TEMPLATE_AS_STRING_LITERAL(X) \
    _bz_stringLiteralForNumericType<X>::string()

#endif // !BZ_RTTI

BZ_NAMESPACE_END

#endif // BZ_DEBUG_H
