/*
 * $Id$
 *
 * $Log$
 * Revision 1.2  2002/09/12 07:02:06  eric
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
 * Revision 1.6  2002/05/27 19:48:57  jcumming
 * Removed use of this->.  Member data_ of templated base class is now declared
 * in derived class Array.
 *
 * Revision 1.5  2002/05/23 00:15:43  jcumming
 * Fixed bug in Array::evaluateWithIndexTraversal1() by removing cast of
 * second argument to T_numtype in call to T_update::update().  This cast
 * will occur automatically when the update operation is performed.  This
 * fixes a problem reported by Masahiro Tatsumi <tatsumi@nfi.co.jp> in
 * which one could not assign a double to an Array of TinyVectors of double
 * without explicitly constructing a TinyVector of doubles on the right-hand
 * side.  Also fixed an unused variable warning emanating from the function
 * Array::evaluateWithFastTraversal() by moving the definition of local
 * variable "last" so that it is only seen if it is used.
 *
 * Revision 1.4  2002/03/06 16:18:34  patricg
 *
 * data_ replaced by this->data_
 *
 * Revision 1.3  2001/01/26 18:30:50  tveldhui
 * More source code reorganization to reduce compile times.
 *
 * Revision 1.2  2001/01/24 22:51:51  tveldhui
 * Reorganized #include orders to avoid including the huge Vector e.t.
 * implementation when using Array.
 *
 * Revision 1.1.1.1  2000/06/19 12:26:13  tveldhui
 * Imported sources
 *
 * Revision 1.2  1998/03/14 00:04:47  tveldhui
 * 0.2-alpha-05
 *
 * Revision 1.1  1998/02/25 20:04:01  tveldhui
 * Initial revision
 *
 */

#ifndef BZ_ARRAYEVAL_CC
#define BZ_ARRAYEVAL_CC

#ifndef BZ_ARRAY_H
 #error <blitz/array/eval.cc> must be included via <blitz/array.h>
#endif

BZ_NAMESPACE(blitz)

/*
 * Assign an expression to an array.  For performance reasons, there are
 * several traversal mechanisms:
 *
 * - Index traversal scans through the destination array in storage order.
 *   The expression is evaluated using a TinyVector<int,N> operand.  This
 *   version is used only when there are index placeholders in the expression
 *   (see <blitz/indexexpr.h>)
 * - Stack traversal also scans through the destination array in storage
 *   order.  However, push/pop stack iterators are used.
 * - Fast traversal follows a Hilbert (or other) space-filling curve to
 *   improve cache reuse for stencilling operations.  Currently, the
 *   space filling curves must be generated by calling 
 *   generateFastTraversalOrder(TinyVector<int,N_dimensions>).
 * - 2D tiled traversal follows a tiled traversal, to improve cache reuse
 *   for 2D stencils.  Space filling curves have too much overhead to use
 *   in two-dimensions.
 *
 * _bz_tryFastTraversal is a helper class.  Fast traversals are only
 * attempted if the expression looks like a stencil -- it's at least
 * three-dimensional, has at least six array operands, and there are
 * no index placeholders in the expression.  These are all things which
 * can be checked at compile time, so the if()/else() syntax has been
 * replaced with this class template.
 */

// Fast traversals require <set> from the ISO/ANSI C++ standard library
#ifdef BZ_HAVE_STD
#ifdef BZ_ARRAY_SPACE_FILLING_TRAVERSAL

template<_bz_bool canTryFastTraversal>
struct _bz_tryFastTraversal {
    template<class T_numtype, int N_rank, class T_expr, class T_update>
    static _bz_bool tryFast(Array<T_numtype,N_rank>& array, 
        BZ_ETPARM(T_expr) expr, T_update)
    {
        return _bz_false;
    }
};

template<>
struct _bz_tryFastTraversal<_bz_true> {
    template<class T_numtype, int N_rank, class T_expr, class T_update>
    static _bz_bool tryFast(Array<T_numtype,N_rank>& array, 
        BZ_ETPARM(T_expr) expr, T_update)
    {
        // See if there's an appropriate space filling curve available.
        // Currently fast traversals use an N-1 dimensional curve.  The
        // Nth dimension column corresponding to each point on the curve
        // is traversed in the normal fashion.
        TraversalOrderCollection<N_rank-1> traversals;
        TinyVector<int, N_rank - 1> traversalGridSize;

        for (int i=0; i < N_rank - 1; ++i)
            traversalGridSize[i] = array.length(array.ordering(i+1));

#ifdef BZ_DEBUG_TRAVERSE
cout << "traversalGridSize = " << traversalGridSize << endl;
cout.flush();
#endif

        const TraversalOrder<N_rank-1>* order =
            traversals.find(traversalGridSize);

        if (order)
        {
#ifdef BZ_DEBUG_TRAVERSE
    cerr << "Array<" << BZ_DEBUG_TEMPLATE_AS_STRING_LITERAL(T_numtype)
         << ", " << N_rank << ">: Using stack traversal" << endl;
#endif
            // A curve was available -- use fast traversal.
            array.evaluateWithFastTraversal(*order, expr, T_update());
            return _bz_true;
        }

        return _bz_false;
    }
};

#endif // BZ_ARRAY_SPACE_FILLING_TRAVERSAL
#endif // BZ_HAVE_STD

template<class T_numtype, int N_rank> template<class T_expr, class T_update>
inline Array<T_numtype, N_rank>& 
Array<T_numtype, N_rank>::evaluate(T_expr expr, 
    T_update)
{
    // Check that all arrays have the same shape
#ifdef BZ_DEBUG
    if (!expr.shapeCheck(shape()))
    {
      if (assertFailMode == _bz_false)
      {
        cerr << "[Blitz++] Shape check failed: Module " << __FILE__
             << " line " << __LINE__ << endl
             << "          Expression: ";
        prettyPrintFormat format(_bz_true);   // Use terse formatting
        string str;
        expr.prettyPrint(str, format);
        cerr << str << endl ;
      }

#if 0
// Shape dumping is broken by change to using string for prettyPrint
             << "          Shapes: " << shape() << " = ";
        prettyPrintFormat format2;
        format2.setDumpArrayShapesMode();
        expr.prettyPrint(cerr, format2);
        cerr << endl;
#endif
        BZ_PRE_FAIL;
    }
#endif

    BZPRECHECK(expr.shapeCheck(shape()),
        "Shape check failed." << endl << "Expression:");

    BZPRECHECK((T_expr::rank == N_rank) || (T_expr::numArrayOperands == 0), 
        "Assigned rank " << T_expr::rank << " expression to rank " 
        << N_rank << " array.");

    /*
     * Check that the arrays are not empty (e.g. length 0 arrays)
     * This fixes a bug found by Peter Bienstman, 6/16/99, where
     * Array<double,2> A(0,0),B(0,0); B=A(tensor::j,tensor::i);
     * went into an infinite loop.
     */

    if (numElements() == 0)
        return *this;

#ifdef BZ_DEBUG_TRAVERSE
cout << "T_expr::numIndexPlaceholders = " << T_expr::numIndexPlaceholders
     << endl; cout.flush();
#endif

    // Tau profiling code.  Provide Tau with a pretty-printed version of
    // the expression.
    // NEEDS_WORK-- use a static initializer somehow.

#ifdef BZ_TAU_PROFILING
    static string exprDescription;
    if (!exprDescription.length())   // faked static initializer
    {
        exprDescription = "A";
        prettyPrintFormat format(_bz_true);   // Terse mode on
        format.nextArrayOperandSymbol();
        T_update::prettyPrint(exprDescription);
        expr.prettyPrint(exprDescription, format);
    }
    TAU_PROFILE(" ", exprDescription, TAU_BLITZ);
#endif

    // Determine which evaluation mechanism to use 
    if (T_expr::numIndexPlaceholders > 0)
    {
        // The expression involves index placeholders, so have to
        // use index traversal rather than stack traversal.

        if (N_rank == 1)
            return evaluateWithIndexTraversal1(expr, T_update());
        else
            return evaluateWithIndexTraversalN(expr, T_update());
    }
    else {

        // If this expression looks like an array stencil, then attempt to
        // use a fast traversal order.
        // Fast traversals require <set> from the ISO/ANSI C++ standard
        // library.

#ifdef BZ_HAVE_STD
#ifdef BZ_ARRAY_SPACE_FILLING_TRAVERSAL

        enum { isStencil = (N_rank >= 3) && (T_expr::numArrayOperands > 6)
            && (T_expr::numIndexPlaceholders == 0) };

        if (_bz_tryFastTraversal<isStencil>::tryFast(*this, expr, T_update()))
            return *this;

#endif
#endif

#ifdef BZ_ARRAY_2D_STENCIL_TILING
        // Does this look like a 2-dimensional stencil on a largeish
        // array?

        if ((N_rank == 2) && (T_expr::numArrayOperands >= 5))
        {
            // Use a heuristic to determine whether a tiled traversal
            // is desirable.  First, estimate how much L1 cache is needed 
            // to achieve a high hit rate using the stack traversal.
            // Try to err on the side of using tiled traversal even when
            // it isn't strictly needed.

            // Assumptions:
            //    Stencil width 3
            //    3 arrays involved in stencil
            //    Uniform data type in arrays (all T_numtype)
            
            int cacheNeeded = 3 * 3 * sizeof(T_numtype) * length(ordering(0));
            if (cacheNeeded > BZ_L1_CACHE_ESTIMATED_SIZE)
                return evaluateWithTiled2DTraversal(expr, T_update());
        }

#endif

        // If fast traversal isn't available or appropriate, then just
        // do a stack traversal.
        if (N_rank == 1)
            return evaluateWithStackTraversal1(expr, T_update());
        else
            return evaluateWithStackTraversalN(expr, T_update());
    }
}

template<class T_numtype, int N_rank> template<class T_expr, class T_update>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::evaluateWithStackTraversal1(
    T_expr expr, T_update)
{
#ifdef BZ_DEBUG_TRAVERSE
    BZ_DEBUG_MESSAGE("Array<" << BZ_DEBUG_TEMPLATE_AS_STRING_LITERAL(T_numtype)
         << ", " << N_rank << ">: Using stack traversal");
#endif
    FastArrayIterator<T_numtype, N_rank> iter(*this);
    iter.loadStride(firstRank);
    expr.loadStride(firstRank);

    _bz_bool useUnitStride = iter.isUnitStride(firstRank)
          && expr.isUnitStride(firstRank);

#ifdef BZ_ARRAY_EXPR_USE_COMMON_STRIDE
    int commonStride = expr.suggestStride(firstRank);
    if (iter.suggestStride(firstRank) > commonStride)
        commonStride = iter.suggestStride(firstRank);
    bool useCommonStride = iter.isStride(firstRank,commonStride)
        && expr.isStride(firstRank,commonStride);

 #ifdef BZ_DEBUG_TRAVERSE
    BZ_DEBUG_MESSAGE("BZ_ARRAY_EXPR_USE_COMMON_STRIDE:" << endl
        << "    commonStride = " << commonStride << " useCommonStride = "
        << useCommonStride);
 #endif
#else
    int commonStride = 1;
    bool useCommonStride = _bz_false;
#endif

    const T_numtype * last = iter.data() + length(firstRank) 
        * stride(firstRank);

    if (useUnitStride || useCommonStride)
    {
#ifdef BZ_USE_FAST_READ_ARRAY_EXPR

#ifdef BZ_DEBUG_TRAVERSE
    BZ_DEBUG_MESSAGE("BZ_USE_FAST_READ_ARRAY_EXPR with commonStride");
#endif
        int ubound = length(firstRank) * commonStride;
        T_numtype* _bz_restrict data = const_cast<T_numtype*>(iter.data());

        if (commonStride == 1)
        {
 #ifndef BZ_ARRAY_STACK_TRAVERSAL_UNROLL
            for (int i=0; i < ubound; ++i)
                T_update::update(data[i], expr.fastRead(i));
 #else
            int n1 = ubound & 3;
            int i = 0;
            for (; i < n1; ++i)
                T_update::update(data[i], expr.fastRead(i));
           
            for (; i < ubound; i += 4)
            {
#ifndef BZ_ARRAY_STACK_TRAVERSAL_CSE_AND_ANTIALIAS
                T_update::update(data[i], expr.fastRead(i));
                T_update::update(data[i+1], expr.fastRead(i+1));
                T_update::update(data[i+2], expr.fastRead(i+2));
                T_update::update(data[i+3], expr.fastRead(i+3));
#else
                int t1 = i+1;
                int t2 = i+2;
                int t3 = i+3;

                _bz_typename T_expr::T_numtype tmp1, tmp2, tmp3, tmp4;

                tmp1 = expr.fastRead(i);
                tmp2 = expr.fastRead(BZ_NO_PROPAGATE(t1));
                tmp3 = expr.fastRead(BZ_NO_PROPAGATE(t2));
                tmp4 = expr.fastRead(BZ_NO_PROPAGATE(t3));

                T_update::update(data[i], BZ_NO_PROPAGATE(tmp1));
                T_update::update(data[BZ_NO_PROPAGATE(t1)], tmp2);
                T_update::update(data[BZ_NO_PROPAGATE(t2)], tmp3);
                T_update::update(data[BZ_NO_PROPAGATE(t3)], tmp4);
#endif
            }
 #endif // BZ_ARRAY_STACK_TRAVERSAL_UNROLL

        }
 #ifdef BZ_ARRAY_EXPR_USE_COMMON_STRIDE
        else {

  #ifndef BZ_ARRAY_STACK_TRAVERSAL_UNROLL
            for (int i=0; i < ubound; i += commonStride)
                T_update::update(data[i], expr.fastRead(i));
  #else
            int n1 = (length(firstRank) & 3) * commonStride;

            int i = 0;
            for (; i < n1; i += commonStride)
                T_update::update(data[i], expr.fastRead(i));

            int strideInc = 4 * commonStride;
            for (; i < ubound; i += strideInc)
            {
                T_update::update(data[i], expr.fastRead(i));
                int i2 = i + commonStride;
                T_update::update(data[i2], expr.fastRead(i2));
                int i3 = i + 2 * commonStride;
                T_update::update(data[i3], expr.fastRead(i3));
                int i4 = i + 3 * commonStride;
                T_update::update(data[i4], expr.fastRead(i4));
            }
  #endif  // BZ_ARRAY_STACK_TRAVERSAL_UNROLL
        }
 #endif  // BZ_ARRAY_EXPR_USE_COMMON_STRIDE

#else   // ! BZ_USE_FAST_READ_ARRAY_EXPR

#ifdef BZ_DEBUG_TRAVERSE
    BZ_DEBUG_MESSAGE("Common stride, no fast read");
#endif
        while (iter.data() != last)
        {
            T_update::update(*const_cast<T_numtype*>(iter.data()), *expr);
            iter.advance(commonStride);
            expr.advance(commonStride);
        }
#endif
    }
    else {
        while (iter.data() != last)
        {
            T_update::update(*const_cast<T_numtype*>(iter.data()), *expr);
            iter.advance();
            expr.advance();
        }
    }

    return *this;
}

template<class T_numtype, int N_rank> template<class T_expr, class T_update>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::evaluateWithStackTraversalN(
    T_expr expr, T_update)
{
    /*
     * A stack traversal replaces the usual nested loops:
     *
     * for (int i=A.lbound(firstDim); i <= A.ubound(firstDim); ++i)
     *   for (int j=A.lbound(secondDim); j <= A.ubound(secondDim); ++j)
     *     for (int k=A.lbound(thirdDim); k <= A.ubound(thirdDim); ++k)
     *       A(i,j,k) = 0;
     *
     * with a stack data structure.  The stack allows this single
     * routine to replace any number of nested loops.
     *
     * For each dimension (loop), these quantities are needed:
     * - a pointer to the first element encountered in the loop
     * - the stride associated with the dimension/loop
     * - a pointer to the last element encountered in the loop
     *
     * The basic idea is that entering each loop is a "push" onto the
     * stack, and exiting each loop is a "pop".  In practice, this
     * routine treats accesses the stack in a random-access way,
     * which confuses the picture a bit.  But conceptually, that's
     * what is going on.
     */

    /*
     * ordering(0) gives the dimension associated with the smallest
     * stride (usually; the exceptions have to do with subarrays and
     * are uninteresting).  We call this dimension maxRank; it will
     * become the innermost "loop".
     *
     * Ordering the loops from ordering(N_rank-1) down to
     * ordering(0) ensures that the largest stride is associated
     * with the outermost loop, and the smallest stride with the
     * innermost.  This is critical for good performance on
     * cached machines.
     */

    const int maxRank = ordering(0);
    const int secondLastRank = ordering(1);

    // Create an iterator for the array receiving the result
    FastArrayIterator<T_numtype, N_rank> iter(*this);

    // Set the initial stack configuration by pushing the pointer
    // to the first element of the array onto the stack N times.

    int i;
    for (i=1; i < N_rank; ++i)
    {
        iter.push(i);
        expr.push(i);
    }

    // Load the strides associated with the innermost loop.
    iter.loadStride(maxRank);
    expr.loadStride(maxRank);

    /* 
     * Is the stride in the innermost loop equal to 1?  If so,
     * we might take advantage of this and generate more
     * efficient code.
     */
    _bz_bool useUnitStride = iter.isUnitStride(maxRank)
                          && expr.isUnitStride(maxRank);

    /*
     * Do all array operands share a common stride in the innermost
     * loop?  If so, we can generate more efficient code (but only
     * if this optimization has been enabled).
     */
#ifdef BZ_ARRAY_EXPR_USE_COMMON_STRIDE
    int commonStride = expr.suggestStride(maxRank);
    if (iter.suggestStride(maxRank) > commonStride)
        commonStride = iter.suggestStride(maxRank);
    bool useCommonStride = iter.isStride(maxRank,commonStride)
        && expr.isStride(maxRank,commonStride);

#ifdef BZ_DEBUG_TRAVERSE
    BZ_DEBUG_MESSAGE("BZ_ARRAY_EXPR_USE_COMMON_STRIDE" << endl
        << "commonStride = " << commonStride << " useCommonStride = "
        << useCommonStride);
#endif

#else
    int commonStride = 1;
    bool useCommonStride = _bz_false;
#endif

    /*
     * The "last" array contains a pointer to the last element
     * encountered in each "loop".
     */
    const T_numtype* last[N_rank];

    // Set up the initial state of the "last" array
    for (i=1; i < N_rank; ++i)
        last[i] = iter.data() + length(ordering(i)) * stride(ordering(i));

    int lastLength = length(maxRank);
    int firstNoncollapsedLoop = 1;

#ifdef BZ_COLLAPSE_LOOPS

    /*
     * This bit of code handles collapsing loops.  When possible,
     * the N nested loops are converted into a single loop (basically,
     * the N-dimensional array is treated as a long vector).
     * This is important for cases where the length of the innermost
     * loop is very small, for example a 100x100x3 array.
     * If this code can't collapse all the loops into a single loop,
     * it will collapse as many loops as possible starting from the
     * innermost and working out.
     */

    // Collapse loops when possible
    for (i=1; i < N_rank; ++i)
    {
        // Figure out which pair of loops we are considering combining.
        int outerLoopRank = ordering(i);
        int innerLoopRank = ordering(i-1);

        /*
         * The canCollapse() routines look at the strides and extents
         * of the loops, and determine if they can be combined into
         * one loop.
         */

        if (canCollapse(outerLoopRank,innerLoopRank) 
          && expr.canCollapse(outerLoopRank,innerLoopRank))
        {
#ifdef BZ_DEBUG_TRAVERSE
            cout << "Collapsing " << outerLoopRank << " and " 
                 << innerLoopRank << endl;
#endif
            lastLength *= length(outerLoopRank);
            firstNoncollapsedLoop = i+1;
        }
        else  
            break;
    }

#endif // BZ_COLLAPSE_LOOPS

    /*
     * Now we actually perform the loops.  This while loop contains
     * two parts: first, the innermost loop is performed.  Then we
     * exit the loop, and pop our way down the stack until we find
     * a loop that isn't completed.  We then restart the inner loops
     * and push them onto the stack.
     */

    while (true) {

        /*
         * This bit of code handles the innermost loop.  It would look
         * a lot simpler if it weren't for unit stride and common stride
         * optimizations; these clutter up the code with multiple versions.
         */

        if ((useUnitStride) || (useCommonStride))
        {
            T_numtype * _bz_restrict end = const_cast<T_numtype*>(iter.data()) 
                + lastLength;

#ifdef BZ_USE_FAST_READ_ARRAY_EXPR

            /*
             * The check for BZ_USE_FAST_READ_ARRAY_EXPR can probably
             * be taken out.  This was put in place while the unit stride/
             * common stride optimizations were being implemented and
             * tested.
             */

            // Calculate the end of the innermost loop
            int ubound = lastLength * commonStride;

            /*
             * This is a real kludge.  I didn't want to have to write
             * a const and non-const version of FastArrayIterator, so I use a
             * const iterator and cast away const.  This could
             * probably be avoided with some trick, but the whole routine
             * is ugly, so why bother.
             */

            T_numtype* _bz_restrict data = const_cast<T_numtype*>(iter.data());

            /*
             * BZ_NEEDS_WORK-- need to implement optional unrolling.
             */
            if (commonStride == 1)
            {
                for (int i=0; i < ubound; ++i)
                    T_update::update(data[i], expr.fastRead(i));
            }
#ifdef BZ_ARRAY_EXPR_USE_COMMON_STRIDE
            else {
                for (int i=0; i < ubound; i += commonStride)
                    T_update::update(data[i], expr.fastRead(i));
            }
#endif
            /*
             * Tidy up for the fact that we haven't actually been
             * incrementing the iterators in the innermost loop, by
             * faking it afterward.
             */
            iter.advance(lastLength * commonStride);
            expr.advance(lastLength * commonStride);
#else        
            // !BZ_USE_FAST_READ_ARRAY_EXPR
            // This bit of code not really needed; should remove at some
            // point, along with the test for BZ_USE_FAST_READ_ARRAY_EXPR 

            while (iter.data() != end) 
            {
                T_update::update(*const_cast<T_numtype*>(iter.data()), *expr);
                iter.advance(commonStride);
                expr.advance(commonStride);
            }
#endif
        }
        else {
            /*
             * We don't have a unit stride or common stride in the innermost
             * loop.  This is going to hurt performance.  Luckily 95% of
             * the time, we hit the cases above.
             */
            T_numtype * _bz_restrict end = const_cast<T_numtype*>(iter.data())
                + lastLength * stride(maxRank);

            while (iter.data() != end)
            {
                T_update::update(*const_cast<T_numtype*>(iter.data()), *expr);
                iter.advance();
                expr.advance();
            }
        }


        /*
         * We just finished the innermost loop.  Now we pop our way down
         * the stack, until we hit a loop that hasn't completed yet.
         */ 
        int j = firstNoncollapsedLoop;
        for (; j < N_rank; ++j)
        {
            // Get the next loop
            int r = ordering(j);

            // Pop-- this restores the data pointers to the first element
            // encountered in the loop.
            iter.pop(j);
            expr.pop(j);

            // Load the stride associated with this loop, and increment
            // once.
            iter.loadStride(r);
            expr.loadStride(r);
            iter.advance();
            expr.advance();

            // If we aren't at the end of this loop, then stop popping.
            if (iter.data() != last[j])
                break;
        }

        // Are we completely done?
        if (j == N_rank)
            break;

        // No, so push all the inner loops back onto the stack.
        for (; j >= firstNoncollapsedLoop; --j)
        {
            int r2 = ordering(j-1);
            iter.push(j);
            expr.push(j);
            last[j-1] = iter.data() + length(r2) * stride(r2);
        }

        // Load the stride for the innermost loop again.
        iter.loadStride(maxRank);
        expr.loadStride(maxRank);
    }

    return *this;
}

template<class T_numtype, int N_rank> template<class T_expr, class T_update>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::evaluateWithIndexTraversal1(
    T_expr expr, T_update)
{
    TinyVector<int,N_rank> index;

    if (stride(firstRank) == 1)
    {
        T_numtype * _bz_restrict iter = data_;
        int last = ubound(firstRank);

        for (index[0] = lbound(firstRank); index[0] <= last;
            ++index[0])
        {
            T_update::update(iter[index[0]], expr(index));
        }
    }
    else {
        FastArrayIterator<T_numtype, N_rank> iter(*this);
        iter.loadStride(0);
        int last = ubound(firstRank);

        for (index[0] = lbound(firstRank); index[0] <= last;
            ++index[0])
        {
            T_update::update(*const_cast<T_numtype*>(iter.data()), 
                expr(index));
            iter.advance();
        }
    }

    return *this;
}

template<class T_numtype, int N_rank> template<class T_expr, class T_update>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::evaluateWithIndexTraversalN(
    T_expr expr, T_update)
{
    // Do a stack-type traversal for the destination array and use
    // index traversal for the source expression
   
    const int maxRank = ordering(0);
    const int secondLastRank = ordering(1);

#ifdef BZ_DEBUG_TRAVERSE
cout << "Index traversal: N_rank = " << N_rank << endl;
cout << "maxRank = " << maxRank << " secondLastRank = " << secondLastRank
     << endl;
cout.flush();
#endif

    FastArrayIterator<T_numtype, N_rank> iter(*this);
    for (int i=1; i < N_rank; ++i)
        iter.push(ordering(i));

    iter.loadStride(maxRank);

    TinyVector<int,N_rank> index, last;

    index = storage_.base();

    for (int i=0; i < N_rank; ++i)
      last(i) = storage_.base(i) + length_(i);

    int lastLength = length(maxRank);

    while (true) {

        for (index[maxRank] = base(maxRank); 
             index[maxRank] < last[maxRank]; 
             ++index[maxRank])
        {
#ifdef BZ_DEBUG_TRAVERSE
#if 0
    cout << "(" << index[0] << "," << index[1] << ") " << endl;
    cout.flush();
#endif
#endif

            T_update::update(*const_cast<T_numtype*>(iter.data()), expr(index));
            iter.advance();
        }

        int j = 1;
        for (; j < N_rank; ++j)
        {
            iter.pop(ordering(j));
            iter.loadStride(ordering(j));
            iter.advance();

            index[ordering(j-1)] = base(ordering(j-1));
            ++index[ordering(j)];
            if (index[ordering(j)] != last[ordering(j)])
                break;
        }

        if (j == N_rank)
            break;

        for (; j > 0; --j)
        {
            iter.push(ordering(j));
        }
        iter.loadStride(maxRank);
    }

    return *this; 
}

// Fast traversals require <set> from the ISO/ANSI C++ standard library

#ifdef BZ_HAVE_STD
#ifdef BZ_ARRAY_SPACE_FILLING_TRAVERSAL

template<class T_numtype, int N_rank> template<class T_expr, class T_update>
inline Array<T_numtype, N_rank>&
Array<T_numtype, N_rank>::evaluateWithFastTraversal(
    const TraversalOrder<N_rank - 1>& order, 
    T_expr expr,
    T_update)
{
    const int maxRank = ordering(0);
    const int secondLastRank = ordering(1);

#ifdef BZ_DEBUG_TRAVERSE
cerr << "maxRank = " << maxRank << " secondLastRank = " << secondLastRank
     << endl;
#endif

    FastArrayIterator<T_numtype, N_rank> iter(*this);
    iter.push(0);
    expr.push(0);

    _bz_bool useUnitStride = iter.isUnitStride(maxRank) 
                          && expr.isUnitStride(maxRank);

#ifdef BZ_ARRAY_EXPR_USE_COMMON_STRIDE
    int commonStride = expr.suggestStride(maxRank);
    if (iter.suggestStride(maxRank) > commonStride)
        commonStride = iter.suggestStride(maxRank);
    bool useCommonStride = iter.isStride(maxRank,commonStride)
        && expr.isStride(maxRank,commonStride);
#else
    int commonStride = 1;
    bool useCommonStride = _bz_false;
#endif

    int lastLength = length(maxRank);

    for (int i=0; i < order.length(); ++i)
    {
        iter.pop(0);
        expr.pop(0);

#ifdef BZ_DEBUG_TRAVERSE
    cerr << "Traversing: " << order[i] << endl;
#endif
        // Position the iterator at the start of the next column       
        for (int j=1; j < N_rank; ++j)
        {
            iter.loadStride(ordering(j));
            expr.loadStride(ordering(j));

            int offset = order[i][j-1];
            iter.advance(offset);
            expr.advance(offset);
        }

        iter.loadStride(maxRank);
        expr.loadStride(maxRank);

        // Evaluate the expression along the column

        if ((useUnitStride) || (useCommonStride))
        {
#ifdef BZ_USE_FAST_READ_ARRAY_EXPR
            int ubound = lastLength * commonStride;
            T_numtype* _bz_restrict data = const_cast<T_numtype*>(iter.data());

            if (commonStride == 1)
            {            
 #ifndef BZ_ARRAY_FAST_TRAVERSAL_UNROLL
                for (int i=0; i < ubound; ++i)
                    T_update::update(data[i], expr.fastRead(i));
 #else
                int n1 = ubound & 3;
                int i=0;
                for (; i < n1; ++i)
                    T_update::update(data[i], expr.fastRead(i));

                for (; i < ubound; i += 4)
                {
                    T_update::update(data[i], expr.fastRead(i));
                    T_update::update(data[i+1], expr.fastRead(i+1));
                    T_update::update(data[i+2], expr.fastRead(i+2));
                    T_update::update(data[i+3], expr.fastRead(i+3));
                }
 #endif  // BZ_ARRAY_FAST_TRAVERSAL_UNROLL
            }
 #ifdef BZ_ARRAY_EXPR_USE_COMMON_STRIDE
            else {
                for (int i=0; i < ubound; i += commonStride)
                    T_update::update(data[i], expr.fastRead(i));
            }
 #endif // BZ_ARRAY_EXPR_USE_COMMON_STRIDE

            iter.advance(lastLength * commonStride);
            expr.advance(lastLength * commonStride);
#else   // ! BZ_USE_FAST_READ_ARRAY_EXPR
            T_numtype* _bz_restrict last = const_cast<T_numtype*>(iter.data()) 
                + lastLength * commonStride;

            while (iter.data() != last)
            {
                T_update::update(*const_cast<T_numtype*>(iter.data()), *expr);
                iter.advance(commonStride);
                expr.advance(commonStride);
            }
#endif  // BZ_USE_FAST_READ_ARRAY_EXPR

        }
        else {
            // No common stride

            T_numtype* _bz_restrict last = const_cast<T_numtype*>(iter.data()) 
                + lastLength * stride(maxRank);

            while (iter.data() != last)
            {
                T_update::update(*const_cast<T_numtype*>(iter.data()), *expr);
                iter.advance();
                expr.advance();
            }
        }
    }

    return *this;
}

#endif // BZ_ARRAY_SPACE_FILLING_TRAVERSAL
#endif // BZ_HAVE_STD

#ifdef BZ_ARRAY_2D_NEW_STENCIL_TILING

#ifdef BZ_ARRAY_2D_STENCIL_TILING

template<class T_numtype, int N_rank> template<class T_expr, class T_update>
inline Array<T_numtype, N_rank>& 
Array<T_numtype, N_rank>::evaluateWithTiled2DTraversal(
    T_expr expr, T_update)
{
    const int minorRank = ordering(0);
    const int majorRank = ordering(1);

    FastArrayIterator<T_numtype, N_rank> iter(*this);
    iter.push(0);
    expr.push(0);

#ifdef BZ_2D_STENCIL_DEBUG
    int count = 0;
#endif

    _bz_bool useUnitStride = iter.isUnitStride(minorRank)
                          && expr.isUnitStride(minorRank);

#ifdef BZ_ARRAY_EXPR_USE_COMMON_STRIDE
    int commonStride = expr.suggestStride(minorRank);
    if (iter.suggestStride(minorRank) > commonStride)
        commonStride = iter.suggestStride(minorRank);
    bool useCommonStride = iter.isStride(minorRank,commonStride)
        && expr.isStride(minorRank,commonStride);
#else
    int commonStride = 1;
    bool useCommonStride = _bz_false;
#endif

    // Determine if a common major stride exists
    int commonMajorStride = expr.suggestStride(majorRank);
    if (iter.suggestStride(majorRank) > commonMajorStride)
        commonMajorStride = iter.suggestStride(majorRank);
    bool haveCommonMajorStride = iter.isStride(majorRank,commonMajorStride)
        && expr.isStride(majorRank,commonMajorStride);


    int maxi = length(majorRank);
    int maxj = length(minorRank);

    const int tileHeight = 16, tileWidth = 3;

    int bi, bj;
    for (bi=0; bi < maxi; bi += tileHeight)
    {
        int ni = bi + tileHeight;
        if (ni > maxi)
            ni = maxi;

        // Move back to the beginning of the array
        iter.pop(0);
        expr.pop(0);

        // Move to the start of this tile row
        iter.loadStride(majorRank);
        iter.advance(bi);
        expr.loadStride(majorRank);
        expr.advance(bi);

        // Save this position
        iter.push(1);
        expr.push(1);

        for (bj=0; bj < maxj; bj += tileWidth)
        {
            // Move to the beginning of the tile row
            iter.pop(1);
            expr.pop(1);

            // Move to the top of the current tile (bi,bj)
            iter.loadStride(minorRank);
            iter.advance(bj);
            expr.loadStride(minorRank);
            expr.advance(bj);

            if (bj + tileWidth <= maxj)
            {
                // Strip mining

                if ((useUnitStride) && (haveCommonMajorStride))
                {
                    int offset = 0;
                    T_numtype* _bz_restrict data = const_cast<T_numtype*>
                        (iter.data());

                    for (int i=bi; i < ni; ++i)
                    {
                        _bz_typename T_expr::T_numtype tmp1, tmp2, tmp3;

                        // Common subexpression elimination -- compilers
                        // won't necessarily do this on their own.
                        int t1 = offset+1;
                        int t2 = offset+2;

                        tmp1 = expr.fastRead(offset);
                        tmp2 = expr.fastRead(t1);
                        tmp3 = expr.fastRead(t2);

                        T_update::update(data[0], tmp1);
                        T_update::update(data[1], tmp2);
                        T_update::update(data[2], tmp3);

                        offset += commonMajorStride;
                        data += commonMajorStride;

#ifdef BZ_2D_STENCIL_DEBUG
    count += 3;
#endif
                    }
                }
                else {

                    for (int i=bi; i < ni; ++i)
                    {
                        iter.loadStride(minorRank);
                        expr.loadStride(minorRank);

                        // Loop through current row elements
                        T_update::update(*const_cast<T_numtype*>(iter.data()),
                            *expr);
                        iter.advance();
                        expr.advance();

                        T_update::update(*const_cast<T_numtype*>(iter.data()),
                            *expr);
                        iter.advance();
                        expr.advance();

                        T_update::update(*const_cast<T_numtype*>(iter.data()),
                            *expr);
                        iter.advance(-2);
                        expr.advance(-2);

                        iter.loadStride(majorRank);
                        expr.loadStride(majorRank);
                        iter.advance();
                        expr.advance();

#ifdef BZ_2D_STENCIL_DEBUG
    count += 3;
#endif

                    }
                }
            }
            else {

                // This code handles partial tiles at the bottom of the
                // array.

                for (int j=bj; j < maxj; ++j)
                {
                    iter.loadStride(majorRank);
                    expr.loadStride(majorRank);

                    for (int i=bi; i < ni; ++i)
                    {
                        T_update::update(*const_cast<T_numtype*>(iter.data()),
                            *expr);
                        iter.advance();
                        expr.advance();
#ifdef BZ_2D_STENCIL_DEBUG
    ++count;
#endif

                    }

                    // Move back to the top of this column
                    iter.advance(bi-ni);
                    expr.advance(bi-ni);

                    // Move over to the next column
                    iter.loadStride(minorRank);
                    expr.loadStride(minorRank);

                    iter.advance();
                    expr.advance();
                }
            }
        }
    }

#ifdef BZ_2D_STENCIL_DEBUG
    cout << "BZ_2D_STENCIL_DEBUG: count = " << count << endl;
#endif

    return *this;
}

#endif // BZ_ARRAY_2D_STENCIL_TILING
#endif // BZ_ARRAY_2D_NEW_STENCIL_TILING



#ifndef BZ_ARRAY_2D_NEW_STENCIL_TILING

#ifdef BZ_ARRAY_2D_STENCIL_TILING

template<class T_numtype, int N_rank> template<class T_expr, class T_update>
inline Array<T_numtype, N_rank>& 
Array<T_numtype, N_rank>::evaluateWithTiled2DTraversal(
    T_expr expr, T_update)
{
    const int minorRank = ordering(0);
    const int majorRank = ordering(1);

    const int blockSize = 16;
    
    FastArrayIterator<T_numtype, N_rank> iter(*this);
    iter.push(0);
    expr.push(0);

    _bz_bool useUnitStride = iter.isUnitStride(minorRank)
                          && expr.isUnitStride(minorRank);

#ifdef BZ_ARRAY_EXPR_USE_COMMON_STRIDE
    int commonStride = expr.suggestStride(minorRank);
    if (iter.suggestStride(minorRank) > commonStride)
        commonStride = iter.suggestStride(minorRank);
    bool useCommonStride = iter.isStride(minorRank,commonStride)
        && expr.isStride(minorRank,commonStride);
#else
    int commonStride = 1;
    bool useCommonStride = _bz_false;
#endif

    int maxi = length(majorRank);
    int maxj = length(minorRank);

    int bi, bj;
    for (bi=0; bi < maxi; bi += blockSize)
    {
        int ni = bi + blockSize;
        if (ni > maxi)
            ni = maxi;

        for (bj=0; bj < maxj; bj += blockSize)
        {
            int nj = bj + blockSize;
            if (nj > maxj)
                nj = maxj;

            // Move to the beginning of the array
            iter.pop(0);
            expr.pop(0);

            // Move to the beginning of the tile (bi,bj)
            iter.loadStride(majorRank);
            iter.advance(bi);
            iter.loadStride(minorRank);
            iter.advance(bj);

            expr.loadStride(majorRank);
            expr.advance(bi);
            expr.loadStride(minorRank);
            expr.advance(bj);

            // Loop through tile rows
            for (int i=bi; i < ni; ++i)
            {
                // Save the beginning of this tile row
                iter.push(1);
                expr.push(1);

                // Load the minor stride
                iter.loadStride(minorRank);
                expr.loadStride(minorRank);

                if (useUnitStride)
                {
                    T_numtype* _bz_restrict data = const_cast<T_numtype*>
                        (iter.data());

                    int ubound = (nj-bj);
                    for (int j=0; j < ubound; ++j)
                        T_update::update(data[j], expr.fastRead(j));
                }
#ifdef BZ_ARRAY_EXPR_USE_COMMON_STRIDE
                else if (useCommonStride)
                {
                    int ubound = (nj-bj) * commonStride;
                    T_numtype* _bz_restrict data = const_cast<T_numtype*>
                        (iter.data());

                    for (int j=0; j < ubound; j += commonStride)
                        T_update::update(data[j], expr.fastRead(j));
                }
#endif
                else {
                    for (int j=bj; j < nj; ++j)
                    {
                        // Loop through current row elements
                        T_update::update(*const_cast<T_numtype*>(iter.data()), 
                            *expr);
                        iter.advance();
                        expr.advance();
                    }
                }

                // Move back to the beginning of the tile row, then
                // move to the next row
                iter.pop(1);
                iter.loadStride(majorRank);
                iter.advance(1);

                expr.pop(1);
                expr.loadStride(majorRank);
                expr.advance(1);
            }
        }
    }

    return *this;
}
#endif // BZ_ARRAY_2D_STENCIL_TILING
#endif // BZ_ARRAY_2D_NEW_STENCIL_TILING

BZ_NAMESPACE_END

#endif // BZ_ARRAYEVAL_CC

