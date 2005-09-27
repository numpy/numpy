      subroutine lmstr(fcn,m,n,x,fvec,fjac,ldfjac,ftol,xtol,gtol,
     *                 maxfev,diag,mode,factor,nprint,info,nfev,njev,
     *                 ipvt,qtf,wa1,wa2,wa3,wa4)
      integer m,n,ldfjac,maxfev,mode,nprint,info,nfev,njev
      integer ipvt(n)
      logical sing
      double precision ftol,xtol,gtol,factor
      double precision x(n),fvec(m),fjac(ldfjac,n),diag(n),qtf(n),
     *                 wa1(n),wa2(n),wa3(n),wa4(m)
c     **********
c
c     subroutine lmstr
c
c     the purpose of lmstr is to minimize the sum of the squares of
c     m nonlinear functions in n variables by a modification of
c     the levenberg-marquardt algorithm which uses minimal storage.
c     the user must provide a subroutine which calculates the
c     functions and the rows of the jacobian.
c
c     the subroutine statement is
c
c       subroutine lmstr(fcn,m,n,x,fvec,fjac,ldfjac,ftol,xtol,gtol,
c                        maxfev,diag,mode,factor,nprint,info,nfev,
c                        njev,ipvt,qtf,wa1,wa2,wa3,wa4)
c
c     where
c
c       fcn is the name of the user-supplied subroutine which
c         calculates the functions and the rows of the jacobian.
c         fcn must be declared in an external statement in the
c         user calling program, and should be written as follows.
c
c         subroutine fcn(m,n,x,fvec,fjrow,iflag)
c         integer m,n,iflag
c         double precision x(n),fvec(m),fjrow(n)
c         ----------
c         if iflag = 1 calculate the functions at x and
c         return this vector in fvec.
c         if iflag = i calculate the (i-1)-st row of the
c         jacobian at x and return this vector in fjrow.
c         ----------
c         return
c         end
c
c         the value of iflag should not be changed by fcn unless
c         the user wants to terminate execution of lmstr.
c         in this case set iflag to a negative integer.
c
c       m is a positive integer input variable set to the number
c         of functions.
c
c       n is a positive integer input variable set to the number
c         of variables. n must not exceed m.
c
c       x is an array of length n. on input x must contain
c         an initial estimate of the solution vector. on output x
c         contains the final estimate of the solution vector.
c
c       fvec is an output array of length m which contains
c         the functions evaluated at the output x.
c
c       fjac is an output n by n array. the upper triangle of fjac
c         contains an upper triangular matrix r such that
c
c                t     t           t
c               p *(jac *jac)*p = r *r,
c
c         where p is a permutation matrix and jac is the final
c         calculated jacobian. column j of p is column ipvt(j)
c         (see below) of the identity matrix. the lower triangular
c         part of fjac contains information generated during
c         the computation of r.
c
c       ldfjac is a positive integer input variable not less than n
c         which specifies the leading dimension of the array fjac.
c
c       ftol is a nonnegative input variable. termination
c         occurs when both the actual and predicted relative
c         reductions in the sum of squares are at most ftol.
c         therefore, ftol measures the relative error desired
c         in the sum of squares.
c
c       xtol is a nonnegative input variable. termination
c         occurs when the relative error between two consecutive
c         iterates is at most xtol. therefore, xtol measures the
c         relative error desired in the approximate solution.
c
c       gtol is a nonnegative input variable. termination
c         occurs when the cosine of the angle between fvec and
c         any column of the jacobian is at most gtol in absolute
c         value. therefore, gtol measures the orthogonality
c         desired between the function vector and the columns
c         of the jacobian.
c
c       maxfev is a positive integer input variable. termination
c         occurs when the number of calls to fcn with iflag = 1
c         has reached maxfev.
c
c       diag is an array of length n. if mode = 1 (see
c         below), diag is internally set. if mode = 2, diag
c         must contain positive entries that serve as
c         multiplicative scale factors for the variables.
c
c       mode is an integer input variable. if mode = 1, the
c         variables will be scaled internally. if mode = 2,
c         the scaling is specified by the input diag. other
c         values of mode are equivalent to mode = 1.
c
c       factor is a positive input variable used in determining the
c         initial step bound. this bound is set to the product of
c         factor and the euclidean norm of diag*x if nonzero, or else
c         to factor itself. in most cases factor should lie in the
c         interval (.1,100.). 100. is a generally recommended value.
c
c       nprint is an integer input variable that enables controlled
c         printing of iterates if it is positive. in this case,
c         fcn is called with iflag = 0 at the beginning of the first
c         iteration and every nprint iterations thereafter and
c         immediately prior to return, with x and fvec available
c         for printing. if nprint is not positive, no special calls
c         of fcn with iflag = 0 are made.
c
c       info is an integer output variable. if the user has
c         terminated execution, info is set to the (negative)
c         value of iflag. see description of fcn. otherwise,
c         info is set as follows.
c
c         info = 0  improper input parameters.
c
c         info = 1  both actual and predicted relative reductions
c                   in the sum of squares are at most ftol.
c
c         info = 2  relative error between two consecutive iterates
c                   is at most xtol.
c
c         info = 3  conditions for info = 1 and info = 2 both hold.
c
c         info = 4  the cosine of the angle between fvec and any
c                   column of the jacobian is at most gtol in
c                   absolute value.
c
c         info = 5  number of calls to fcn with iflag = 1 has
c                   reached maxfev.
c
c         info = 6  ftol is too small. no further reduction in
c                   the sum of squares is possible.
c
c         info = 7  xtol is too small. no further improvement in
c                   the approximate solution x is possible.
c
c         info = 8  gtol is too small. fvec is orthogonal to the
c                   columns of the jacobian to machine precision.
c
c       nfev is an integer output variable set to the number of
c         calls to fcn with iflag = 1.
c
c       njev is an integer output variable set to the number of
c         calls to fcn with iflag = 2.
c
c       ipvt is an integer output array of length n. ipvt
c         defines a permutation matrix p such that jac*p = q*r,
c         where jac is the final calculated jacobian, q is
c         orthogonal (not stored), and r is upper triangular.
c         column j of p is column ipvt(j) of the identity matrix.
c
c       qtf is an output array of length n which contains
c         the first n elements of the vector (q transpose)*fvec.
c
c       wa1, wa2, and wa3 are work arrays of length n.
c
c       wa4 is a work array of length m.
c
c     subprograms called
c
c       user-supplied ...... fcn
c
c       minpack-supplied ... dpmpar,enorm,lmpar,qrfac,rwupdt
c
c       fortran-supplied ... dabs,dmax1,dmin1,dsqrt,mod
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, dudley v. goetschel, kenneth e. hillstrom,
c     jorge j. more
c
c     **********
      integer i,iflag,iter,j,l
      double precision actred,delta,dirder,epsmch,fnorm,fnorm1,gnorm,
     *                 one,par,pnorm,prered,p1,p5,p25,p75,p0001,ratio,
     *                 sum,temp,temp1,temp2,xnorm,zero
      double precision dpmpar,enorm
      data one,p1,p5,p25,p75,p0001,zero
     *     /1.0d0,1.0d-1,5.0d-1,2.5d-1,7.5d-1,1.0d-4,0.0d0/
c
c     epsmch is the machine precision.
c
      epsmch = dpmpar(1)
c
      info = 0
      iflag = 0
      nfev = 0
      njev = 0
c
c     check the input parameters for errors.
c
      if (n .le. 0 .or. m .lt. n .or. ldfjac .lt. n
     *    .or. ftol .lt. zero .or. xtol .lt. zero .or. gtol .lt. zero
     *    .or. maxfev .le. 0 .or. factor .le. zero) go to 340
      if (mode .ne. 2) go to 20
      do 10 j = 1, n
         if (diag(j) .le. zero) go to 340
   10    continue
   20 continue
c
c     evaluate the function at the starting point
c     and calculate its norm.
c
      iflag = 1
      call fcn(m,n,x,fvec,wa3,iflag)
      nfev = 1
      if (iflag .lt. 0) go to 340
      fnorm = enorm(m,fvec)
c
c     initialize levenberg-marquardt parameter and iteration counter.
c
      par = zero
      iter = 1
c
c     beginning of the outer loop.
c
   30 continue
c
c        if requested, call fcn to enable printing of iterates.
c
         if (nprint .le. 0) go to 40
         iflag = 0
         if (mod(iter-1,nprint) .eq. 0) call fcn(m,n,x,fvec,wa3,iflag)
         if (iflag .lt. 0) go to 340
   40    continue
c
c        compute the qr factorization of the jacobian matrix
c        calculated one row at a time, while simultaneously
c        forming (q transpose)*fvec and storing the first
c        n components in qtf.
c
         do 60 j = 1, n
            qtf(j) = zero
            do 50 i = 1, n
               fjac(i,j) = zero
   50          continue
   60       continue
         iflag = 2
         do 70 i = 1, m
            call fcn(m,n,x,fvec,wa3,iflag)
            if (iflag .lt. 0) go to 340
            temp = fvec(i)
            call rwupdt(n,fjac,ldfjac,wa3,qtf,temp,wa1,wa2)
            iflag = iflag + 1
   70       continue
         njev = njev + 1
c
c        if the jacobian is rank deficient, call qrfac to
c        reorder its columns and update the components of qtf.
c
         sing = .false.
         do 80 j = 1, n
            if (fjac(j,j) .eq. zero) sing = .true.
            ipvt(j) = j
            wa2(j) = enorm(j,fjac(1,j))
   80       continue
         if (.not.sing) go to 130
         call qrfac(n,n,fjac,ldfjac,.true.,ipvt,n,wa1,wa2,wa3)
         do 120 j = 1, n
            if (fjac(j,j) .eq. zero) go to 110
            sum = zero
            do 90 i = j, n
               sum = sum + fjac(i,j)*qtf(i)
   90          continue
            temp = -sum/fjac(j,j)
            do 100 i = j, n
               qtf(i) = qtf(i) + fjac(i,j)*temp
  100          continue
  110       continue
            fjac(j,j) = wa1(j)
  120       continue
  130    continue
c
c        on the first iteration and if mode is 1, scale according
c        to the norms of the columns of the initial jacobian.
c
         if (iter .ne. 1) go to 170
         if (mode .eq. 2) go to 150
         do 140 j = 1, n
            diag(j) = wa2(j)
            if (wa2(j) .eq. zero) diag(j) = one
  140       continue
  150    continue
c
c        on the first iteration, calculate the norm of the scaled x
c        and initialize the step bound delta.
c
         do 160 j = 1, n
            wa3(j) = diag(j)*x(j)
  160       continue
         xnorm = enorm(n,wa3)
         delta = factor*xnorm
         if (delta .eq. zero) delta = factor
  170    continue
c
c        compute the norm of the scaled gradient.
c
         gnorm = zero
         if (fnorm .eq. zero) go to 210
         do 200 j = 1, n
            l = ipvt(j)
            if (wa2(l) .eq. zero) go to 190
            sum = zero
            do 180 i = 1, j
               sum = sum + fjac(i,j)*(qtf(i)/fnorm)
  180          continue
            gnorm = dmax1(gnorm,dabs(sum/wa2(l)))
  190       continue
  200       continue
  210    continue
c
c        test for convergence of the gradient norm.
c
         if (gnorm .le. gtol) info = 4
         if (info .ne. 0) go to 340
c
c        rescale if necessary.
c
         if (mode .eq. 2) go to 230
         do 220 j = 1, n
            diag(j) = dmax1(diag(j),wa2(j))
  220       continue
  230    continue
c
c        beginning of the inner loop.
c
  240    continue
c
c           determine the levenberg-marquardt parameter.
c
            call lmpar(n,fjac,ldfjac,ipvt,diag,qtf,delta,par,wa1,wa2,
     *                 wa3,wa4)
c
c           store the direction p and x + p. calculate the norm of p.
c
            do 250 j = 1, n
               wa1(j) = -wa1(j)
               wa2(j) = x(j) + wa1(j)
               wa3(j) = diag(j)*wa1(j)
  250          continue
            pnorm = enorm(n,wa3)
c
c           on the first iteration, adjust the initial step bound.
c
            if (iter .eq. 1) delta = dmin1(delta,pnorm)
c
c           evaluate the function at x + p and calculate its norm.
c
            iflag = 1
            call fcn(m,n,wa2,wa4,wa3,iflag)
            nfev = nfev + 1
            if (iflag .lt. 0) go to 340
            fnorm1 = enorm(m,wa4)
c
c           compute the scaled actual reduction.
c
            actred = -one
            if (p1*fnorm1 .lt. fnorm) actred = one - (fnorm1/fnorm)**2
c
c           compute the scaled predicted reduction and
c           the scaled directional derivative.
c
            do 270 j = 1, n
               wa3(j) = zero
               l = ipvt(j)
               temp = wa1(l)
               do 260 i = 1, j
                  wa3(i) = wa3(i) + fjac(i,j)*temp
  260             continue
  270          continue
            temp1 = enorm(n,wa3)/fnorm
            temp2 = (dsqrt(par)*pnorm)/fnorm
            prered = temp1**2 + temp2**2/p5
            dirder = -(temp1**2 + temp2**2)
c
c           compute the ratio of the actual to the predicted
c           reduction.
c
            ratio = zero
            if (prered .ne. zero) ratio = actred/prered
c
c           update the step bound.
c
            if (ratio .gt. p25) go to 280
               if (actred .ge. zero) temp = p5
               if (actred .lt. zero)
     *            temp = p5*dirder/(dirder + p5*actred)
               if (p1*fnorm1 .ge. fnorm .or. temp .lt. p1) temp = p1
               delta = temp*dmin1(delta,pnorm/p1)
               par = par/temp
               go to 300
  280       continue
               if (par .ne. zero .and. ratio .lt. p75) go to 290
               delta = pnorm/p5
               par = p5*par
  290          continue
  300       continue
c
c           test for successful iteration.
c
            if (ratio .lt. p0001) go to 330
c
c           successful iteration. update x, fvec, and their norms.
c
            do 310 j = 1, n
               x(j) = wa2(j)
               wa2(j) = diag(j)*x(j)
  310          continue
            do 320 i = 1, m
               fvec(i) = wa4(i)
  320          continue
            xnorm = enorm(n,wa2)
            fnorm = fnorm1
            iter = iter + 1
  330       continue
c
c           tests for convergence.
c
            if (dabs(actred) .le. ftol .and. prered .le. ftol
     *          .and. p5*ratio .le. one) info = 1
            if (delta .le. xtol*xnorm) info = 2
            if (dabs(actred) .le. ftol .and. prered .le. ftol
     *          .and. p5*ratio .le. one .and. info .eq. 2) info = 3
            if (info .ne. 0) go to 340
c
c           tests for termination and stringent tolerances.
c
            if (nfev .ge. maxfev) info = 5
            if (dabs(actred) .le. epsmch .and. prered .le. epsmch
     *          .and. p5*ratio .le. one) info = 6
            if (delta .le. epsmch*xnorm) info = 7
            if (gnorm .le. epsmch) info = 8
            if (info .ne. 0) go to 340
c
c           end of the inner loop. repeat if iteration unsuccessful.
c
            if (ratio .lt. p0001) go to 240
c
c        end of the outer loop.
c
         go to 30
  340 continue
c
c     termination, either normal or user imposed.
c
      if (iflag .lt. 0) info = iflag
      iflag = 0
      if (nprint .gt. 0) call fcn(m,n,x,fvec,wa3,iflag)
      return
c
c     last card of subroutine lmstr.
c
      end
