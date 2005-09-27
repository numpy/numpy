      subroutine fdjac1(fcn,n,x,fvec,fjac,ldfjac,iflag,ml,mu,epsfcn,
     *                  wa1,wa2)
      integer n,ldfjac,iflag,ml,mu
      double precision epsfcn
      double precision x(n),fvec(n),fjac(ldfjac,n),wa1(n),wa2(n)
c     **********
c
c     subroutine fdjac1
c
c     this subroutine computes a forward-difference approximation
c     to the n by n jacobian matrix associated with a specified
c     problem of n functions in n variables. if the jacobian has
c     a banded form, then function evaluations are saved by only
c     approximating the nonzero terms.
c
c     the subroutine statement is
c
c       subroutine fdjac1(fcn,n,x,fvec,fjac,ldfjac,iflag,ml,mu,epsfcn,
c                         wa1,wa2)
c
c     where
c
c       fcn is the name of the user-supplied subroutine which
c         calculates the functions. fcn must be declared
c         in an external statement in the user calling
c         program, and should be written as follows.
c
c         subroutine fcn(n,x,fvec,iflag)
c         integer n,iflag
c         double precision x(n),fvec(n)
c         ----------
c         calculate the functions at x and
c         return this vector in fvec.
c         ----------
c         return
c         end
c
c         the value of iflag should not be changed by fcn unless
c         the user wants to terminate execution of fdjac1.
c         in this case set iflag to a negative integer.
c
c       n is a positive integer input variable set to the number
c         of functions and variables.
c
c       x is an input array of length n.
c
c       fvec is an input array of length n which must contain the
c         functions evaluated at x.
c
c       fjac is an output n by n array which contains the
c         approximation to the jacobian matrix evaluated at x.
c
c       ldfjac is a positive integer input variable not less than n
c         which specifies the leading dimension of the array fjac.
c
c       iflag is an integer variable which can be used to terminate
c         the execution of fdjac1. see description of fcn.
c
c       ml is a nonnegative integer input variable which specifies
c         the number of subdiagonals within the band of the
c         jacobian matrix. if the jacobian is not banded, set
c         ml to at least n - 1.
c
c       epsfcn is an input variable used in determining a suitable
c         step length for the forward-difference approximation. this
c         approximation assumes that the relative errors in the
c         functions are of the order of epsfcn. if epsfcn is less
c         than the machine precision, it is assumed that the relative
c         errors in the functions are of the order of the machine
c         precision.
c
c       mu is a nonnegative integer input variable which specifies
c         the number of superdiagonals within the band of the
c         jacobian matrix. if the jacobian is not banded, set
c         mu to at least n - 1.
c
c       wa1 and wa2 are work arrays of length n. if ml + mu + 1 is at
c         least n, then the jacobian is considered dense, and wa2 is
c         not referenced.
c
c     subprograms called
c
c       minpack-supplied ... dpmpar
c
c       fortran-supplied ... dabs,dmax1,dsqrt
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********
      integer i,j,k,msum
      double precision eps,epsmch,h,temp,zero
      double precision dpmpar
      data zero /0.0d0/
c
c     epsmch is the machine precision.
c
      epsmch = dpmpar(1)
c
      eps = dsqrt(dmax1(epsfcn,epsmch))
      msum = ml + mu + 1
      if (msum .lt. n) go to 40
c
c        computation of dense approximate jacobian.
c
         do 20 j = 1, n
            temp = x(j)
            h = eps*dabs(temp)
            if (h .eq. zero) h = eps
            x(j) = temp + h
            call fcn(n,x,wa1,iflag)
            if (iflag .lt. 0) go to 30
            x(j) = temp
            do 10 i = 1, n
               fjac(i,j) = (wa1(i) - fvec(i))/h
   10          continue
   20       continue
   30    continue
         go to 110
   40 continue
c
c        computation of banded approximate jacobian.
c
         do 90 k = 1, msum
            do 60 j = k, n, msum
               wa2(j) = x(j)
               h = eps*dabs(wa2(j))
               if (h .eq. zero) h = eps
               x(j) = wa2(j) + h
   60          continue
            call fcn(n,x,wa1,iflag)
            if (iflag .lt. 0) go to 100
            do 80 j = k, n, msum
               x(j) = wa2(j)
               h = eps*dabs(wa2(j))
               if (h .eq. zero) h = eps
               do 70 i = 1, n
                  fjac(i,j) = zero
                  if (i .ge. j - mu .and. i .le. j + ml)
     *               fjac(i,j) = (wa1(i) - fvec(i))/h
   70             continue
   80          continue
   90       continue
  100    continue
  110 continue
      return
c
c     last card of subroutine fdjac1.
c
      end

