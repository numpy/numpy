      subroutine fdjac2(fcn,m,n,x,fvec,fjac,ldfjac,iflag,epsfcn,wa)
      integer m,n,ldfjac,iflag
      double precision epsfcn
      double precision x(n),fvec(m),fjac(ldfjac,n),wa(m)
c     **********
c
c     subroutine fdjac2
c
c     this subroutine computes a forward-difference approximation
c     to the m by n jacobian matrix associated with a specified
c     problem of m functions in n variables.
c
c     the subroutine statement is
c
c       subroutine fdjac2(fcn,m,n,x,fvec,fjac,ldfjac,iflag,epsfcn,wa)
c
c     where
c
c       fcn is the name of the user-supplied subroutine which
c         calculates the functions. fcn must be declared
c         in an external statement in the user calling
c         program, and should be written as follows.
c
c         subroutine fcn(m,n,x,fvec,iflag)
c         integer m,n,iflag
c         double precision x(n),fvec(m)
c         ----------
c         calculate the functions at x and
c         return this vector in fvec.
c         ----------
c         return
c         end
c
c         the value of iflag should not be changed by fcn unless
c         the user wants to terminate execution of fdjac2.
c         in this case set iflag to a negative integer.
c
c       m is a positive integer input variable set to the number
c         of functions.
c
c       n is a positive integer input variable set to the number
c         of variables. n must not exceed m.
c
c       x is an input array of length n.
c
c       fvec is an input array of length m which must contain the
c         functions evaluated at x.
c
c       fjac is an output m by n array which contains the
c         approximation to the jacobian matrix evaluated at x.
c
c       ldfjac is a positive integer input variable not less than m
c         which specifies the leading dimension of the array fjac.
c
c       iflag is an integer variable which can be used to terminate
c         the execution of fdjac2. see description of fcn.
c
c       epsfcn is an input variable used in determining a suitable
c         step length for the forward-difference approximation. this
c         approximation assumes that the relative errors in the
c         functions are of the order of epsfcn. if epsfcn is less
c         than the machine precision, it is assumed that the relative
c         errors in the functions are of the order of the machine
c         precision.
c
c       wa is a work array of length m.
c
c     subprograms called
c
c       user-supplied ...... fcn
c
c       minpack-supplied ... dpmpar
c
c       fortran-supplied ... dabs,dmax1,dsqrt
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********
      integer i,j
      double precision eps,epsmch,h,temp,zero
      double precision dpmpar
      data zero /0.0d0/
c
c     epsmch is the machine precision.
c
      epsmch = dpmpar(1)
c
      eps = dsqrt(dmax1(epsfcn,epsmch))
      do 20 j = 1, n
         temp = x(j)
         h = eps*dabs(temp)
         if (h .eq. zero) h = eps
         x(j) = temp + h
         call fcn(m,n,x,wa,iflag)
         if (iflag .lt. 0) go to 30
         x(j) = temp
         do 10 i = 1, m
            fjac(i,j) = (wa(i) - fvec(i))/h
   10       continue
   20    continue
   30 continue
      return
c
c     last card of subroutine fdjac2.
c
      end
