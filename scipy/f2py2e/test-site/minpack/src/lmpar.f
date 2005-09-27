      subroutine lmpar(n,r,ldr,ipvt,diag,qtb,delta,par,x,sdiag,wa1,
     *                 wa2)
      integer n,ldr
      integer ipvt(n)
      double precision delta,par
      double precision r(ldr,n),diag(n),qtb(n),x(n),sdiag(n),wa1(n),
     *                 wa2(n)
c     **********
c
c     subroutine lmpar
c
c     given an m by n matrix a, an n by n nonsingular diagonal
c     matrix d, an m-vector b, and a positive number delta,
c     the problem is to determine a value for the parameter
c     par such that if x solves the system
c
c           a*x = b ,     sqrt(par)*d*x = 0 ,
c
c     in the least squares sense, and dxnorm is the euclidean
c     norm of d*x, then either par is zero and
c
c           (dxnorm-delta) .le. 0.1*delta ,
c
c     or par is positive and
c
c           abs(dxnorm-delta) .le. 0.1*delta .
c
c     this subroutine completes the solution of the problem
c     if it is provided with the necessary information from the
c     qr factorization, with column pivoting, of a. that is, if
c     a*p = q*r, where p is a permutation matrix, q has orthogonal
c     columns, and r is an upper triangular matrix with diagonal
c     elements of nonincreasing magnitude, then lmpar expects
c     the full upper triangle of r, the permutation matrix p,
c     and the first n components of (q transpose)*b. on output
c     lmpar also provides an upper triangular matrix s such that
c
c            t   t                   t
c           p *(a *a + par*d*d)*p = s *s .
c
c     s is employed within lmpar and may be of separate interest.
c
c     only a few iterations are generally needed for convergence
c     of the algorithm. if, however, the limit of 10 iterations
c     is reached, then the output par will contain the best
c     value obtained so far.
c
c     the subroutine statement is
c
c       subroutine lmpar(n,r,ldr,ipvt,diag,qtb,delta,par,x,sdiag,
c                        wa1,wa2)
c
c     where
c
c       n is a positive integer input variable set to the order of r.
c
c       r is an n by n array. on input the full upper triangle
c         must contain the full upper triangle of the matrix r.
c         on output the full upper triangle is unaltered, and the
c         strict lower triangle contains the strict upper triangle
c         (transposed) of the upper triangular matrix s.
c
c       ldr is a positive integer input variable not less than n
c         which specifies the leading dimension of the array r.
c
c       ipvt is an integer input array of length n which defines the
c         permutation matrix p such that a*p = q*r. column j of p
c         is column ipvt(j) of the identity matrix.
c
c       diag is an input array of length n which must contain the
c         diagonal elements of the matrix d.
c
c       qtb is an input array of length n which must contain the first
c         n elements of the vector (q transpose)*b.
c
c       delta is a positive input variable which specifies an upper
c         bound on the euclidean norm of d*x.
c
c       par is a nonnegative variable. on input par contains an
c         initial estimate of the levenberg-marquardt parameter.
c         on output par contains the final estimate.
c
c       x is an output array of length n which contains the least
c         squares solution of the system a*x = b, sqrt(par)*d*x = 0,
c         for the output par.
c
c       sdiag is an output array of length n which contains the
c         diagonal elements of the upper triangular matrix s.
c
c       wa1 and wa2 are work arrays of length n.
c
c     subprograms called
c
c       minpack-supplied ... dpmpar,enorm,qrsolv
c
c       fortran-supplied ... dabs,dmax1,dmin1,dsqrt
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********
      integer i,iter,j,jm1,jp1,k,l,nsing
      double precision dxnorm,dwarf,fp,gnorm,parc,parl,paru,p1,p001,
     *                 sum,temp,zero
      double precision dpmpar,enorm
      data p1,p001,zero /1.0d-1,1.0d-3,0.0d0/
c
c     dwarf is the smallest positive magnitude.
c
      dwarf = dpmpar(2)
c
c     compute and store in x the gauss-newton direction. if the
c     jacobian is rank-deficient, obtain a least squares solution.
c
      nsing = n
      do 10 j = 1, n
         wa1(j) = qtb(j)
         if (r(j,j) .eq. zero .and. nsing .eq. n) nsing = j - 1
         if (nsing .lt. n) wa1(j) = zero
   10    continue
      if (nsing .lt. 1) go to 50
      do 40 k = 1, nsing
         j = nsing - k + 1
         wa1(j) = wa1(j)/r(j,j)
         temp = wa1(j)
         jm1 = j - 1
         if (jm1 .lt. 1) go to 30
         do 20 i = 1, jm1
            wa1(i) = wa1(i) - r(i,j)*temp
   20       continue
   30    continue
   40    continue
   50 continue
      do 60 j = 1, n
         l = ipvt(j)
         x(l) = wa1(j)
   60    continue
c
c     initialize the iteration counter.
c     evaluate the function at the origin, and test
c     for acceptance of the gauss-newton direction.
c
      iter = 0
      do 70 j = 1, n
         wa2(j) = diag(j)*x(j)
   70    continue
      dxnorm = enorm(n,wa2)
      fp = dxnorm - delta
      if (fp .le. p1*delta) go to 220
c
c     if the jacobian is not rank deficient, the newton
c     step provides a lower bound, parl, for the zero of
c     the function. otherwise set this bound to zero.
c
      parl = zero
      if (nsing .lt. n) go to 120
      do 80 j = 1, n
         l = ipvt(j)
         wa1(j) = diag(l)*(wa2(l)/dxnorm)
   80    continue
      do 110 j = 1, n
         sum = zero
         jm1 = j - 1
         if (jm1 .lt. 1) go to 100
         do 90 i = 1, jm1
            sum = sum + r(i,j)*wa1(i)
   90       continue
  100    continue
         wa1(j) = (wa1(j) - sum)/r(j,j)
  110    continue
      temp = enorm(n,wa1)
      parl = ((fp/delta)/temp)/temp
  120 continue
c
c     calculate an upper bound, paru, for the zero of the function.
c
      do 140 j = 1, n
         sum = zero
         do 130 i = 1, j
            sum = sum + r(i,j)*qtb(i)
  130       continue
         l = ipvt(j)
         wa1(j) = sum/diag(l)
  140    continue
      gnorm = enorm(n,wa1)
      paru = gnorm/delta
      if (paru .eq. zero) paru = dwarf/dmin1(delta,p1)
c
c     if the input par lies outside of the interval (parl,paru),
c     set par to the closer endpoint.
c
      par = dmax1(par,parl)
      par = dmin1(par,paru)
      if (par .eq. zero) par = gnorm/dxnorm
c
c     beginning of an iteration.
c
  150 continue
         iter = iter + 1
c
c        evaluate the function at the current value of par.
c
         if (par .eq. zero) par = dmax1(dwarf,p001*paru)
         temp = dsqrt(par)
         do 160 j = 1, n
            wa1(j) = temp*diag(j)
  160       continue
         call qrsolv(n,r,ldr,ipvt,wa1,qtb,x,sdiag,wa2)
         do 170 j = 1, n
            wa2(j) = diag(j)*x(j)
  170       continue
         dxnorm = enorm(n,wa2)
         temp = fp
         fp = dxnorm - delta
c
c        if the function is small enough, accept the current value
c        of par. also test for the exceptional cases where parl
c        is zero or the number of iterations has reached 10.
c
         if (dabs(fp) .le. p1*delta
     *       .or. parl .eq. zero .and. fp .le. temp
     *            .and. temp .lt. zero .or. iter .eq. 10) go to 220
c
c        compute the newton correction.
c
         do 180 j = 1, n
            l = ipvt(j)
            wa1(j) = diag(l)*(wa2(l)/dxnorm)
  180       continue
         do 210 j = 1, n
            wa1(j) = wa1(j)/sdiag(j)
            temp = wa1(j)
            jp1 = j + 1
            if (n .lt. jp1) go to 200
            do 190 i = jp1, n
               wa1(i) = wa1(i) - r(i,j)*temp
  190          continue
  200       continue
  210       continue
         temp = enorm(n,wa1)
         parc = ((fp/delta)/temp)/temp
c
c        depending on the sign of the function, update parl or paru.
c
         if (fp .gt. zero) parl = dmax1(parl,par)
         if (fp .lt. zero) paru = dmin1(paru,par)
c
c        compute an improved estimate for par.
c
         par = dmax1(parl,par+parc)
c
c        end of an iteration.
c
         go to 150
  220 continue
c
c     termination.
c
      if (iter .eq. 0) par = zero
      return
c
c     last card of subroutine lmpar.
c
      end
