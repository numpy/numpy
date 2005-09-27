      subroutine dogleg(n,r,lr,diag,qtb,delta,x,wa1,wa2)
      integer n,lr
      double precision delta
      double precision r(lr),diag(n),qtb(n),x(n),wa1(n),wa2(n)
c     **********
c
c     subroutine dogleg
c
c     given an m by n matrix a, an n by n nonsingular diagonal
c     matrix d, an m-vector b, and a positive number delta, the
c     problem is to determine the convex combination x of the
c     gauss-newton and scaled gradient directions that minimizes
c     (a*x - b) in the least squares sense, subject to the
c     restriction that the euclidean norm of d*x be at most delta.
c
c     this subroutine completes the solution of the problem
c     if it is provided with the necessary information from the
c     qr factorization of a. that is, if a = q*r, where q has
c     orthogonal columns and r is an upper triangular matrix,
c     then dogleg expects the full upper triangle of r and
c     the first n components of (q transpose)*b.
c
c     the subroutine statement is
c
c       subroutine dogleg(n,r,lr,diag,qtb,delta,x,wa1,wa2)
c
c     where
c
c       n is a positive integer input variable set to the order of r.
c
c       r is an input array of length lr which must contain the upper
c         triangular matrix r stored by rows.
c
c       lr is a positive integer input variable not less than
c         (n*(n+1))/2.
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
c       x is an output array of length n which contains the desired
c         convex combination of the gauss-newton direction and the
c         scaled gradient direction.
c
c       wa1 and wa2 are work arrays of length n.
c
c     subprograms called
c
c       minpack-supplied ... dpmpar,enorm
c
c       fortran-supplied ... dabs,dmax1,dmin1,dsqrt
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********
      integer i,j,jj,jp1,k,l
      double precision alpha,bnorm,epsmch,gnorm,one,qnorm,sgnorm,sum,
     *                 temp,zero
      double precision dpmpar,enorm
      data one,zero /1.0d0,0.0d0/
c
c     epsmch is the machine precision.
c
      epsmch = dpmpar(1)
c
c     first, calculate the gauss-newton direction.
c
      jj = (n*(n + 1))/2 + 1
      do 50 k = 1, n
         j = n - k + 1
         jp1 = j + 1
         jj = jj - k
         l = jj + 1
         sum = zero
         if (n .lt. jp1) go to 20
         do 10 i = jp1, n
            sum = sum + r(l)*x(i)
            l = l + 1
   10       continue
   20    continue
         temp = r(jj)
         if (temp .ne. zero) go to 40
         l = j
         do 30 i = 1, j
            temp = dmax1(temp,dabs(r(l)))
            l = l + n - i
   30       continue
         temp = epsmch*temp
         if (temp .eq. zero) temp = epsmch
   40    continue
         x(j) = (qtb(j) - sum)/temp
   50    continue
c
c     test whether the gauss-newton direction is acceptable.
c
      do 60 j = 1, n
         wa1(j) = zero
         wa2(j) = diag(j)*x(j)
   60    continue
      qnorm = enorm(n,wa2)
      if (qnorm .le. delta) go to 140
c
c     the gauss-newton direction is not acceptable.
c     next, calculate the scaled gradient direction.
c
      l = 1
      do 80 j = 1, n
         temp = qtb(j)
         do 70 i = j, n
            wa1(i) = wa1(i) + r(l)*temp
            l = l + 1
   70       continue
         wa1(j) = wa1(j)/diag(j)
   80    continue
c
c     calculate the norm of the scaled gradient and test for
c     the special case in which the scaled gradient is zero.
c
      gnorm = enorm(n,wa1)
      sgnorm = zero
      alpha = delta/qnorm
      if (gnorm .eq. zero) go to 120
c
c     calculate the point along the scaled gradient
c     at which the quadratic is minimized.
c
      do 90 j = 1, n
         wa1(j) = (wa1(j)/gnorm)/diag(j)
   90    continue
      l = 1
      do 110 j = 1, n
         sum = zero
         do 100 i = j, n
            sum = sum + r(l)*wa1(i)
            l = l + 1
  100       continue
         wa2(j) = sum
  110    continue
      temp = enorm(n,wa2)
      sgnorm = (gnorm/temp)/temp
c
c     test whether the scaled gradient direction is acceptable.
c
      alpha = zero
      if (sgnorm .ge. delta) go to 120
c
c     the scaled gradient direction is not acceptable.
c     finally, calculate the point along the dogleg
c     at which the quadratic is minimized.
c
      bnorm = enorm(n,qtb)
      temp = (bnorm/gnorm)*(bnorm/qnorm)*(sgnorm/delta)
      temp = temp - (delta/qnorm)*(sgnorm/delta)**2
     *       + dsqrt((temp-(delta/qnorm))**2
     *               +(one-(delta/qnorm)**2)*(one-(sgnorm/delta)**2))
      alpha = ((delta/qnorm)*(one - (sgnorm/delta)**2))/temp
  120 continue
c
c     form appropriate convex combination of the gauss-newton
c     direction and the scaled gradient direction.
c
      temp = (one - alpha)*dmin1(sgnorm,delta)
      do 130 j = 1, n
         x(j) = temp*wa1(j) + alpha*x(j)
  130    continue
  140 continue
      return
c
c     last card of subroutine dogleg.
c
      end
