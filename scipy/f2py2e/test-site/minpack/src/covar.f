      subroutine covar(n,r,ldr,ipvt,tol,wa)
      integer n,ldr
      integer ipvt(n)
      double precision tol
      double precision r(ldr,n),wa(n)
c     **********
c
c     subroutine covar
c
c     given an m by n matrix a, the problem is to determine
c     the covariance matrix corresponding to a, defined as
c
c                    t
c           inverse(a *a) .
c
c     this subroutine completes the solution of the problem
c     if it is provided with the necessary information from the
c     qr factorization, with column pivoting, of a. that is, if
c     a*p = q*r, where p is a permutation matrix, q has orthogonal
c     columns, and r is an upper triangular matrix with diagonal
c     elements of nonincreasing magnitude, then covar expects
c     the full upper triangle of r and the permutation matrix p.
c     the covariance matrix is then computed as
c
c                      t     t
c           p*inverse(r *r)*p  .
c
c     if a is nearly rank deficient, it may be desirable to compute
c     the covariance matrix corresponding to the linearly independent
c     columns of a. to define the numerical rank of a, covar uses
c     the tolerance tol. if l is the largest integer such that
c
c           abs(r(l,l)) .gt. tol*abs(r(1,1)) ,
c
c     then covar computes the covariance matrix corresponding to
c     the first l columns of r. for k greater than l, column
c     and row ipvt(k) of the covariance matrix are set to zero.
c
c     the subroutine statement is
c
c       subroutine covar(n,r,ldr,ipvt,tol,wa)
c
c     where
c
c       n is a positive integer input variable set to the order of r.
c
c       r is an n by n array. on input the full upper triangle must
c         contain the full upper triangle of the matrix r. on output
c         r contains the square symmetric covariance matrix.
c
c       ldr is a positive integer input variable not less than n
c         which specifies the leading dimension of the array r.
c
c       ipvt is an integer input array of length n which defines the
c         permutation matrix p such that a*p = q*r. column j of p
c         is column ipvt(j) of the identity matrix.
c
c       tol is a nonnegative input variable used to define the
c         numerical rank of a in the manner described above.
c
c       wa is a work array of length n.
c
c     subprograms called
c
c       fortran-supplied ... dabs
c
c     argonne national laboratory. minpack project. august 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********
      integer i,ii,j,jj,k,km1,l
      logical sing
      double precision one,temp,tolr,zero
      data one,zero /1.0d0,0.0d0/
c
c     form the inverse of r in the full upper triangle of r.
c
      tolr = tol*dabs(r(1,1))
      l = 0
      do 40 k = 1, n
         if (dabs(r(k,k)) .le. tolr) go to 50
         r(k,k) = one/r(k,k)
         km1 = k - 1
         if (km1 .lt. 1) go to 30
         do 20 j = 1, km1
            temp = r(k,k)*r(j,k)
            r(j,k) = zero
            do 10 i = 1, j
               r(i,k) = r(i,k) - temp*r(i,j)
   10          continue
   20       continue
   30    continue
         l = k
   40    continue
   50 continue
c
c     form the full upper triangle of the inverse of (r transpose)*r
c     in the full upper triangle of r.
c
      if (l .lt. 1) go to 110
      do 100 k = 1, l
         km1 = k - 1
         if (km1 .lt. 1) go to 80
         do 70 j = 1, km1
            temp = r(j,k)
            do 60 i = 1, j
               r(i,j) = r(i,j) + temp*r(i,k)
   60          continue
   70       continue
   80    continue
         temp = r(k,k)
         do 90 i = 1, k
            r(i,k) = temp*r(i,k)
   90       continue
  100    continue
  110 continue
c
c     form the full lower triangle of the covariance matrix
c     in the strict lower triangle of r and in wa.
c
      do 130 j = 1, n
         jj = ipvt(j)
         sing = j .gt. l
         do 120 i = 1, j
            if (sing) r(i,j) = zero
            ii = ipvt(i)
            if (ii .gt. jj) r(ii,jj) = r(i,j)
            if (ii .lt. jj) r(jj,ii) = r(i,j)
  120       continue
         wa(jj) = r(j,j)
  130    continue
c
c     symmetrize the covariance matrix in r.
c
      do 150 j = 1, n
         do 140 i = 1, j
            r(i,j) = r(j,i)
  140       continue
         r(j,j) = wa(j)
  150    continue
      return
c
c     last card of subroutine covar.
c
      end
