      subroutine qrsolv(n,r,ldr,ipvt,diag,qtb,x,sdiag,wa)
      integer n,ldr
      integer ipvt(n)
      double precision r(ldr,n),diag(n),qtb(n),x(n),sdiag(n),wa(n)
c     **********
c
c     subroutine qrsolv
c
c     given an m by n matrix a, an n by n diagonal matrix d,
c     and an m-vector b, the problem is to determine an x which
c     solves the system
c
c           a*x = b ,     d*x = 0 ,
c
c     in the least squares sense.
c
c     this subroutine completes the solution of the problem
c     if it is provided with the necessary information from the
c     qr factorization, with column pivoting, of a. that is, if
c     a*p = q*r, where p is a permutation matrix, q has orthogonal
c     columns, and r is an upper triangular matrix with diagonal
c     elements of nonincreasing magnitude, then qrsolv expects
c     the full upper triangle of r, the permutation matrix p,
c     and the first n components of (q transpose)*b. the system
c     a*x = b, d*x = 0, is then equivalent to
c
c                  t       t
c           r*z = q *b ,  p *d*p*z = 0 ,
c
c     where x = p*z. if this system does not have full rank,
c     then a least squares solution is obtained. on output qrsolv
c     also provides an upper triangular matrix s such that
c
c            t   t               t
c           p *(a *a + d*d)*p = s *s .
c
c     s is computed within qrsolv and may be of separate interest.
c
c     the subroutine statement is
c
c       subroutine qrsolv(n,r,ldr,ipvt,diag,qtb,x,sdiag,wa)
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
c       x is an output array of length n which contains the least
c         squares solution of the system a*x = b, d*x = 0.
c
c       sdiag is an output array of length n which contains the
c         diagonal elements of the upper triangular matrix s.
c
c       wa is a work array of length n.
c
c     subprograms called
c
c       fortran-supplied ... dabs,dsqrt
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********
      integer i,j,jp1,k,kp1,l,nsing
      double precision cos,cotan,p5,p25,qtbpj,sin,sum,tan,temp,zero
      data p5,p25,zero /5.0d-1,2.5d-1,0.0d0/
c
c     copy r and (q transpose)*b to preserve input and initialize s.
c     in particular, save the diagonal elements of r in x.
c
      do 20 j = 1, n
         do 10 i = j, n
            r(i,j) = r(j,i)
   10       continue
         x(j) = r(j,j)
         wa(j) = qtb(j)
   20    continue
c
c     eliminate the diagonal matrix d using a givens rotation.
c
      do 100 j = 1, n
c
c        prepare the row of d to be eliminated, locating the
c        diagonal element using p from the qr factorization.
c
         l = ipvt(j)
         if (diag(l) .eq. zero) go to 90
         do 30 k = j, n
            sdiag(k) = zero
   30       continue
         sdiag(j) = diag(l)
c
c        the transformations to eliminate the row of d
c        modify only a single element of (q transpose)*b
c        beyond the first n, which is initially zero.
c
         qtbpj = zero
         do 80 k = j, n
c
c           determine a givens rotation which eliminates the
c           appropriate element in the current row of d.
c
            if (sdiag(k) .eq. zero) go to 70
            if (dabs(r(k,k)) .ge. dabs(sdiag(k))) go to 40
               cotan = r(k,k)/sdiag(k)
               sin = p5/dsqrt(p25+p25*cotan**2)
               cos = sin*cotan
               go to 50
   40       continue
               tan = sdiag(k)/r(k,k)
               cos = p5/dsqrt(p25+p25*tan**2)
               sin = cos*tan
   50       continue
c
c           compute the modified diagonal element of r and
c           the modified element of ((q transpose)*b,0).
c
            r(k,k) = cos*r(k,k) + sin*sdiag(k)
            temp = cos*wa(k) + sin*qtbpj
            qtbpj = -sin*wa(k) + cos*qtbpj
            wa(k) = temp
c
c           accumulate the tranformation in the row of s.
c
            kp1 = k + 1
            if (n .lt. kp1) go to 70
            do 60 i = kp1, n
               temp = cos*r(i,k) + sin*sdiag(i)
               sdiag(i) = -sin*r(i,k) + cos*sdiag(i)
               r(i,k) = temp
   60          continue
   70       continue
   80       continue
   90    continue
c
c        store the diagonal element of s and restore
c        the corresponding diagonal element of r.
c
         sdiag(j) = r(j,j)
         r(j,j) = x(j)
  100    continue
c
c     solve the triangular system for z. if the system is
c     singular, then obtain a least squares solution.
c
      nsing = n
      do 110 j = 1, n
         if (sdiag(j) .eq. zero .and. nsing .eq. n) nsing = j - 1
         if (nsing .lt. n) wa(j) = zero
  110    continue
      if (nsing .lt. 1) go to 150
      do 140 k = 1, nsing
         j = nsing - k + 1
         sum = zero
         jp1 = j + 1
         if (nsing .lt. jp1) go to 130
         do 120 i = jp1, nsing
            sum = sum + r(i,j)*wa(i)
  120       continue
  130    continue
         wa(j) = (wa(j) - sum)/sdiag(j)
  140    continue
  150 continue
c
c     permute the components of z back to components of x.
c
      do 160 j = 1, n
         l = ipvt(j)
         x(l) = wa(j)
  160    continue
      return
c
c     last card of subroutine qrsolv.
c
      end
