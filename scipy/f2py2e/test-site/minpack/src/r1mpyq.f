      subroutine r1mpyq(m,n,a,lda,v,w)
      integer m,n,lda
      double precision a(lda,n),v(n),w(n)
c     **********
c
c     subroutine r1mpyq
c
c     given an m by n matrix a, this subroutine computes a*q where
c     q is the product of 2*(n - 1) transformations
c
c           gv(n-1)*...*gv(1)*gw(1)*...*gw(n-1)
c
c     and gv(i), gw(i) are givens rotations in the (i,n) plane which
c     eliminate elements in the i-th and n-th planes, respectively.
c     q itself is not given, rather the information to recover the
c     gv, gw rotations is supplied.
c
c     the subroutine statement is
c
c       subroutine r1mpyq(m,n,a,lda,v,w)
c
c     where
c
c       m is a positive integer input variable set to the number
c         of rows of a.
c
c       n is a positive integer input variable set to the number
c         of columns of a.
c
c       a is an m by n array. on input a must contain the matrix
c         to be postmultiplied by the orthogonal matrix q
c         described above. on output a*q has replaced a.
c
c       lda is a positive integer input variable not less than m
c         which specifies the leading dimension of the array a.
c
c       v is an input array of length n. v(i) must contain the
c         information necessary to recover the givens rotation gv(i)
c         described above.
c
c       w is an input array of length n. w(i) must contain the
c         information necessary to recover the givens rotation gw(i)
c         described above.
c
c     subroutines called
c
c       fortran-supplied ... dabs,dsqrt
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********
      integer i,j,nmj,nm1
      double precision cos,one,sin,temp
      data one /1.0d0/
c
c     apply the first set of givens rotations to a.
c
      nm1 = n - 1
      if (nm1 .lt. 1) go to 50
      do 20 nmj = 1, nm1
         j = n - nmj
         if (dabs(v(j)) .gt. one) cos = one/v(j)
         if (dabs(v(j)) .gt. one) sin = dsqrt(one-cos**2)
         if (dabs(v(j)) .le. one) sin = v(j)
         if (dabs(v(j)) .le. one) cos = dsqrt(one-sin**2)
         do 10 i = 1, m
            temp = cos*a(i,j) - sin*a(i,n)
            a(i,n) = sin*a(i,j) + cos*a(i,n)
            a(i,j) = temp
   10       continue
   20    continue
c
c     apply the second set of givens rotations to a.
c
      do 40 j = 1, nm1
         if (dabs(w(j)) .gt. one) cos = one/w(j)
         if (dabs(w(j)) .gt. one) sin = dsqrt(one-cos**2)
         if (dabs(w(j)) .le. one) sin = w(j)
         if (dabs(w(j)) .le. one) cos = dsqrt(one-sin**2)
         do 30 i = 1, m
            temp = cos*a(i,j) + sin*a(i,n)
            a(i,n) = -sin*a(i,j) + cos*a(i,n)
            a(i,j) = temp
   30       continue
   40    continue
   50 continue
      return
c
c     last card of subroutine r1mpyq.
c
      end
