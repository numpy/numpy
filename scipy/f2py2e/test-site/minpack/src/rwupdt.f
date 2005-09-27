      subroutine rwupdt(n,r,ldr,w,b,alpha,cos,sin)
      integer n,ldr
      double precision alpha
      double precision r(ldr,n),w(n),b(n),cos(n),sin(n)
c     **********
c
c     subroutine rwupdt
c
c     given an n by n upper triangular matrix r, this subroutine
c     computes the qr decomposition of the matrix formed when a row
c     is added to r. if the row is specified by the vector w, then
c     rwupdt determines an orthogonal matrix q such that when the
c     n+1 by n matrix composed of r augmented by w is premultiplied
c     by (q transpose), the resulting matrix is upper trapezoidal.
c     the matrix (q transpose) is the product of n transformations
c
c           g(n)*g(n-1)* ... *g(1)
c
c     where g(i) is a givens rotation in the (i,n+1) plane which
c     eliminates elements in the (n+1)-st plane. rwupdt also
c     computes the product (q transpose)*c where c is the
c     (n+1)-vector (b,alpha). q itself is not accumulated, rather
c     the information to recover the g rotations is supplied.
c
c     the subroutine statement is
c
c       subroutine rwupdt(n,r,ldr,w,b,alpha,cos,sin)
c
c     where
c
c       n is a positive integer input variable set to the order of r.
c
c       r is an n by n array. on input the upper triangular part of
c         r must contain the matrix to be updated. on output r
c         contains the updated triangular matrix.
c
c       ldr is a positive integer input variable not less than n
c         which specifies the leading dimension of the array r.
c
c       w is an input array of length n which must contain the row
c         vector to be added to r.
c
c       b is an array of length n. on input b must contain the
c         first n elements of the vector c. on output b contains
c         the first n elements of the vector (q transpose)*c.
c
c       alpha is a variable. on input alpha must contain the
c         (n+1)-st element of the vector c. on output alpha contains
c         the (n+1)-st element of the vector (q transpose)*c.
c
c       cos is an output array of length n which contains the
c         cosines of the transforming givens rotations.
c
c       sin is an output array of length n which contains the
c         sines of the transforming givens rotations.
c
c     subprograms called
c
c       fortran-supplied ... dabs,dsqrt
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, dudley v. goetschel, kenneth e. hillstrom,
c     jorge j. more
c
c     **********
      integer i,j,jm1
      double precision cotan,one,p5,p25,rowj,tan,temp,zero
      data one,p5,p25,zero /1.0d0,5.0d-1,2.5d-1,0.0d0/
c
      do 60 j = 1, n
         rowj = w(j)
         jm1 = j - 1
c
c        apply the previous transformations to
c        r(i,j), i=1,2,...,j-1, and to w(j).
c
         if (jm1 .lt. 1) go to 20
         do 10 i = 1, jm1
            temp = cos(i)*r(i,j) + sin(i)*rowj
            rowj = -sin(i)*r(i,j) + cos(i)*rowj
            r(i,j) = temp
   10       continue
   20    continue
c
c        determine a givens rotation which eliminates w(j).
c
         cos(j) = one
         sin(j) = zero
         if (rowj .eq. zero) go to 50
         if (dabs(r(j,j)) .ge. dabs(rowj)) go to 30
            cotan = r(j,j)/rowj
            sin(j) = p5/dsqrt(p25+p25*cotan**2)
            cos(j) = sin(j)*cotan
            go to 40
   30    continue
            tan = rowj/r(j,j)
            cos(j) = p5/dsqrt(p25+p25*tan**2)
            sin(j) = cos(j)*tan
   40    continue
c
c        apply the current transformation to r(j,j), b(j), and alpha.
c
         r(j,j) = cos(j)*r(j,j) + sin(j)*rowj
         temp = cos(j)*b(j) + sin(j)*alpha
         alpha = -sin(j)*b(j) + cos(j)*alpha
         b(j) = temp
   50    continue
   60    continue
      return
c
c     last card of subroutine rwupdt.
c
      end
