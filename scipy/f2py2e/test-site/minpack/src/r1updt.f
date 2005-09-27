      subroutine r1updt(m,n,s,ls,u,v,w,sing)
      integer m,n,ls
      logical sing
      double precision s(ls),u(m),v(n),w(m)
c     **********
c
c     subroutine r1updt
c
c     given an m by n lower trapezoidal matrix s, an m-vector u,
c     and an n-vector v, the problem is to determine an
c     orthogonal matrix q such that
c
c                   t
c           (s + u*v )*q
c
c     is again lower trapezoidal.
c
c     this subroutine determines q as the product of 2*(n - 1)
c     transformations
c
c           gv(n-1)*...*gv(1)*gw(1)*...*gw(n-1)
c
c     where gv(i), gw(i) are givens rotations in the (i,n) plane
c     which eliminate elements in the i-th and n-th planes,
c     respectively. q itself is not accumulated, rather the
c     information to recover the gv, gw rotations is returned.
c
c     the subroutine statement is
c
c       subroutine r1updt(m,n,s,ls,u,v,w,sing)
c
c     where
c
c       m is a positive integer input variable set to the number
c         of rows of s.
c
c       n is a positive integer input variable set to the number
c         of columns of s. n must not exceed m.
c
c       s is an array of length ls. on input s must contain the lower
c         trapezoidal matrix s stored by columns. on output s contains
c         the lower trapezoidal matrix produced as described above.
c
c       ls is a positive integer input variable not less than
c         (n*(2*m-n+1))/2.
c
c       u is an input array of length m which must contain the
c         vector u.
c
c       v is an array of length n. on input v must contain the vector
c         v. on output v(i) contains the information necessary to
c         recover the givens rotation gv(i) described above.
c
c       w is an output array of length m. w(i) contains information
c         necessary to recover the givens rotation gw(i) described
c         above.
c
c       sing is a logical output variable. sing is set true if any
c         of the diagonal elements of the output s are zero. otherwise
c         sing is set false.
c
c     subprograms called
c
c       minpack-supplied ... dpmpar
c
c       fortran-supplied ... dabs,dsqrt
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more,
c     john l. nazareth
c
c     **********
      integer i,j,jj,l,nmj,nm1
      double precision cos,cotan,giant,one,p5,p25,sin,tan,tau,temp,
     *                 zero
      double precision dpmpar
      data one,p5,p25,zero /1.0d0,5.0d-1,2.5d-1,0.0d0/
c
c     giant is the largest magnitude.
c
      giant = dpmpar(3)
c
c     initialize the diagonal element pointer.
c
      jj = (n*(2*m - n + 1))/2 - (m - n)
c
c     move the nontrivial part of the last column of s into w.
c
      l = jj
      do 10 i = n, m
         w(i) = s(l)
         l = l + 1
   10    continue
c
c     rotate the vector v into a multiple of the n-th unit vector
c     in such a way that a spike is introduced into w.
c
      nm1 = n - 1
      if (nm1 .lt. 1) go to 70
      do 60 nmj = 1, nm1
         j = n - nmj
         jj = jj - (m - j + 1)
         w(j) = zero
         if (v(j) .eq. zero) go to 50
c
c        determine a givens rotation which eliminates the
c        j-th element of v.
c
         if (dabs(v(n)) .ge. dabs(v(j))) go to 20
            cotan = v(n)/v(j)
            sin = p5/dsqrt(p25+p25*cotan**2)
            cos = sin*cotan
            tau = one
            if (dabs(cos)*giant .gt. one) tau = one/cos
            go to 30
   20    continue
            tan = v(j)/v(n)
            cos = p5/dsqrt(p25+p25*tan**2)
            sin = cos*tan
            tau = sin
   30    continue
c
c        apply the transformation to v and store the information
c        necessary to recover the givens rotation.
c
         v(n) = sin*v(j) + cos*v(n)
         v(j) = tau
c
c        apply the transformation to s and extend the spike in w.
c
         l = jj
         do 40 i = j, m
            temp = cos*s(l) - sin*w(i)
            w(i) = sin*s(l) + cos*w(i)
            s(l) = temp
            l = l + 1
   40       continue
   50    continue
   60    continue
   70 continue
c
c     add the spike from the rank 1 update to w.
c
      do 80 i = 1, m
         w(i) = w(i) + v(n)*u(i)
   80    continue
c
c     eliminate the spike.
c
      sing = .false.
      if (nm1 .lt. 1) go to 140
      do 130 j = 1, nm1
         if (w(j) .eq. zero) go to 120
c
c        determine a givens rotation which eliminates the
c        j-th element of the spike.
c
         if (dabs(s(jj)) .ge. dabs(w(j))) go to 90
            cotan = s(jj)/w(j)
            sin = p5/dsqrt(p25+p25*cotan**2)
            cos = sin*cotan
            tau = one
            if (dabs(cos)*giant .gt. one) tau = one/cos
            go to 100
   90    continue
            tan = w(j)/s(jj)
            cos = p5/dsqrt(p25+p25*tan**2)
            sin = cos*tan
            tau = sin
  100    continue
c
c        apply the transformation to s and reduce the spike in w.
c
         l = jj
         do 110 i = j, m
            temp = cos*s(l) + sin*w(i)
            w(i) = -sin*s(l) + cos*w(i)
            s(l) = temp
            l = l + 1
  110       continue
c
c        store the information necessary to recover the
c        givens rotation.
c
         w(j) = tau
  120    continue
c
c        test for zero diagonal elements in the output s.
c
         if (s(jj) .eq. zero) sing = .true.
         jj = jj + (m - j + 1)
  130    continue
  140 continue
c
c     move w back into the last column of the output s.
c
      l = jj
      do 150 i = n, m
         s(l) = w(i)
         l = l + 1
  150    continue
      if (s(jj) .eq. zero) sing = .true.
      return
c
c     last card of subroutine r1updt.
c
      end
