      double precision function enorm(n,x)
      integer n
      double precision x(n)
c     **********
c
c     function enorm
c
c     given an n-vector x, this function calculates the
c     euclidean norm of x.
c
c     the euclidean norm is computed by accumulating the sum of
c     squares in three different sums. the sums of squares for the
c     small and large components are scaled so that no overflows
c     occur. non-destructive underflows are permitted. underflows
c     and overflows do not occur in the computation of the unscaled
c     sum of squares for the intermediate components.
c     the definitions of small, intermediate and large components
c     depend on two constants, rdwarf and rgiant. the main
c     restrictions on these constants are that rdwarf**2 not
c     underflow and rgiant**2 not overflow. the constants
c     given here are suitable for every known computer.
c
c     the function statement is
c
c       double precision function enorm(n,x)
c
c     where
c
c       n is a positive integer input variable.
c
c       x is an input array of length n.
c
c     subprograms called
c
c       fortran-supplied ... dabs,dsqrt
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********
      integer i
      double precision agiant,floatn,one,rdwarf,rgiant,s1,s2,s3,xabs,
     *                 x1max,x3max,zero
      data one,zero,rdwarf,rgiant /1.0d0,0.0d0,3.834d-20,1.304d19/
      s1 = zero
      s2 = zero
      s3 = zero
      x1max = zero
      x3max = zero
      floatn = n
      agiant = rgiant/floatn
      do 90 i = 1, n
         xabs = dabs(x(i))
         if (xabs .gt. rdwarf .and. xabs .lt. agiant) go to 70
            if (xabs .le. rdwarf) go to 30
c
c              sum for large components.
c
               if (xabs .le. x1max) go to 10
                  s1 = one + s1*(x1max/xabs)**2
                  x1max = xabs
                  go to 20
   10          continue
                  s1 = s1 + (xabs/x1max)**2
   20          continue
               go to 60
   30       continue
c
c              sum for small components.
c
               if (xabs .le. x3max) go to 40
                  s3 = one + s3*(x3max/xabs)**2
                  x3max = xabs
                  go to 50
   40          continue
                  if (xabs .ne. zero) s3 = s3 + (xabs/x3max)**2
   50          continue
   60       continue
            go to 80
   70    continue
c
c           sum for intermediate components.
c
            s2 = s2 + xabs**2
   80    continue
   90    continue
c
c     calculation of norm.
c
      if (s1 .eq. zero) go to 100
         enorm = x1max*dsqrt(s1+(s2/x1max)/x1max)
         go to 130
  100 continue
         if (s2 .eq. zero) go to 110
            if (s2 .ge. x3max)
     *         enorm = dsqrt(s2*(one+(x3max/s2)*(x3max*s3)))
            if (s2 .lt. x3max)
     *         enorm = dsqrt(x3max*((s2/x3max)+(x3max*s3)))
            go to 120
  110    continue
            enorm = x3max*dsqrt(s3)
  120    continue
  130 continue
      return
c
c     last card of function enorm.
c
      end
