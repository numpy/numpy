      subroutine vecfcn(n,x,fvec,nprob)
      integer n,nprob
      double precision x(n),fvec(n)
c     **********
c
c     subroutine vecfcn
c
c     this subroutine defines fourteen test functions. the first
c     five test functions are of dimensions 2,4,2,4,3, respectively,
c     while the remaining test functions are of variable dimension
c     n for any n greater than or equal to 1 (problem 6 is an
c     exception to this, since it does not allow n = 1).
c
c     the subroutine statement is
c
c       subroutine vecfcn(n,x,fvec,nprob)
c
c     where
c
c       n is a positive integer input variable.
c
c       x is an input array of length n.
c
c       fvec is an output array of length n which contains the nprob
c         function vector evaluated at x.
c
c       nprob is a positive integer input variable which defines the
c         number of the problem. nprob must not exceed 14.
c
c     subprograms called
c
c       fortran-supplied ... datan,dcos,dexp,dsign,dsin,dsqrt,
c                            max0,min0
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********
      integer i,iev,ivar,j,k,k1,k2,kp1,ml,mu
      double precision c1,c2,c3,c4,c5,c6,c7,c8,c9,eight,five,h,one,
     *                 prod,sum,sum1,sum2,temp,temp1,temp2,ten,three,
     *                 ti,tj,tk,tpi,two,zero
      double precision dfloat
      data zero,one,two,three,five,eight,ten
     *     /0.0d0,1.0d0,2.0d0,3.0d0,5.0d0,8.0d0,1.0d1/
      data c1,c2,c3,c4,c5,c6,c7,c8,c9
     *     /1.0d4,1.0001d0,2.0d2,2.02d1,1.98d1,1.8d2,2.5d-1,5.0d-1,
     *      2.9d1/
      dfloat(ivar) = ivar
c
c     problem selector.
c
      go to (10,20,30,40,50,60,120,170,200,220,270,300,330,350), nprob
c
c     rosenbrock function.
c
   10 continue
      fvec(1) = one - x(1)
      fvec(2) = ten*(x(2) - x(1)**2)
      go to 380
c
c     powell singular function.
c
   20 continue
      fvec(1) = x(1) + ten*x(2)
      fvec(2) = dsqrt(five)*(x(3) - x(4))
      fvec(3) = (x(2) - two*x(3))**2
      fvec(4) = dsqrt(ten)*(x(1) - x(4))**2
      go to 380
c
c     powell badly scaled function.
c
   30 continue
      fvec(1) = c1*x(1)*x(2) - one
      fvec(2) = dexp(-x(1)) + dexp(-x(2)) - c2
      go to 380
c
c     wood function.
c
   40 continue
      temp1 = x(2) - x(1)**2
      temp2 = x(4) - x(3)**2
      fvec(1) = -c3*x(1)*temp1 - (one - x(1))
      fvec(2) = c3*temp1 + c4*(x(2) - one) + c5*(x(4) - one)
      fvec(3) = -c6*x(3)*temp2 - (one - x(3))
      fvec(4) = c6*temp2 + c4*(x(4) - one) + c5*(x(2) - one)
      go to 380
c
c     helical valley function.
c
   50 continue
      tpi = eight*datan(one)
      temp1 = dsign(c7,x(2))
      if (x(1) .gt. zero) temp1 = datan(x(2)/x(1))/tpi
      if (x(1) .lt. zero) temp1 = datan(x(2)/x(1))/tpi + c8
      temp2 = dsqrt(x(1)**2+x(2)**2)
      fvec(1) = ten*(x(3) - ten*temp1)
      fvec(2) = ten*(temp2 - one)
      fvec(3) = x(3)
      go to 380
c
c     watson function.
c
   60 continue
      do 70 k = 1, n
         fvec(k) = zero
   70    continue
      do 110 i = 1, 29
         ti = dfloat(i)/c9
         sum1 = zero
         temp = one
         do 80 j = 2, n
            sum1 = sum1 + dfloat(j-1)*temp*x(j)
            temp = ti*temp
   80       continue
         sum2 = zero
         temp = one
         do 90 j = 1, n
            sum2 = sum2 + temp*x(j)
            temp = ti*temp
   90       continue
         temp1 = sum1 - sum2**2 - one
         temp2 = two*ti*sum2
         temp = one/ti
         do 100 k = 1, n
            fvec(k) = fvec(k) + temp*(dfloat(k-1) - temp2)*temp1
            temp = ti*temp
  100       continue
  110    continue
      temp = x(2) - x(1)**2 - one
      fvec(1) = fvec(1) + x(1)*(one - two*temp)
      fvec(2) = fvec(2) + temp
      go to 380
c
c     chebyquad function.
c
  120 continue
      do 130 k = 1, n
         fvec(k) = zero
  130    continue
      do 150 j = 1, n
         temp1 = one
         temp2 = two*x(j) - one
         temp = two*temp2
         do 140 i = 1, n
            fvec(i) = fvec(i) + temp2
            ti = temp*temp2 - temp1
            temp1 = temp2
            temp2 = ti
  140       continue
  150    continue
      tk = one/dfloat(n)
      iev = -1
      do 160 k = 1, n
         fvec(k) = tk*fvec(k)
         if (iev .gt. 0) fvec(k) = fvec(k) + one/(dfloat(k)**2 - one)
         iev = -iev
  160    continue
      go to 380
c
c     brown almost-linear function.
c
  170 continue
      sum = -dfloat(n+1)
      prod = one
      do 180 j = 1, n
         sum = sum + x(j)
         prod = x(j)*prod
  180    continue
      do 190 k = 1, n
         fvec(k) = x(k) + sum
  190    continue
      fvec(n) = prod - one
      go to 380
c
c     discrete boundary value function.
c
  200 continue
      h = one/dfloat(n+1)
      do 210 k = 1, n
         temp = (x(k) + dfloat(k)*h + one)**3
         temp1 = zero
         if (k .ne. 1) temp1 = x(k-1)
         temp2 = zero
         if (k .ne. n) temp2 = x(k+1)
         fvec(k) = two*x(k) - temp1 - temp2 + temp*h**2/two
  210    continue
      go to 380
c
c     discrete integral equation function.
c
  220 continue
      h = one/dfloat(n+1)
      do 260 k = 1, n
         tk = dfloat(k)*h
         sum1 = zero
         do 230 j = 1, k
            tj = dfloat(j)*h
            temp = (x(j) + tj + one)**3
            sum1 = sum1 + tj*temp
  230       continue
         sum2 = zero
         kp1 = k + 1
         if (n .lt. kp1) go to 250
         do 240 j = kp1, n
            tj = dfloat(j)*h
            temp = (x(j) + tj + one)**3
            sum2 = sum2 + (one - tj)*temp
  240       continue
  250    continue
         fvec(k) = x(k) + h*((one - tk)*sum1 + tk*sum2)/two
  260    continue
      go to 380
c
c     trigonometric function.
c
  270 continue
      sum = zero
      do 280 j = 1, n
         fvec(j) = dcos(x(j))
         sum = sum + fvec(j)
  280    continue
      do 290 k = 1, n
         fvec(k) = dfloat(n+k) - dsin(x(k)) - sum - dfloat(k)*fvec(k)
  290    continue
      go to 380
c
c     variably dimensioned function.
c
  300 continue
      sum = zero
      do 310 j = 1, n
         sum = sum + dfloat(j)*(x(j) - one)
  310    continue
      temp = sum*(one + two*sum**2)
      do 320 k = 1, n
         fvec(k) = x(k) - one + dfloat(k)*temp
  320    continue
      go to 380
c
c     broyden tridiagonal function.
c
  330 continue
      do 340 k = 1, n
         temp = (three - two*x(k))*x(k)
         temp1 = zero
         if (k .ne. 1) temp1 = x(k-1)
         temp2 = zero
         if (k .ne. n) temp2 = x(k+1)
         fvec(k) = temp - temp1 - two*temp2 + one
  340    continue
      go to 380
c
c     broyden banded function.
c
  350 continue
      ml = 5
      mu = 1
      do 370 k = 1, n
         k1 = max0(1,k-ml)
         k2 = min0(k+mu,n)
         temp = zero
         do 360 j = k1, k2
            if (j .ne. k) temp = temp + x(j)*(one + x(j))
  360       continue
         fvec(k) = x(k)*(two + five*x(k)**2) + one - temp
  370    continue
  380 continue
      return
c
c     last card of subroutine vecfcn.
c
      end
