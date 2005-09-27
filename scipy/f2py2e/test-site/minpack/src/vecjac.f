      subroutine vecjac(n,x,fjac,ldfjac,nprob)
      integer n,ldfjac,nprob
      double precision x(n),fjac(ldfjac,n)
c     **********
c
c     subroutine vecjac
c
c     this subroutine defines the jacobian matrices of fourteen
c     test functions. the problem dimensions are as described
c     in the prologue comments of vecfcn.
c
c     the subroutine statement is
c
c       subroutine vecjac(n,x,fjac,ldfjac,nprob)
c
c     where
c
c       n is a positive integer variable.
c
c       x is an array of length n.
c
c       fjac is an n by n array. on output fjac contains the
c         jacobian matrix of the nprob function evaluated at x.
c
c       ldfjac is a positive integer variable not less than n
c         which specifies the leading dimension of the array fjac.
c
c       nprob is a positive integer variable which defines the
c         number of the problem. nprob must not exceed 14.
c
c     subprograms called
c
c       fortran-supplied ... datan,dcos,dexp,dmin1,dsin,dsqrt,
c                            max0,min0
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********
      integer i,ivar,j,k,k1,k2,ml,mu
      double precision c1,c3,c4,c5,c6,c9,eight,fiftn,five,four,h,
     *                 hundrd,one,prod,six,sum,sum1,sum2,temp,temp1,
     *                 temp2,temp3,temp4,ten,three,ti,tj,tk,tpi,
     *                 twenty,two,zero
      double precision dfloat
      data zero,one,two,three,four,five,six,eight,ten,fiftn,twenty,
     *     hundrd
     *     /0.0d0,1.0d0,2.0d0,3.0d0,4.0d0,5.0d0,6.0d0,8.0d0,1.0d1,
     *      1.5d1,2.0d1,1.0d2/
      data c1,c3,c4,c5,c6,c9 /1.0d4,2.0d2,2.02d1,1.98d1,1.8d2,2.9d1/
      dfloat(ivar) = ivar
c
c     jacobian routine selector.
c
      go to (10,20,50,60,90,100,200,230,290,320,350,380,420,450),
     *      nprob
c
c     rosenbrock function.
c
   10 continue
      fjac(1,1) = -one
      fjac(1,2) = zero
      fjac(2,1) = -twenty*x(1)
      fjac(2,2) = ten
      go to 490
c
c     powell singular function.
c
   20 continue
      do 40 k = 1, 4
         do 30 j = 1, 4
            fjac(k,j) = zero
   30       continue
   40    continue
      fjac(1,1) = one
      fjac(1,2) = ten
      fjac(2,3) = dsqrt(five)
      fjac(2,4) = -fjac(2,3)
      fjac(3,2) = two*(x(2) - two*x(3))
      fjac(3,3) = -two*fjac(3,2)
      fjac(4,1) = two*dsqrt(ten)*(x(1) - x(4))
      fjac(4,4) = -fjac(4,1)
      go to 490
c
c     powell badly scaled function.
c
   50 continue
      fjac(1,1) = c1*x(2)
      fjac(1,2) = c1*x(1)
      fjac(2,1) = -dexp(-x(1))
      fjac(2,2) = -dexp(-x(2))
      go to 490
c
c     wood function.
c
   60 continue
      do 80 k = 1, 4
         do 70 j = 1, 4
            fjac(k,j) = zero
   70       continue
   80    continue
      temp1 = x(2) - three*x(1)**2
      temp2 = x(4) - three*x(3)**2
      fjac(1,1) = -c3*temp1 + one
      fjac(1,2) = -c3*x(1)
      fjac(2,1) = -two*c3*x(1)
      fjac(2,2) = c3 + c4
      fjac(2,4) = c5
      fjac(3,3) = -c6*temp2 + one
      fjac(3,4) = -c6*x(3)
      fjac(4,2) = c5
      fjac(4,3) = -two*c6*x(3)
      fjac(4,4) = c6 + c4
      go to 490
c
c     helical valley function.
c
   90 continue
      tpi = eight*datan(one)
      temp = x(1)**2 + x(2)**2
      temp1 = tpi*temp
      temp2 = dsqrt(temp)
      fjac(1,1) = hundrd*x(2)/temp1
      fjac(1,2) = -hundrd*x(1)/temp1
      fjac(1,3) = ten
      fjac(2,1) = ten*x(1)/temp2
      fjac(2,2) = ten*x(2)/temp2
      fjac(2,3) = zero
      fjac(3,1) = zero
      fjac(3,2) = zero
      fjac(3,3) = one
      go to 490
c
c     watson function.
c
  100 continue
      do 120 k = 1, n
         do 110 j = k, n
            fjac(k,j) = zero
  110       continue
  120    continue
      do 170 i = 1, 29
         ti = dfloat(i)/c9
         sum1 = zero
         temp = one
         do 130 j = 2, n
            sum1 = sum1 + dfloat(j-1)*temp*x(j)
            temp = ti*temp
  130       continue
         sum2 = zero
         temp = one
         do 140 j = 1, n
            sum2 = sum2 + temp*x(j)
            temp = ti*temp
  140       continue
         temp1 = two*(sum1 - sum2**2 - one)
         temp2 = two*sum2
         temp = ti**2
         tk = one
         do 160 k = 1, n
            tj = tk
            do 150 j = k, n
               fjac(k,j) = fjac(k,j)
     *                     + tj
     *                       *((dfloat(k-1)/ti - temp2)
     *                         *(dfloat(j-1)/ti - temp2) - temp1)
               tj = ti*tj
  150          continue
            tk = temp*tk
  160       continue
  170    continue
      fjac(1,1) = fjac(1,1) + six*x(1)**2 - two*x(2) + three
      fjac(1,2) = fjac(1,2) - two*x(1)
      fjac(2,2) = fjac(2,2) + one
      do 190 k = 1, n
         do 180 j = k, n
            fjac(j,k) = fjac(k,j)
  180       continue
  190    continue
      go to 490
c
c     chebyquad function.
c
  200 continue
      tk = one/dfloat(n)
      do 220 j = 1, n
         temp1 = one
         temp2 = two*x(j) - one
         temp = two*temp2
         temp3 = zero
         temp4 = two
         do 210 k = 1, n
            fjac(k,j) = tk*temp4
            ti = four*temp2 + temp*temp4 - temp3
            temp3 = temp4
            temp4 = ti
            ti = temp*temp2 - temp1
            temp1 = temp2
            temp2 = ti
  210       continue
  220    continue
      go to 490
c
c     brown almost-linear function.
c
  230 continue
      prod = one
      do 250 j = 1, n
         prod = x(j)*prod
         do 240 k = 1, n
            fjac(k,j) = one
  240       continue
         fjac(j,j) = two
  250    continue
      do 280 j = 1, n
         temp = x(j)
         if (temp .ne. zero) go to 270
         temp = one
         prod = one
         do 260 k = 1, n
            if (k .ne. j) prod = x(k)*prod
  260       continue
  270    continue
         fjac(n,j) = prod/temp
  280    continue
      go to 490
c
c     discrete boundary value function.
c
  290 continue
      h = one/dfloat(n+1)
      do 310 k = 1, n
         temp = three*(x(k) + dfloat(k)*h + one)**2
         do 300 j = 1, n
            fjac(k,j) = zero
  300       continue
         fjac(k,k) = two + temp*h**2/two
         if (k .ne. 1) fjac(k,k-1) = -one
         if (k .ne. n) fjac(k,k+1) = -one
  310    continue
      go to 490
c
c     discrete integral equation function.
c
  320 continue
      h = one/dfloat(n+1)
      do 340 k = 1, n
         tk = dfloat(k)*h
         do 330 j = 1, n
            tj = dfloat(j)*h
            temp = three*(x(j) + tj + one)**2
            fjac(k,j) = h*dmin1(tj*(one-tk),tk*(one-tj))*temp/two
  330       continue
         fjac(k,k) = fjac(k,k) + one
  340    continue
      go to 490
c
c     trigonometric function.
c
  350 continue
      do 370 j = 1, n
         temp = dsin(x(j))
         do 360 k = 1, n
            fjac(k,j) = temp
  360       continue
         fjac(j,j) = dfloat(j+1)*temp - dcos(x(j))
  370    continue
      go to 490
c
c     variably dimensioned function.
c
  380 continue
      sum = zero
      do 390 j = 1, n
         sum = sum + dfloat(j)*(x(j) - one)
  390    continue
      temp = one + six*sum**2
      do 410 k = 1, n
         do 400 j = k, n
            fjac(k,j) = dfloat(k*j)*temp
            fjac(j,k) = fjac(k,j)
  400       continue
         fjac(k,k) = fjac(k,k) + one
  410    continue
      go to 490
c
c     broyden tridiagonal function.
c
  420 continue
      do 440 k = 1, n
         do 430 j = 1, n
            fjac(k,j) = zero
  430       continue
         fjac(k,k) = three - four*x(k)
         if (k .ne. 1) fjac(k,k-1) = -one
         if (k .ne. n) fjac(k,k+1) = -two
  440    continue
      go to 490
c
c     broyden banded function.
c
  450 continue
      ml = 5
      mu = 1
      do 480 k = 1, n
         do 460 j = 1, n
            fjac(k,j) = zero
  460       continue
         k1 = max0(1,k-ml)
         k2 = min0(k+mu,n)
         do 470 j = k1, k2
            if (j .ne. k) fjac(k,j) = -(one + two*x(j))
  470       continue
         fjac(k,k) = two + fiftn*x(k)**2
  480    continue
  490 continue
      return
c
c     last card of subroutine vecjac.
c
      end
