      subroutine ssqjac(m,n,x,fjac,ldfjac,nprob)
      integer m,n,ldfjac,nprob
      double precision x(n),fjac(ldfjac,n)
c     **********
c
c     subroutine ssqjac
c
c     this subroutine defines the jacobian matrices of eighteen
c     nonlinear least squares problems. the problem dimensions are
c     as described in the prologue comments of ssqfcn.
c
c     the subroutine statement is
c
c       subroutine ssqjac(m,n,x,fjac,ldfjac,nprob)
c
c     where
c
c       m and n are positive integer input variables. n must not
c         exceed m.
c
c       x is an input array of length n.
c
c       fjac is an m by n output array which contains the jacobian
c         matrix of the nprob function evaluated at x.
c
c       ldfjac is a positive integer input variable not less than m
c         which specifies the leading dimension of the array fjac.
c
c       nprob is a positive integer variable which defines the
c         number of the problem. nprob must not exceed 18.
c
c     subprograms called
c
c       fortran-supplied ... datan,dcos,dexp,dsin,dsqrt
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********
      integer i,ivar,j,k,mm1,nm1
      double precision c14,c20,c29,c45,c100,div,dx,eight,five,four,
     *                 one,prod,s2,temp,ten,three,ti,tmp1,tmp2,tmp3,
     *                 tmp4,tpi,two,zero
      double precision v(11)
      double precision dfloat
      data zero,one,two,three,four,five,eight,ten,c14,c20,c29,c45,c100
     *     /0.0d0,1.0d0,2.0d0,3.0d0,4.0d0,5.0d0,8.0d0,1.0d1,1.4d1,
     *      2.0d1,2.9d1,4.5d1,1.0d2/
      data v(1),v(2),v(3),v(4),v(5),v(6),v(7),v(8),v(9),v(10),v(11)
     *     /4.0d0,2.0d0,1.0d0,5.0d-1,2.5d-1,1.67d-1,1.25d-1,1.0d-1,
     *      8.33d-2,7.14d-2,6.25d-2/
      dfloat(ivar) = ivar
c
c     jacobian routine selector.
c
      go to (10,40,70,130,140,150,180,190,210,230,250,310,330,350,370,
     *       400,460,480), nprob
c
c     linear function - full rank.
c
   10 continue
      temp = two/dfloat(m)
      do 30 j = 1, n
         do 20 i = 1, m
            fjac(i,j) = -temp
   20       continue
         fjac(j,j) = fjac(j,j) + one
   30    continue
      go to 500
c
c     linear function - rank 1.
c
   40 continue
      do 60 j = 1, n
         do 50 i = 1, m
            fjac(i,j) = dfloat(i)*dfloat(j)
   50       continue
   60    continue
      go to 500
c
c     linear function - rank 1 with zero columns and rows.
c
   70 continue
      do 90 j = 1, n
         do 80 i = 1, m
            fjac(i,j) = zero
   80       continue
   90    continue
      nm1 = n - 1
      mm1 = m - 1
      if (nm1 .lt. 2) go to 120
      do 110 j = 2, nm1
         do 100 i = 2, mm1
            fjac(i,j) = dfloat(i-1)*dfloat(j)
  100       continue
  110    continue
  120 continue
      go to 500
c
c     rosenbrock function.
c
  130 continue
      fjac(1,1) = -c20*x(1)
      fjac(1,2) = ten
      fjac(2,1) = -one
      fjac(2,2) = zero
      go to 500
c
c     helical valley function.
c
  140 continue
      tpi = eight*datan(one)
      temp = x(1)**2 + x(2)**2
      tmp1 = tpi*temp
      tmp2 = dsqrt(temp)
      fjac(1,1) = c100*x(2)/tmp1
      fjac(1,2) = -c100*x(1)/tmp1
      fjac(1,3) = ten
      fjac(2,1) = ten*x(1)/tmp2
      fjac(2,2) = ten*x(2)/tmp2
      fjac(2,3) = zero
      fjac(3,1) = zero
      fjac(3,2) = zero
      fjac(3,3) = one
      go to 500
c
c     powell singular function.
c
  150 continue
      do 170 j = 1, 4
         do 160 i = 1, 4
            fjac(i,j) = zero
  160       continue
  170    continue
      fjac(1,1) = one
      fjac(1,2) = ten
      fjac(2,3) = dsqrt(five)
      fjac(2,4) = -fjac(2,3)
      fjac(3,2) = two*(x(2) - two*x(3))
      fjac(3,3) = -two*fjac(3,2)
      fjac(4,1) = two*dsqrt(ten)*(x(1) - x(4))
      fjac(4,4) = -fjac(4,1)
      go to 500
c
c     freudenstein and roth function.
c
  180 continue
      fjac(1,1) = one
      fjac(1,2) = x(2)*(ten - three*x(2)) - two
      fjac(2,1) = one
      fjac(2,2) = x(2)*(two + three*x(2)) - c14
      go to 500
c
c     bard function.
c
  190 continue
      do 200 i = 1, 15
         tmp1 = dfloat(i)
         tmp2 = dfloat(16-i)
         tmp3 = tmp1
         if (i .gt. 8) tmp3 = tmp2
         tmp4 = (x(2)*tmp2 + x(3)*tmp3)**2
         fjac(i,1) = -one
         fjac(i,2) = tmp1*tmp2/tmp4
         fjac(i,3) = tmp1*tmp3/tmp4
  200    continue
      go to 500
c
c     kowalik and osborne function.
c
  210 continue
      do 220 i = 1, 11
         tmp1 = v(i)*(v(i) + x(2))
         tmp2 = v(i)*(v(i) + x(3)) + x(4)
         fjac(i,1) = -tmp1/tmp2
         fjac(i,2) = -v(i)*x(1)/tmp2
         fjac(i,3) = fjac(i,1)*fjac(i,2)
         fjac(i,4) = fjac(i,3)/v(i)
  220    continue
      go to 500
c
c     meyer function.
c
  230 continue
      do 240 i = 1, 16
         temp = five*dfloat(i) + c45 + x(3)
         tmp1 = x(2)/temp
         tmp2 = dexp(tmp1)
         fjac(i,1) = tmp2
         fjac(i,2) = x(1)*tmp2/temp
         fjac(i,3) = -tmp1*fjac(i,2)
  240    continue
      go to 500
c
c     watson function.
c
  250 continue
      do 280 i = 1, 29
         div = dfloat(i)/c29
         s2 = zero
         dx = one
         do 260 j = 1, n
            s2 = s2 + dx*x(j)
            dx = div*dx
  260       continue
         temp = two*div*s2
         dx = one/div
         do 270 j = 1, n
            fjac(i,j) = dx*(dfloat(j-1) - temp)
            dx = div*dx
  270       continue
  280    continue
      do 300 j = 1, n
         do 290 i = 30, 31
            fjac(i,j) = zero
  290       continue
  300    continue
      fjac(30,1) = one
      fjac(31,1) = -two*x(1)
      fjac(31,2) = one
      go to 500
c
c     box 3-dimensional function.
c
  310 continue
      do 320 i = 1, m
         temp = dfloat(i)
         tmp1 = temp/ten
         fjac(i,1) = -tmp1*dexp(-tmp1*x(1))
         fjac(i,2) = tmp1*dexp(-tmp1*x(2))
         fjac(i,3) = dexp(-temp) - dexp(-tmp1)
  320    continue
      go to 500
c
c     jennrich and sampson function.
c
  330 continue
      do 340 i = 1, m
         temp = dfloat(i)
         fjac(i,1) = -temp*dexp(temp*x(1))
         fjac(i,2) = -temp*dexp(temp*x(2))
  340    continue
      go to 500
c
c     brown and dennis function.
c
  350 continue
      do 360 i = 1, m
         temp = dfloat(i)/five
         ti = dsin(temp)
         tmp1 = x(1) + temp*x(2) - dexp(temp)
         tmp2 = x(3) + ti*x(4) - dcos(temp)
         fjac(i,1) = two*tmp1
         fjac(i,2) = temp*fjac(i,1)
         fjac(i,3) = two*tmp2
         fjac(i,4) = ti*fjac(i,3)
  360    continue
      go to 500
c
c     chebyquad function.
c
  370 continue
      dx = one/dfloat(n)
      do 390 j = 1, n
         tmp1 = one
         tmp2 = two*x(j) - one
         temp = two*tmp2
         tmp3 = zero
         tmp4 = two
         do 380 i = 1, m
            fjac(i,j) = dx*tmp4
            ti = four*tmp2 + temp*tmp4 - tmp3
            tmp3 = tmp4
            tmp4 = ti
            ti = temp*tmp2 - tmp1
            tmp1 = tmp2
            tmp2 = ti
  380       continue
  390    continue
      go to 500
c
c     brown almost-linear function.
c
  400 continue
      prod = one
      do 420 j = 1, n
         prod = x(j)*prod
         do 410 i = 1, n
            fjac(i,j) = one
  410       continue
         fjac(j,j) = two
  420    continue
      do 450 j = 1, n
         temp = x(j)
         if (temp .ne. zero) go to 440
         temp = one
         prod = one
         do 430 k = 1, n
            if (k .ne. j) prod = x(k)*prod
  430       continue
  440    continue
         fjac(n,j) = prod/temp
  450    continue
      go to 500
c
c     osborne 1 function.
c
  460 continue
      do 470 i = 1, 33
         temp = ten*dfloat(i-1)
         tmp1 = dexp(-x(4)*temp)
         tmp2 = dexp(-x(5)*temp)
         fjac(i,1) = -one
         fjac(i,2) = -tmp1
         fjac(i,3) = -tmp2
         fjac(i,4) = temp*x(2)*tmp1
         fjac(i,5) = temp*x(3)*tmp2
  470    continue
      go to 500
c
c     osborne 2 function.
c
  480 continue
      do 490 i = 1, 65
         temp = dfloat(i-1)/ten
         tmp1 = dexp(-x(5)*temp)
         tmp2 = dexp(-x(6)*(temp-x(9))**2)
         tmp3 = dexp(-x(7)*(temp-x(10))**2)
         tmp4 = dexp(-x(8)*(temp-x(11))**2)
         fjac(i,1) = -tmp1
         fjac(i,2) = -tmp2
         fjac(i,3) = -tmp3
         fjac(i,4) = -tmp4
         fjac(i,5) = temp*x(1)*tmp1
         fjac(i,6) = x(2)*(temp - x(9))**2*tmp2
         fjac(i,7) = x(3)*(temp - x(10))**2*tmp3
         fjac(i,8) = x(4)*(temp - x(11))**2*tmp4
         fjac(i,9) = -two*x(2)*x(6)*(temp - x(9))*tmp2
         fjac(i,10) = -two*x(3)*x(7)*(temp - x(10))*tmp3
         fjac(i,11) = -two*x(4)*x(8)*(temp - x(11))*tmp4
  490    continue
  500 continue
      return
c
c     last card of subroutine ssqjac.
c
      end
