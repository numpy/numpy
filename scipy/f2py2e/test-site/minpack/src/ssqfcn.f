      subroutine ssqfcn(m,n,x,fvec,nprob)
      integer m,n,nprob
      double precision x(n),fvec(m)
c     **********
c
c     subroutine ssqfcn
c
c     this subroutine defines the functions of eighteen nonlinear
c     least squares problems. the allowable values of (m,n) for
c     functions 1,2 and 3 are variable but with m .ge. n.
c     for functions 4,5,6,7,8,9 and 10 the values of (m,n) are
c     (2,2),(3,3),(4,4),(2,2),(15,3),(11,4) and (16,3), respectively.
c     function 11 (watson) has m = 31 with n usually 6 or 9.
c     however, any n, n = 2,...,31, is permitted.
c     functions 12,13 and 14 have n = 3,2 and 4, respectively, but
c     allow any m .ge. n, with the usual choices being 10,10 and 20.
c     function 15 (chebyquad) allows m and n variable with m .ge. n.
c     function 16 (brown) allows n variable with m = n.
c     for functions 17 and 18, the values of (m,n) are
c     (33,5) and (65,11), respectively.
c
c     the subroutine statement is
c
c       subroutine ssqfcn(m,n,x,fvec,nprob)
c
c     where
c
c       m and n are positive integer input variables. n must not
c         exceed m.
c
c       x is an input array of length n.
c
c       fvec is an output array of length m which contains the nprob
c         function evaluated at x.
c
c       nprob is a positive integer input variable which defines the
c         number of the problem. nprob must not exceed 18.
c
c     subprograms called
c
c       fortran-supplied ... datan,dcos,dexp,dsin,dsqrt,dsign
c
c     argonne national laboratory. minpack project. march 1980.
c     burton s. garbow, kenneth e. hillstrom, jorge j. more
c
c     **********
      integer i,iev,ivar,j,nm1
      double precision c13,c14,c29,c45,div,dx,eight,five,one,prod,sum,
     *                 s1,s2,temp,ten,ti,tmp1,tmp2,tmp3,tmp4,tpi,two,
     *                 zero,zp25,zp5
      double precision v(11),y1(15),y2(11),y3(16),y4(33),y5(65)
      double precision dfloat
      data zero,zp25,zp5,one,two,five,eight,ten,c13,c14,c29,c45
     *     /0.0d0,2.5d-1,5.0d-1,1.0d0,2.0d0,5.0d0,8.0d0,1.0d1,1.3d1,
     *      1.4d1,2.9d1,4.5d1/
      data v(1),v(2),v(3),v(4),v(5),v(6),v(7),v(8),v(9),v(10),v(11)
     *     /4.0d0,2.0d0,1.0d0,5.0d-1,2.5d-1,1.67d-1,1.25d-1,1.0d-1,
     *      8.33d-2,7.14d-2,6.25d-2/
      data y1(1),y1(2),y1(3),y1(4),y1(5),y1(6),y1(7),y1(8),y1(9),
     *     y1(10),y1(11),y1(12),y1(13),y1(14),y1(15)
     *     /1.4d-1,1.8d-1,2.2d-1,2.5d-1,2.9d-1,3.2d-1,3.5d-1,3.9d-1,
     *      3.7d-1,5.8d-1,7.3d-1,9.6d-1,1.34d0,2.1d0,4.39d0/
      data y2(1),y2(2),y2(3),y2(4),y2(5),y2(6),y2(7),y2(8),y2(9),
     *     y2(10),y2(11)
     *     /1.957d-1,1.947d-1,1.735d-1,1.6d-1,8.44d-2,6.27d-2,4.56d-2,
     *      3.42d-2,3.23d-2,2.35d-2,2.46d-2/
      data y3(1),y3(2),y3(3),y3(4),y3(5),y3(6),y3(7),y3(8),y3(9),
     *     y3(10),y3(11),y3(12),y3(13),y3(14),y3(15),y3(16)
     *     /3.478d4,2.861d4,2.365d4,1.963d4,1.637d4,1.372d4,1.154d4,
     *      9.744d3,8.261d3,7.03d3,6.005d3,5.147d3,4.427d3,3.82d3,
     *      3.307d3,2.872d3/
      data y4(1),y4(2),y4(3),y4(4),y4(5),y4(6),y4(7),y4(8),y4(9),
     *     y4(10),y4(11),y4(12),y4(13),y4(14),y4(15),y4(16),y4(17),
     *     y4(18),y4(19),y4(20),y4(21),y4(22),y4(23),y4(24),y4(25),
     *     y4(26),y4(27),y4(28),y4(29),y4(30),y4(31),y4(32),y4(33)
     *     /8.44d-1,9.08d-1,9.32d-1,9.36d-1,9.25d-1,9.08d-1,8.81d-1,
     *      8.5d-1,8.18d-1,7.84d-1,7.51d-1,7.18d-1,6.85d-1,6.58d-1,
     *      6.28d-1,6.03d-1,5.8d-1,5.58d-1,5.38d-1,5.22d-1,5.06d-1,
     *      4.9d-1,4.78d-1,4.67d-1,4.57d-1,4.48d-1,4.38d-1,4.31d-1,
     *      4.24d-1,4.2d-1,4.14d-1,4.11d-1,4.06d-1/
      data y5(1),y5(2),y5(3),y5(4),y5(5),y5(6),y5(7),y5(8),y5(9),
     *     y5(10),y5(11),y5(12),y5(13),y5(14),y5(15),y5(16),y5(17),
     *     y5(18),y5(19),y5(20),y5(21),y5(22),y5(23),y5(24),y5(25),
     *     y5(26),y5(27),y5(28),y5(29),y5(30),y5(31),y5(32),y5(33),
     *     y5(34),y5(35),y5(36),y5(37),y5(38),y5(39),y5(40),y5(41),
     *     y5(42),y5(43),y5(44),y5(45),y5(46),y5(47),y5(48),y5(49),
     *     y5(50),y5(51),y5(52),y5(53),y5(54),y5(55),y5(56),y5(57),
     *     y5(58),y5(59),y5(60),y5(61),y5(62),y5(63),y5(64),y5(65)
     *     /1.366d0,1.191d0,1.112d0,1.013d0,9.91d-1,8.85d-1,8.31d-1,
     *      8.47d-1,7.86d-1,7.25d-1,7.46d-1,6.79d-1,6.08d-1,6.55d-1,
     *      6.16d-1,6.06d-1,6.02d-1,6.26d-1,6.51d-1,7.24d-1,6.49d-1,
     *      6.49d-1,6.94d-1,6.44d-1,6.24d-1,6.61d-1,6.12d-1,5.58d-1,
     *      5.33d-1,4.95d-1,5.0d-1,4.23d-1,3.95d-1,3.75d-1,3.72d-1,
     *      3.91d-1,3.96d-1,4.05d-1,4.28d-1,4.29d-1,5.23d-1,5.62d-1,
     *      6.07d-1,6.53d-1,6.72d-1,7.08d-1,6.33d-1,6.68d-1,6.45d-1,
     *      6.32d-1,5.91d-1,5.59d-1,5.97d-1,6.25d-1,7.39d-1,7.1d-1,
     *      7.29d-1,7.2d-1,6.36d-1,5.81d-1,4.28d-1,2.92d-1,1.62d-1,
     *      9.8d-2,5.4d-2/
      dfloat(ivar) = ivar
c
c     function routine selector.
c
      go to (10,40,70,110,120,130,140,150,170,190,210,250,270,290,310,
     *       360,390,410), nprob
c
c     linear function - full rank.
c
   10 continue
      sum = zero
      do 20 j = 1, n
         sum = sum + x(j)
   20    continue
      temp = two*sum/dfloat(m) + one
      do 30 i = 1, m
         fvec(i) = -temp
         if (i .le. n) fvec(i) = fvec(i) + x(i)
   30    continue
      go to 430
c
c     linear function - rank 1.
c
   40 continue
      sum = zero
      do 50 j = 1, n
         sum = sum + dfloat(j)*x(j)
   50    continue
      do 60 i = 1, m
         fvec(i) = dfloat(i)*sum - one
   60    continue
      go to 430
c
c     linear function - rank 1 with zero columns and rows.
c
   70 continue
      sum = zero
      nm1 = n - 1
      if (nm1 .lt. 2) go to 90
      do 80 j = 2, nm1
         sum = sum + dfloat(j)*x(j)
   80    continue
   90 continue
      do 100 i = 1, m
         fvec(i) = dfloat(i-1)*sum - one
  100    continue
      fvec(m) = -one
      go to 430
c
c     rosenbrock function.
c
  110 continue
      fvec(1) = ten*(x(2) - x(1)**2)
      fvec(2) = one - x(1)
      go to 430
c
c     helical valley function.
c
  120 continue
      tpi = eight*datan(one)
      tmp1 = dsign(zp25,x(2))
      if (x(1) .gt. zero) tmp1 = datan(x(2)/x(1))/tpi
      if (x(1) .lt. zero) tmp1 = datan(x(2)/x(1))/tpi + zp5
      tmp2 = dsqrt(x(1)**2+x(2)**2)
      fvec(1) = ten*(x(3) - ten*tmp1)
      fvec(2) = ten*(tmp2 - one)
      fvec(3) = x(3)
      go to 430
c
c     powell singular function.
c
  130 continue
      fvec(1) = x(1) + ten*x(2)
      fvec(2) = dsqrt(five)*(x(3) - x(4))
      fvec(3) = (x(2) - two*x(3))**2
      fvec(4) = dsqrt(ten)*(x(1) - x(4))**2
      go to 430
c
c     freudenstein and roth function.
c
  140 continue
      fvec(1) = -c13 + x(1) + ((five - x(2))*x(2) - two)*x(2)
      fvec(2) = -c29 + x(1) + ((one + x(2))*x(2) - c14)*x(2)
      go to 430
c
c     bard function.
c
  150 continue
      do 160 i = 1, 15
         tmp1 = dfloat(i)
         tmp2 = dfloat(16-i)
         tmp3 = tmp1
         if (i .gt. 8) tmp3 = tmp2
         fvec(i) = y1(i) - (x(1) + tmp1/(x(2)*tmp2 + x(3)*tmp3))
  160    continue
      go to 430
c
c     kowalik and osborne function.
c
  170 continue
      do 180 i = 1, 11
         tmp1 = v(i)*(v(i) + x(2))
         tmp2 = v(i)*(v(i) + x(3)) + x(4)
         fvec(i) = y2(i) - x(1)*tmp1/tmp2
  180    continue
      go to 430
c
c     meyer function.
c
  190 continue
      do 200 i = 1, 16
         temp = five*dfloat(i) + c45 + x(3)
         tmp1 = x(2)/temp
         tmp2 = dexp(tmp1)
         fvec(i) = x(1)*tmp2 - y3(i)
  200    continue
      go to 430
c
c     watson function.
c
  210 continue
      do 240 i = 1, 29
         div = dfloat(i)/c29
         s1 = zero
         dx = one
         do 220 j = 2, n
            s1 = s1 + dfloat(j-1)*dx*x(j)
            dx = div*dx
  220       continue
         s2 = zero
         dx = one
         do 230 j = 1, n
            s2 = s2 + dx*x(j)
            dx = div*dx
  230       continue
         fvec(i) = s1 - s2**2 - one
  240    continue
      fvec(30) = x(1)
      fvec(31) = x(2) - x(1)**2 - one
      go to 430
c
c     box 3-dimensional function.
c
  250 continue
      do 260 i = 1, m
         temp = dfloat(i)
         tmp1 = temp/ten
         fvec(i) = dexp(-tmp1*x(1)) - dexp(-tmp1*x(2))
     *             + (dexp(-temp) - dexp(-tmp1))*x(3)
  260    continue
      go to 430
c
c     jennrich and sampson function.
c
  270 continue
      do 280 i = 1, m
         temp = dfloat(i)
         fvec(i) = two + two*temp - dexp(temp*x(1)) - dexp(temp*x(2))
  280    continue
      go to 430
c
c     brown and dennis function.
c
  290 continue
      do 300 i = 1, m
         temp = dfloat(i)/five
         tmp1 = x(1) + temp*x(2) - dexp(temp)
         tmp2 = x(3) + dsin(temp)*x(4) - dcos(temp)
         fvec(i) = tmp1**2 + tmp2**2
  300    continue
      go to 430
c
c     chebyquad function.
c
  310 continue
      do 320 i = 1, m
         fvec(i) = zero
  320    continue
      do 340 j = 1, n
         tmp1 = one
         tmp2 = two*x(j) - one
         temp = two*tmp2
         do 330 i = 1, m
            fvec(i) = fvec(i) + tmp2
            ti = temp*tmp2 - tmp1
            tmp1 = tmp2
            tmp2 = ti
  330       continue
  340    continue
      dx = one/dfloat(n)
      iev = -1
      do 350 i = 1, m
         fvec(i) = dx*fvec(i)
         if (iev .gt. 0) fvec(i) = fvec(i) + one/(dfloat(i)**2 - one)
         iev = -iev
  350    continue
      go to 430
c
c     brown almost-linear function.
c
  360 continue
      sum = -dfloat(n+1)
      prod = one
      do 370 j = 1, n
         sum = sum + x(j)
         prod = x(j)*prod
  370    continue
      do 380 i = 1, n
         fvec(i) = x(i) + sum
  380    continue
      fvec(n) = prod - one
      go to 430
c
c     osborne 1 function.
c
  390 continue
      do 400 i = 1, 33
         temp = ten*dfloat(i-1)
         tmp1 = dexp(-x(4)*temp)
         tmp2 = dexp(-x(5)*temp)
         fvec(i) = y4(i) - (x(1) + x(2)*tmp1 + x(3)*tmp2)
  400    continue
      go to 430
c
c     osborne 2 function.
c
  410 continue
      do 420 i = 1, 65
         temp = dfloat(i-1)/ten
         tmp1 = dexp(-x(5)*temp)
         tmp2 = dexp(-x(6)*(temp-x(9))**2)
         tmp3 = dexp(-x(7)*(temp-x(10))**2)
         tmp4 = dexp(-x(8)*(temp-x(11))**2)
         fvec(i) = y5(i)
     *             - (x(1)*tmp1 + x(2)*tmp2 + x(3)*tmp3 + x(4)*tmp4)
  420    continue
  430 continue
      return
c
c     last card of subroutine ssqfcn.
c
      end
