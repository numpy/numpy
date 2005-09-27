      subroutine exp1(l,u,n)
C     Input: n is number of iterations
C     Output: l,u are such that
C       l(1)/l(2) < exp(1) < u(1)/u(2)
C
Cf2py integer*4 :: n = 1
Cf2py intent(out) l,u
      integer*4 n,i
      real*8 l(2),u(2),t,t1,t2,t3,t4
      l(2) = 1
      l(1) = 0
      u(2) = 0
      u(1) = 1
      do 10 i=0,n
         t1 = 4 + 32*(1+i)*i
         t2 = 11 + (40+32*i)*i
         t3 = 3 + (24+32*i)*i
         t4 = 8 + 32*(1+i)*i
         t = u(1)
         u(1) = l(1)*t1 + t*t2
         l(1) = l(1)*t3 + t*t4
         t = u(2)
         u(2) = l(2)*t1 + t*t2
         l(2) = l(2)*t3 + t*t4
 10   continue
      end
