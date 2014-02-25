cFile: simple.f
      subroutine foo(a,m,n)
      integer m,n,i,j
      real a(m,n)
cf2py intent(in,out) a
cf2py intent(hide) m,n
      do i=1,m
         do j=1,n
            a(i,j) = a(i,j) + 10*i+j
         enddo
      enddo
      end
cEOF
