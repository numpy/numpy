      subroutine foo()
      integer i,j
      common /c12data3/ i,j
      write(*,*) 'Fortran foo: i=',i,',j=',j
      end
      subroutine bar()
      integer i
      real*8 r(3,2),h
Cf2py note(this is 2-d array) r
Cf2py intent(hide) h
      common /c12data3/ i(2)
      common /anodata/ r,h
      write(*,*) 'Fortran bar: i=',i
      write(*,*) 'Fortran bar: r=',r
      end
