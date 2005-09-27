      subroutine foo(i)
      integer y
      character*10 str(3,2),str2
      common /foodata/ z , y (5 ),str,str2
      write(*,*) 'Fortran foo:1: i=',i,',y=',y,',z=',z,',str=',str
      y(1) = y(1)+i
      y(2) = y(2)+i+1
      y(3) = y(3)+333
      z = 777
      write(*,*) 'Fortran foo:2: i=',i,',y=',y,',z=',z
      end
