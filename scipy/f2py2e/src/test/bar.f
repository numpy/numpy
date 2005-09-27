      subroutine bar()
      integer a
      real*8 b,c(3)
      common /foodata/ a,b,c
      a = 4
      b = 6.7
      c(2) = 3.0
      write(*,*) "bar:a=",a
      write(*,*) "bar:b=",b
      write(*,*) "bar:c=",c
      end
