      function dfoo (a,b,c)
      real*8 a,b,c,dfoo
      dfoo=a+b+c
      a=b
      c=a
      end
      function cfoo (a,b,c)
      complex a,b,c
      complex*16 cfoo
      cfoo=a+b+c
      c=a
      a=b
      end
      function ifoo (a,b,c)
      integer a,b,c,ifoo
      ifoo=a+b+c
      c=a
      a=b
      x=a
      end
      subroutine sfoo (a,b,c)
      character*(*) a,b,c
      a(1:3)=c(2:5)
      b(1:6) = a // c
      end
