      subroutine foo(a,m,n)
      integer a(m,n), m,n,i,j
      print*, "F77:"
      print*, "m=",m,", n=",n
      do 100,i=1,m
         print*, "Row ",i,":"
         do 50,j=1,n
            print*, "a(i=",i,",j=",j,") = ",a(i,j)
 50      continue
 100  continue
      if (m*n.gt.0) a(1,1) = 77777
      end
      
