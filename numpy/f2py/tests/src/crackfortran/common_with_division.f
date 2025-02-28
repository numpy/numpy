      subroutine test
      implicit none
      integer lmu,lb,lub,lpmin
      parameter (lmu=1)
      parameter (lb=20)
c     crackfortran fails to parse this  
      parameter (lub=(lb-1)*lmu+1)
c     crackfortran can successfully parse this though
c     parameter (lub=lb*lmu-lmu+1)
      parameter (lpmin=1)

c     crackfortran fails to parse this correctly 
      common /mortmp/ ctmp((lub*(lub+1)*(lub+1))/lpmin+1)
      return
      end
