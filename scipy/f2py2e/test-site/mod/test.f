      subroutine test
      use data_mod
      implicit none
      integer i, j
      
      ni = 2
      nj = 4
      
      allocate(data(ni, nj))
      
      data(1,1)=2.
      data(1,2)=3.
      data(1,3)=4.
      data(1,4)=5.
      data(2,1)=12.
      data(2,2)=13.
      data(2,3)=14.
      data(2,4)=15.

      do i = 1, ni
         do j = 1, nj
            print *, i, j, data(i, j)
         end do
      end do
      
      end subroutine test
