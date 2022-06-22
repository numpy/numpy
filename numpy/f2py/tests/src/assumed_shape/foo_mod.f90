
module mod

contains

   subroutine sum(x, res)
      implicit none
      real, intent(in) :: x(:)
      real, intent(out) :: res

      integer :: i

      !print *, "sum: size(x) = ", size(x)

      res = 0.0

      do i = 1, size(x)
         res = res + x(i)
      end do

   end subroutine sum

   function fsum(x) result(res)
      implicit none
      real, intent(in) :: x(:)
      real :: res

      integer :: i

      !print *, "fsum: size(x) = ", size(x)

      res = 0.0

      do i = 1, size(x)
         res = res + x(i)
      end do

   end function fsum

end module mod
