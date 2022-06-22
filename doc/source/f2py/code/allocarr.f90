module mod
   real, allocatable, dimension(:, :) :: b
contains
   subroutine foo
      integer k
      if (allocated(b)) then
         print *, "b=["
         do k = 1, size(b, 1)
            print *, b(k, 1:size(b, 2))
         end do
         print *, "]"
      else
         print *, "b is not allocated"
      end if
   end subroutine foo
end module mod
