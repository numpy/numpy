module mod
  real, allocatable, dimension(:,:) :: b 
contains
  subroutine foo
    integer k
    if (allocated(b)) then
       print*, "b=["
       do k = 1,size(b,1)
          print*, b(k,1:size(b,2))
       enddo
       print*, "]"
    else
       print*, "b is not allocated"
    endif
  end subroutine foo
end module mod
