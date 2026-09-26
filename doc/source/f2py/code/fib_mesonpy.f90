subroutine fib(a, n)
  use iso_c_binding
  integer(c_int), intent(in) :: n
  integer(c_int), intent(out) :: a(n)
  integer :: i
  do i = 1, n
     if (i == 1) then
        a(i) = 0
     else if (i == 2) then
        a(i) = 1
     else
        a(i) = a(i - 1) + a(i - 2)
     end if
  end do
end subroutine fib
