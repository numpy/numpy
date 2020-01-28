! Check that parameter arrays are correctly intercepted.
subroutine foo_array(x, y, z)
  implicit none
  integer, parameter :: dp = selected_real_kind(15)
  integer, parameter :: intparamarray(2) = (/ 3, 5 /)
  real(dp), parameter :: doubleparamarray(3) = (/ 3._dp, 4._dp, 6._dp /)
  real(dp), intent(inout) :: x(intparamarray(1))
  real(dp), intent(inout) :: y(intparamarray(2))
  real(dp), intent(out) :: z

  x = x/10
  y = y*10
  z = doubleparamarray(1)*doubleparamarray(2) + doubleparamarray(3)

  return
end subroutine
