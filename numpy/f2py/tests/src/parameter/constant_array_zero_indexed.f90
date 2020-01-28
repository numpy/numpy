! Check that zero-indexed parameter arrays are correctly intercepted.
subroutine foo_array_zero(x)
  implicit none
  integer, parameter :: dp = selected_real_kind(15)
  integer, parameter, dimension(0:2) :: zeroparamarray = (/ 3, 5, 7 /)
  real(dp), intent(inout) :: x(zeroparamarray(0))

  x = x/10

  return
end subroutine
