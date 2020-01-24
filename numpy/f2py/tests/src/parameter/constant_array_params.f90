! Check that parameter arrays are correctly intercepted.
subroutine foo_array(x)
  implicit none
  integer, parameter :: dp = selected_real_kind(15)
  integer, parameter :: myparamarray(2) = (/ 3, 5 /)
  real(dp), intent(inout) :: x(myparamarray(1)) 
  real(dp), intent(inout) :: y(myparamarray(2))

  x = x/10
  y = y*10

  return
end subroutine
