! Check that parameter arrays are correctly intercepted.
subroutine foo_array(x)
  implicit none
  integer, parameter :: dp = selected_real_kind(15)
  real(dp), parameter :: myparamarray_d(2) = (/ 3._dp, 5._dp /)
  real(dp), intent(inout) :: x
  dimension x(3)
  x(1) = x(1) + myparamarray_d(1)
  x(2) = x(2) * myparamarray_d(2)
  x(3) = myparamarray_d(1) + myparamarray_d(2)
  return
end subroutine
