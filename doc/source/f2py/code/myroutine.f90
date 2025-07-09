subroutine s(n, m, c, x)
	implicit none
  	integer, intent(in) :: n, m
  	real(kind=8), intent(out), dimension(n,m) :: x
  	real(kind=8), intent(in) :: c(:)

	x = 0.0d0
	x(1, 1) = c(1)

end subroutine s