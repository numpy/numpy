
subroutine foo(a, n, m, b)
  implicit none

  real, intent(in) :: a(n, m)
  integer, intent(in) :: n, m
  real, intent(out) :: b(size(a, 1))

  integer :: i

  do i = 1, size(b)
    b(i) = sum(a(i,:))
  enddo
end subroutine
