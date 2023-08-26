  module coddity
    use iso_c_binding, only: c_double
    implicit none
    contains
      subroutine c_add(a, b, c) bind(c, name="c_add")
        real(c_double), intent(in) :: a, b
        real(c_double), intent(out) :: c
        c = a + b
      end subroutine c_add
  end module coddity
