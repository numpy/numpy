  module foddity
    use iso_fortran_env, only: int8, int16, int32, int64, real32, real64
    implicit none
    contains
      subroutine f_add(a, b, c)
        real(real64), intent(in) :: a, b
        real(real64), intent(out) :: c
        c = a + b
      end subroutine f_add
      subroutine f_addf(a, b, c)
        real(real32), intent(in) :: a, b
        real(real32), intent(out) :: c
        c = a + b
      end subroutine f_addf
      ! gh-9693
      function wat(x, y) result(z)
          integer(int32), intent(in) :: x, y
          integer(int32) :: z

          z = x + 7
      end function wat
      ! gh-25207
      subroutine f_add_int64(a, b, c)
        integer(int64), intent(in) :: a, b
        integer(int64), intent(out) :: c
        c = a + b
      end subroutine f_add_int64
      subroutine f_add_int16_arr(a, b, c)
        integer(int16), dimension(3), intent(in) :: a, b
        integer(int16), dimension(3), intent(out) :: c
        c = a + b
      end subroutine f_add_int16_arr
      subroutine f_add_int8_arr(a, b, c)
        integer(int8), dimension(3), intent(in) :: a, b
        integer(int8), dimension(3), intent(out) :: c
        c = a + b
      end subroutine f_add_int8_arr
      ! gh-25207
      subroutine add_arr(A, B, C)
         integer(int64), intent(in) :: A(3)
         integer(int64), intent(in) :: B(3)
         integer(int64), intent(out) :: C(3)
         integer :: j

         do j = 1, 3
            C(j) = A(j)+B(j)
         end do
      end subroutine
  end module foddity
