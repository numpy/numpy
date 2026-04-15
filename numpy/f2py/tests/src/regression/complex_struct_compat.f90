      subroutine zero_imag(c, n)
        complex*16, intent(inout) :: c(n)
        integer, intent(in) :: n
        integer :: k
        do k = 1, n
          c(k) = cmplx(dble(c(k)), 0.0d0, kind=8)
        end do
      end subroutine
