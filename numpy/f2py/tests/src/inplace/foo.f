c     Test inplace calculations in array c, by squaring all its values.
c     As a sanity check on the input, stores the original content in copy.
      subroutine inplace(c, m1, m2, copy)
      integer*4 m1, m2, i, j
      real*4 c(m1, m2), copy(m1, m2)
cf2py intent(inplace) c
cf2py intent(out) copy
cf2py integer, depend(c), intent(hide) :: m1 = len(c)
cf2py integer, depend(c), intent(hide) :: m2 = shape(c, 1)
      do i=1,m1
         do j=1,m2
            copy(i, j) = c(i, j)
            c(i, j) = c(i, j) ** 2
         end do
      end do
      end

      subroutine inplace_out(c, m1, m2, copy)
      integer*4 m1, m2, i, j
      real*4 c(m1, m2), copy(m1, m2)
cf2py intent(inplace, out) c
cf2py intent(out) copy
cf2py integer, depend(c), intent(hide) :: m1 = len(c)
cf2py integer, depend(c), intent(hide) :: m2 = shape(c, 1)
      do i=1,m1
         do j=1,m2
            copy(i, j) = c(i, j)
            c(i, j) = c(i, j) ** 2
         end do
      end do
      end
