! Test file for gh-30809: intent(inplace) deprecation warning
subroutine inplace_sum(arr, n)
    implicit none
    integer, intent(hide), depend(arr) :: n = len(arr)
    real*8, intent(inplace) :: arr(n)
    integer :: i

    do i = 1, n
        arr(i) = arr(i) + 1.0d0
    end do
end subroutine inplace_sum
