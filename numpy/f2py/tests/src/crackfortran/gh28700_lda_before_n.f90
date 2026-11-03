subroutine probe(a, lda, n, work, gotn, gotlda)
  implicit none
  integer :: lda, n, gotn, gotlda
  real :: a(lda, n)
  real :: work(3 * n)
  gotn = n
  gotlda = lda
  if (n > 0) then
    work(1) = work(1)
  end if
end subroutine probe
