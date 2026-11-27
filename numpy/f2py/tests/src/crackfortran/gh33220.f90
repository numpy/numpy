subroutine probe(n, got)
  implicit none
  character(len=*), intent(in) :: n
  integer, intent(out) :: got
  ! The literal ends at its closing quote, a backslash is not an escape
  character(len=len('c:\'//n)) :: m
  got = len(m)
end subroutine probe

subroutine doubled(n, got)
  implicit none
  character(len=*), intent(in) :: n
  integer, intent(out) :: got
  character(len=len('it''s'//n)) :: m
  got = len(m)
end subroutine doubled
