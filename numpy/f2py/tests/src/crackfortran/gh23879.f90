module gh23879
    implicit none
    private
    public :: foo

 contains

    subroutine foo42bar(a, b)
       integer, intent(in) :: a
       integer, intent(out) :: b
       b = a
       call bar(b)
    end subroutine

    subroutine bar1337baz(x)
        integer, intent(inout) :: x
        x = 2*x
     end subroutine

 end module gh23879
