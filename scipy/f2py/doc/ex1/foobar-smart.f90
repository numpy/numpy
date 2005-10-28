!%f90
module foobar ! in 
  note(This module contains two examples that are used in &
       \texttt{f2py} documentation.) foobar
    interface  ! in :foobar
        subroutine foo(a) ! in :foobar:foo.f
            note(Example of a wrapper function of a Fortran subroutine.) foo
            integer intent(inout),&
                 note(5 is added to the variable {{}\verb@a@{}} ``in place''.) :: a
        end subroutine foo
        function bar(a,b) result (ab) ! in :foobar:bar.f
            integer :: a
            integer :: b
            integer :: ab
            note(The first value.) a
            note(The second value.) b
            note(Add two values.) bar
            note(The result.) ab
        end function bar
    end interface 
end module foobar

! This file was auto-generated with f2py (version:0.95).
! See http://cens.ioc.ee/projects/f2py2e/
