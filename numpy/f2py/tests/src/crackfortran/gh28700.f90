module mod1
    character(6), public, parameter :: mkdir = 'mkdir '
    character(7), public, parameter :: badvar2 = "badvar2"
    character(13), public, parameter :: realdep = 'mkdir '//badvar2
end module mod1

module np_bug
contains
    subroutine sub1
        use mod1
    end subroutine sub1
end module np_bug
