      subroutine bar(fun,a,b)
      external fun
      integer a(1),b(1)
cf2py intent(in) a
cf2py intent(out) b
cf2py check(len(a)>0) a

cf2py      interface bar_user_interface 
cf2py        subroutine fun(a)
cf2py            integer dimension(1),intent(in,out),check(len(a)>0) :: a
cf2py        end subroutine fun
cf2py    end interface bar_user_interface

      call fun(a)

      b(1) = a(1)

      end


