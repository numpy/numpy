      subroutine f1()
         print *, "in f1, calling f2 twice.."
         call f2()
         call f2()
         return
      end
      
      subroutine f2()
cf2py    intent(callback, hide) fpy
         external fpy
         print *, "in f2, calling f2py.."
         call fpy()
         return
      end
