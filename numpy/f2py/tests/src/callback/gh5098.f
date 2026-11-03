      subroutine array_result(array_cb, n, x)
      external array_cb
      integer n
      double precision x(n)
      call array_cb(n, x)
      end

      subroutine strings_result(strings_cb, n, s)
      external strings_cb
      integer n
      character*4 s(n)
      call strings_cb(n, s)
      end

      subroutine string_result(string_cb, s)
      external string_cb
      character*4 s
      call string_cb(s)
      end
