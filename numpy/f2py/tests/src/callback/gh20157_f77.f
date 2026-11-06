C Legacy fixed-form callback (gh-20157 / review request).
C The assumed-shape F90 fix must leave this path on bare EXTERNAL.
      integer function addcb(cb, n)
      integer cb, n
      external cb
      addcb = cb(n)
      return
      end
