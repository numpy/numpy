
import os
from scipy.distutils.core import setup, Extension
from distutils.dep_util import newer

fib3_f = '''
C FILE: FIB3.F
      SUBROUTINE FIB(A,N)
C
C     CALCULATE FIRST N FIBONACCI NUMBERS
C
      INTEGER N
      REAL*8 A(N)
Cf2py intent(in) n
Cf2py intent(out) a
Cf2py depend(n) a
      DO I=1,N
         IF (I.EQ.1) THEN
            A(I) = 0.0D0
         ELSEIF (I.EQ.2) THEN
            A(I) = 1.0D0
         ELSE 
            A(I) = A(I-1) + A(I-2)
         ENDIF
      ENDDO
      END
C END FILE FIB3.F
'''

package = 'gen_ext'

def source_func(ext, src_dir):
    source = os.path.join(src_dir,'fib3.f')
    if newer(__file__, source):
        f = open(source,'w')
        f.write(fib3_f)
        f.close()
    return [source]

ext = Extension(package+'.fib3',[source_func])

setup(
    name = package,
    ext_modules = [ext],
    packages = [package+'.tests',package],
    package_dir = {package:'.'})

