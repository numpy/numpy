#if (__VEC__ < 10302) || (__ARCH__ < 12)
    #error VXE not supported
#endif

#include <vecintrin.h>
#include <stdio.h>

int main(void) {
  __vector float a = {
       25.0, 36.0, 81.0, 100.0
  };

  __vector float d;

  d  = vec_nabs(a);

  return (int)d[0];
}
