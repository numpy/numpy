#if (__VEC__ < 10301) || (__ARCH__ < 11)
    #error VX not supported
#endif

#include <vecintrin.h>
#include<stdio.h>

__vector int input= {1, 2, 4, 5 };

int main(void)
{
   __vector int  out;
   out = vec_abs(input);
   return out[0];
}
