#ifndef __loongarch_asx
#error "HOST/ARCH doesn't support LASX"
#endif

#include <lasxintrin.h>

int main(void)
{
    __m256i a = __lasx_xvadd_d(__lasx_xvldi(0), __lasx_xvldi(0));
    return __lasx_xvpickve2gr_w(a, 0);
}
