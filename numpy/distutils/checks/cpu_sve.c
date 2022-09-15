#include <arm_sve.h>

int main(void)
{
    svbool_t p = svptrue_b32();
    svint32_t a = svdup_s32(0);
    return (int)svaddv(p, a);
}
