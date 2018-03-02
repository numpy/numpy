/*
Adapted from random123's threefry.h
*/

#include <inttypes.h>
#include <assert.h>

enum r123_enum_threefry64x4
{
    /* These are the R_256 constants from the Threefish reference sources
       with names changed to R_64x4... */
    R_64x4_0_0 = 14,
    R_64x4_0_1 = 16,
    R_64x4_1_0 = 52,
    R_64x4_1_1 = 57,
    R_64x4_2_0 = 23,
    R_64x4_2_1 = 40,
    R_64x4_3_0 = 5,
    R_64x4_3_1 = 37,
    R_64x4_4_0 = 25,
    R_64x4_4_1 = 33,
    R_64x4_5_0 = 46,
    R_64x4_5_1 = 12,
    R_64x4_6_0 = 58,
    R_64x4_6_1 = 22,
    R_64x4_7_0 = 32,
    R_64x4_7_1 = 32
};

struct r123array4x64
{
    uint64_t v[4];
}; /* r123array4x64 */

typedef struct r123array4x64 threefry4x64_key_t;
typedef struct r123array4x64 threefry4x64_ctr_t;

static __inline _forceinline uint64_t RotL_64(uint64_t x, unsigned int N);
static __inline uint64_t RotL_64(uint64_t x, unsigned int N)
{
    return (x << (N & 63)) | (x >> ((64 - N) & 63));
}


static __inline _forceinline threefry4x64_ctr_t threefry4x64_R(unsigned int Nrounds, threefry4x64_ctr_t in, threefry4x64_key_t k);
static __inline threefry4x64_ctr_t threefry4x64_R(unsigned int Nrounds, threefry4x64_ctr_t in, threefry4x64_key_t k)
{
    threefry4x64_ctr_t X;
    uint64_t ks[4 + 1];
    int i;
    (void)((!!(Nrounds <= 72)) || (_wassert(L"Nrounds<=72", L"c:\\temp\\random123-1.09\\include\\random123\\threefry.h", (unsigned)(728)), 0));
    ks[4] = ((0xA9FC1A22) + (((uint64_t)(0x1BD11BDA)) << 32));
    for (i = 0; i < 4; i++)
    {
        ks[i] = k.v[i];
        X.v[i] = in.v[i];
        ks[4] ^= k.v[i];
    }
    X.v[0] += ks[0];
    X.v[1] += ks[1];
    X.v[2] += ks[2];
    X.v[3] += ks[3];
    if (Nrounds > 0)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 1)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 2)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 3)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 3)
    {
        X.v[0] += ks[1];
        X.v[1] += ks[2];
        X.v[2] += ks[3];
        X.v[3] += ks[4];
        X.v[4 - 1] += 1;
    }
    if (Nrounds > 4)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 5)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 6)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 7)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 7)
    {
        X.v[0] += ks[2];
        X.v[1] += ks[3];
        X.v[2] += ks[4];
        X.v[3] += ks[0];
        X.v[4 - 1] += 2;
    }
    if (Nrounds > 8)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 9)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 10)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 11)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 11)
    {
        X.v[0] += ks[3];
        X.v[1] += ks[4];
        X.v[2] += ks[0];
        X.v[3] += ks[1];
        X.v[4 - 1] += 3;
    }
    if (Nrounds > 12)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 13)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 14)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 15)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 15)
    {
        X.v[0] += ks[4];
        X.v[1] += ks[0];
        X.v[2] += ks[1];
        X.v[3] += ks[2];
        X.v[4 - 1] += 4;
    }
    if (Nrounds > 16)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 17)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 18)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 19)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 19)
    {
        X.v[0] += ks[0];
        X.v[1] += ks[1];
        X.v[2] += ks[2];
        X.v[3] += ks[3];
        X.v[4 - 1] += 5;
    }
    if (Nrounds > 20)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 21)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 22)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 23)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 23)
    {
        X.v[0] += ks[1];
        X.v[1] += ks[2];
        X.v[2] += ks[3];
        X.v[3] += ks[4];
        X.v[4 - 1] += 6;
    }
    if (Nrounds > 24)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 25)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 26)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 27)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 27)
    {
        X.v[0] += ks[2];
        X.v[1] += ks[3];
        X.v[2] += ks[4];
        X.v[3] += ks[0];
        X.v[4 - 1] += 7;
    }
    if (Nrounds > 28)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 29)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 30)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 31)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 31)
    {
        X.v[0] += ks[3];
        X.v[1] += ks[4];
        X.v[2] += ks[0];
        X.v[3] += ks[1];
        X.v[4 - 1] += 8;
    }
    if (Nrounds > 32)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 33)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 34)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 35)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 35)
    {
        X.v[0] += ks[4];
        X.v[1] += ks[0];
        X.v[2] += ks[1];
        X.v[3] += ks[2];
        X.v[4 - 1] += 9;
    }
    if (Nrounds > 36)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 37)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 38)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 39)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 39)
    {
        X.v[0] += ks[0];
        X.v[1] += ks[1];
        X.v[2] += ks[2];
        X.v[3] += ks[3];
        X.v[4 - 1] += 10;
    }
    if (Nrounds > 40)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 41)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 42)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 43)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 43)
    {
        X.v[0] += ks[1];
        X.v[1] += ks[2];
        X.v[2] += ks[3];
        X.v[3] += ks[4];
        X.v[4 - 1] += 11;
    }
    if (Nrounds > 44)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 45)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 46)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 47)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 47)
    {
        X.v[0] += ks[2];
        X.v[1] += ks[3];
        X.v[2] += ks[4];
        X.v[3] += ks[0];
        X.v[4 - 1] += 12;
    }
    if (Nrounds > 48)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 49)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 50)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 51)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 51)
    {
        X.v[0] += ks[3];
        X.v[1] += ks[4];
        X.v[2] += ks[0];
        X.v[3] += ks[1];
        X.v[4 - 1] += 13;
    }
    if (Nrounds > 52)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 53)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 54)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 55)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 55)
    {
        X.v[0] += ks[4];
        X.v[1] += ks[0];
        X.v[2] += ks[1];
        X.v[3] += ks[2];
        X.v[4 - 1] += 14;
    }
    if (Nrounds > 56)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 57)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 58)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 59)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 59)
    {
        X.v[0] += ks[0];
        X.v[1] += ks[1];
        X.v[2] += ks[2];
        X.v[3] += ks[3];
        X.v[4 - 1] += 15;
    }
    if (Nrounds > 60)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 61)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 62)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 63)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 63)
    {
        X.v[0] += ks[1];
        X.v[1] += ks[2];
        X.v[2] += ks[3];
        X.v[3] += ks[4];
        X.v[4 - 1] += 16;
    }
    if (Nrounds > 64)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_0_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_0_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 65)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_1_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_1_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 66)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_2_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_2_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 67)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_3_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_3_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 67)
    {
        X.v[0] += ks[2];
        X.v[1] += ks[3];
        X.v[2] += ks[4];
        X.v[3] += ks[0];
        X.v[4 - 1] += 17;
    }
    if (Nrounds > 68)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_4_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_4_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 69)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_5_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_5_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 70)
    {
        X.v[0] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_6_0);
        X.v[1] ^= X.v[0];
        X.v[2] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_6_1);
        X.v[3] ^= X.v[2];
    }
    if (Nrounds > 71)
    {
        X.v[0] += X.v[3];
        X.v[3] = RotL_64(X.v[3], R_64x4_7_0);
        X.v[3] ^= X.v[0];
        X.v[2] += X.v[1];
        X.v[1] = RotL_64(X.v[1], R_64x4_7_1);
        X.v[1] ^= X.v[2];
    }
    if (Nrounds > 71)
    {
        X.v[0] += ks[3];
        X.v[1] += ks[4];
        X.v[2] += ks[0];
        X.v[3] += ks[1];
        X.v[4 - 1] += 18;
    }
    return X;
}
enum r123_enum_threefry4x64
{
    threefry4x64_rounds = 20
};

typedef struct s_threefry_state {
  threefry4x64_key_t *ctr;
  threefry4x64_ctr_t *key;
  int buffer_pos;
  uint64_t buffer[4];
  int has_uint32;
  uint32_t uinteger;
} threefry_state;


static inline uint64_t threefry_next(threefry_state *state) {
  /* TODO: This 4 should be a constant somewhere */
  if (state->buffer_pos >= 4) {
    /* generate 4 new uint64_t */
    int i;
    threefry4x64_ctr_t ct;
    state->ctr->v[0]++;
    ct = threefry4x64_R(threefry4x64_rounds, *state->ctr, *state->key);
    for (i=0; i<4; i++){
        state->buffer[i] = ct.v[i];
    }
    state->buffer_pos = 0;
  }
  uint64_t out = state->buffer[state->buffer_pos];
  state->buffer_pos++;
  return out;
}

static inline uint64_t threefry_next64(threefry_state *state) {
  return threefry_next(state);
}

static inline uint64_t threefry_next32(threefry_state *state) {
  if (state->has_uint32) {
    state->has_uint32 = 0;
    return state->uinteger;
  }
  uint64_t next = threefry_next(state);

  state->has_uint32 = 1;
  state->uinteger = (uint32_t)(next & 0xffffffff);

  return (uint32_t)(next >> 32);
}
