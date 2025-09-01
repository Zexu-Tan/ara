

#include <riscv_vector.h>



void vcwmul_vv_i16m1_sim(
    vint16m1_t ar, vint16m1_t ai,
    vint16m1_t br, vint16m1_t bi,
    vint32m2_t *cr,
    vint32m2_t *ci,
    size_t vl)
{
    // real: ar*br - ai*bi
    vint32m2_t r = vwmul_vv_i32m2(ar, br, vl);
    *cr = vnmsac_vv_i32m2(r, ai, bi, vl);   // r -= ai*bi

    // imag: ar*bi + ai*br
    vint32m2_t i = vwmul_vv_i32m2(ar, bi, vl);
    *ci = vmacc_vv_i32m2(i, ai, br, vl);    // i += ai*br
}
