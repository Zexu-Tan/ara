#ifndef _ARITH_LIB_H_
#define _ARITH_LIB_H_

#include <stdint.h>
#include <string.h>


#define VCFMUL_VV_SIM(ar, ai, br, bi, cr, ci, vl)        \
  asm volatile (                                            \
    "vsetvli zero, %0, e32,m1\n"                            \
    "vfmul.vv " #cr ", " #ar ", " #br "\n"                  \
    "vfmul.vv v1, "   #ai ", " #bi "\n"                     \
    "vfsub.vv " #cr ", " #cr ", v1\n"                       \
    "vfmul.vv " #ci ", " #ar ", " #bi "\n"                  \
    "vfmul.vv v1, "   #ai ", " #br "\n"                     \
    "vfadd.vv " #ci ", " #ci ", v1\n"                       \
    : /* no outputs */                                      \
    : "r"(vl)                                               \
    : "v1"                                                  \
  )

void vcfmul_vv_f32m1_sim(vfloat32m1_t vr_ar, vfloat32m1_t vr_ai, vfloat32m1_t vr_br, vfloat32m1_t vr_bi, vfloat32m1_t *vr_cr, vfloat32m1_t *vr_ci, size_t vl);

#endif

