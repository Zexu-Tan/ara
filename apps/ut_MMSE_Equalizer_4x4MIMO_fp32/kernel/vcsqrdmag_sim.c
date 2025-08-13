// Copyright 2025 Vodafone Sponsored Chair of Communication System, TU Dresden and Synopsys

// Author: Zexu Tan:zexu.tan@tu-dresden.de, Vodafone Chair, TU Dresden


#include <riscv_vector.h>
#include "arith_lib.h"

void vcsqrdmag_vv_f32m1_sim(    
    vfloat32m1_t vr_ar,
    vfloat32m1_t vr_ai,
    vfloat32m1_t *vr_cr,
    size_t vl)
{
    vfloat32m1_t sqr_ar = vfmul_vv_f32m1(vr_ar, vr_ar, vl);
    vfloat32m1_t sqr_ai = vfmul_vv_f32m1(vr_ai, vr_ai, vl);
    *vr_cr = vfadd_vv_f32m1(sqr_ar, sqr_ai, vl);
}


