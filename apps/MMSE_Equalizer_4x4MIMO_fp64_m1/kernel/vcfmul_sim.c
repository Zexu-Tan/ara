// Copyright 2025 Vodafone Sponsored Chair of Communication System, TU Dresden and Synopsys

// Author: Zexu Tan:zexu.tan@tu-dresden.de, Vodafone Chair, TU Dresden


#include <riscv_vector.h>
#include "arith_lib.h"




vfloat32m1_t vcfmul_vv_f32m1_sim_real(
    vfloat32m1_t ar, vfloat32m1_t ai,
    vfloat32m1_t br, vfloat32m1_t bi, 
    size_t vl
){
    return vfnmsac_vv_f32m1(vfmul_vv_f32m1(ar, br, vl), ai, bi, vl);
}

vfloat32m1_t vcfmul_vv_f32m1_sim_imag(
    vfloat32m1_t ar, vfloat32m1_t ai,
    vfloat32m1_t br, vfloat32m1_t bi, 
    size_t vl
){
    return vfmacc_vv_f32m1(vfmul_vv_f32m1(ar, bi, vl), ai, br, vl);
}







