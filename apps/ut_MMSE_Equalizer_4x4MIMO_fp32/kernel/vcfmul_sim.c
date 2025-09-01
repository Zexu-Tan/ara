// Copyright 2025 Vodafone Sponsored Chair of Communication System, TU Dresden and Synopsys

// Author: Zexu Tan:zexu.tan@tu-dresden.de, Vodafone Chair, TU Dresden


#include <riscv_vector.h>
#include "arith_lib.h"


void vcfmul_vv_f32m1_sim(
    vfloat32m1_t ar, vfloat32m1_t ai,
    vfloat32m1_t br, vfloat32m1_t bi,
    vfloat32m1_t *cr,
    vfloat32m1_t *ci,
    size_t vl)
{
    // real: ar*br - ai*bi
    vfloat32m1_t r = vfmul_vv_f32m1(ar, br, vl);
    *cr = vfnmsac_vv_f32m1(r, ai, bi, vl);   // r -= ai*bi

    // imag: ar*bi + ai*br
    vfloat32m1_t i = vfmul_vv_f32m1(ar, bi, vl);
    *ci = vfmacc_vv_f32m1(i, ai, br, vl);    // i += ai*br
}


void vcfmac_vv_f32m1_sim(  
    //c = a * b + p0  
    vfloat32m1_t a_real,
    vfloat32m1_t a_imag,
    vfloat32m1_t b_real,
    vfloat32m1_t b_imag,
    vfloat32m1_t p0_real,
    vfloat32m1_t p0_imag,
    vfloat32m1_t *c_real,
    vfloat32m1_t *c_imag,
    size_t vl){
    
    vfloat32m1_t t0_real, t0_imag;
    vcfmul_vv_f32m1_sim(a_real, a_imag, b_real, b_imag, &t0_real, &t0_imag, vl);

    *c_real = vfadd_vv_f32m1(p0_real, t0_real, vl);
    *c_imag = vfadd_vv_f32m1(p0_imag, t0_imag, vl);
    
    
}

void vcfnmsac_vv_f32m1_sim(  
    //c = -a*b + p0
    vfloat32m1_t a_real,
    vfloat32m1_t a_imag,
    vfloat32m1_t b_real,
    vfloat32m1_t b_imag,
    vfloat32m1_t p0_real,
    vfloat32m1_t p0_imag,
    vfloat32m1_t *c_real,
    vfloat32m1_t *c_imag,
    size_t vl){
    
    vfloat32m1_t t0_real, t0_imag;
    vcfmul_vv_f32m1_sim(a_real, a_imag, b_real, b_imag, &t0_real, &t0_imag, vl);

    *c_real = vfsub_vv_f32m1(p0_real, t0_real, vl);
    *c_imag = vfsub_vv_f32m1(p0_imag, t0_imag, vl);
    
}

