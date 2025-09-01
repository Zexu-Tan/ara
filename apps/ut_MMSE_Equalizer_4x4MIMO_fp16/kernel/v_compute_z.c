// Copyright 2025 Vodafone Sponsored Chair of Communication System, TU Dresden and Synopsys

// Author: Zexu Tan:zexu.tan@tu-dresden.de, Vodafone Chair, TU Dresden

#include <riscv_vector.h>
#include "arith_lib.h"


void v_compute_z(
    size_t nt, 
    size_t nr, 
    size_t vl, 
    float *H_real,
    float *H_imag,
    float *Y_real,
    float *Y_imag,
    float *z_real,
    float *z_imag) 
{
    vfloat32m1_t A_real, A_imag;
    vfloat32m1_t B_real, B_imag;
    vfloat32m1_t C_real, C_imag;
    vfloat32m1_t D_real, D_imag;
    vfloat32m1_t t_y_real, t_y_imag;

    vfloat32m1_t z0_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t z0_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t z1_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t z1_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t z2_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t z2_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t z3_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t z3_imag = vfmv_v_f_f32m1(0.0f, vl);


    for (int ptr_col = 0; ptr_col < nr; ptr_col++) {
        //y0
        t_y_real = vle32_v_f32m1(&Y_real[GET_IDX(ptr_col, 0, 0, 1, vl)], vl);
        t_y_imag = vle32_v_f32m1(&Y_imag[GET_IDX(ptr_col, 0, 0, 1, vl)], vl);

        //A0
        A_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_col, 0, 0, nr, vl)], vl);
        A_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_col, 0, 0, nr, vl)], vl);
        A_imag = vfneg_v_f32m1(A_imag, vl);

        vcfmac_vv_f32m1_sim(A_real, A_imag, t_y_real, t_y_imag, z0_real, z0_imag, &z0_real, &z0_imag, vl);

        
        //B0
        B_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_col, 1, 0, nr, vl)], vl);
        B_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_col, 1, 0, nr, vl)], vl);
        B_imag = vfneg_v_f32m1(B_imag, vl);

        vcfmac_vv_f32m1_sim(B_real, B_imag, t_y_real, t_y_imag, z1_real, z1_imag, &z1_real, &z1_imag, vl);

        
        //C0
        C_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_col, 2, 0, nr, vl)], vl);
        C_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_col, 2, 0, nr, vl)], vl);
        C_imag = vfneg_v_f32m1(C_imag, vl);

        vcfmac_vv_f32m1_sim(C_real, C_imag, t_y_real, t_y_imag, z2_real, z2_imag, &z2_real, &z2_imag, vl);
        
        //D0
        D_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_col, 3, 0, nr, vl)], vl);
        D_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_col, 3, 0, nr, vl)], vl);
        D_imag = vfneg_v_f32m1(D_imag, vl);

        vcfmac_vv_f32m1_sim(D_real, D_imag, t_y_real, t_y_imag, z3_real, z3_imag, &z3_real, &z3_imag, vl);
        
    }

    vse32_v_f32m1(&z_real[GET_IDX(0,0,0,1,vl)], z0_real, vl);
    vse32_v_f32m1(&z_imag[GET_IDX(0,0,0,1,vl)], z0_imag, vl);

    vse32_v_f32m1(&z_real[GET_IDX(1,0,0,1,vl)], z1_real, vl);
    vse32_v_f32m1(&z_imag[GET_IDX(1,0,0,1,vl)], z1_imag, vl);

    vse32_v_f32m1(&z_real[GET_IDX(2,0,0,1,vl)], z2_real, vl);
    vse32_v_f32m1(&z_imag[GET_IDX(2,0,0,1,vl)], z2_imag, vl);

    vse32_v_f32m1(&z_real[GET_IDX(3,0,0,1,vl)], z3_real, vl);
    vse32_v_f32m1(&z_imag[GET_IDX(3,0,0,1,vl)], z3_imag, vl);

        
}
