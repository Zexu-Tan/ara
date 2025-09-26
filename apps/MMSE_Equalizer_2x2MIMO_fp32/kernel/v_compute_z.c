// Copyright 2025 Vodafone Sponsored Chair of Communication System, TU Dresden and Synopsys

// Author: Zexu Tan:zexu.tan@tu-dresden.de, Vodafone Chair, TU Dresden

#include <riscv_vector.h>
#include "arith_lib.h"


void v_compute_z_fp32(
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

    vfloat32m1_t t_y_real, t_y_imag;
    vfloat32m1_t A_real, A_imag, B_real, B_imag, C_real, C_imag, D_real, D_imag;


    vfloat32m1_t z0_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t z0_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t z1_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t z1_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t z2_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t z2_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t z3_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t z3_imag = vfmv_v_f_f32m1(0.0f, vl);


    for (int ptr_col = 0; ptr_col < nr - 1; ptr_col++) {
        
        t_y_real = vle32_v_f32m1(&Y_real[GET_IDX(ptr_col, 0, 0, 1, vl)], vl);
        A_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_col, 0, 0, nr, vl)], vl);

        z0_real = vfmacc_vv_f32m1(z0_real, t_y_real, A_real, vl);
        
        t_y_imag = vle32_v_f32m1(&Y_imag[GET_IDX(ptr_col, 0, 0, 1, vl)], vl);
        z0_imag = vfmacc_vv_f32m1(z0_imag, t_y_imag, A_real, vl);

        A_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_col, 0, 0, nr, vl)], vl);
        z0_real = vfmacc_vv_f32m1(z0_real, t_y_imag, A_imag, vl);

        z0_imag = vfnmsac_vv_f32m1(z0_imag, t_y_real, A_imag, vl);

        

        //B0
        B_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_col, 1, 0, nr, vl)], vl);
        z1_real = vfmacc_vv_f32m1(z1_real, t_y_real, B_real, vl);

        B_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_col, 1, 0, nr, vl)], vl);
        z1_real = vfmacc_vv_f32m1(z1_real, t_y_imag, B_imag, vl);

        z1_imag = vfmacc_vv_f32m1(z1_imag, t_y_imag, B_real, vl);

        z1_imag = vfnmsac_vv_f32m1(z1_imag, t_y_real, B_imag, vl);

        
        
        //C0
        C_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_col, 2, 0, nr, vl)], vl);
        z2_real = vfmacc_vv_f32m1(z2_real, t_y_real, C_real, vl);

        C_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_col, 2, 0, nr, vl)], vl);
        z2_real = vfmacc_vv_f32m1(z2_real, t_y_imag, C_imag, vl);

        z2_imag = vfmacc_vv_f32m1(z2_imag, t_y_imag, C_real, vl);

        z2_imag = vfnmsac_vv_f32m1(z2_imag, t_y_real, C_imag, vl);
        
        //D0
        D_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_col, 3, 0, nr, vl)], vl);
        z3_real = vfmacc_vv_f32m1(z3_real, t_y_real, D_real, vl);

        D_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_col, 3, 0, nr, vl)], vl);
        z3_real = vfmacc_vv_f32m1(z3_real, t_y_imag, D_imag, vl);

        z3_imag = vfmacc_vv_f32m1(z3_imag, t_y_imag, D_real, vl);

        z3_imag = vfnmsac_vv_f32m1(z3_imag, t_y_real, D_imag, vl);
        
    }

    int16_t t_col = nt - 1;

    //A3
    t_y_real = vle32_v_f32m1(&Y_real[GET_IDX(t_col, 0, 0, 1, vl)], vl);
    A_real = vle32_v_f32m1(&H_real[GET_IDX(t_col, 0, 0, nr, vl)], vl);

    z0_real = vfmacc_vv_f32m1(z0_real, t_y_real, A_real, vl);
        
    t_y_imag = vle32_v_f32m1(&Y_imag[GET_IDX(t_col, 0, 0, 1, vl)], vl);
    z0_imag = vfmacc_vv_f32m1(z0_imag, t_y_imag, A_real, vl);

    A_imag = vle32_v_f32m1(&H_imag[GET_IDX(t_col, 0, 0, nr, vl)], vl);
    z0_real = vfmacc_vv_f32m1(z0_real, t_y_imag, A_imag, vl);

    vse32_v_f32m1(&z_real[GET_IDX(0,0,0,1,vl)], z0_real, vl);

    z0_imag = vfnmsac_vv_f32m1(z0_imag, t_y_real, A_imag, vl);
    
    vse32_v_f32m1(&z_imag[GET_IDX(0,0,0,1,vl)], z0_imag, vl);
        

    //B3
    B_real = vle32_v_f32m1(&H_real[GET_IDX(t_col, 1, 0, nr, vl)], vl);
    z1_real = vfmacc_vv_f32m1(z1_real, t_y_real, B_real, vl);

    B_imag = vle32_v_f32m1(&H_imag[GET_IDX(t_col, 1, 0, nr, vl)], vl);
    z1_real = vfmacc_vv_f32m1(z1_real, t_y_imag, B_imag, vl);

    z1_imag = vfmacc_vv_f32m1(z1_imag, t_y_imag, B_real, vl);

    vse32_v_f32m1(&z_real[GET_IDX(1,0,0,1,vl)], z1_real, vl);

    z1_imag = vfnmsac_vv_f32m1(z1_imag, t_y_real, B_imag, vl);

    vse32_v_f32m1(&z_imag[GET_IDX(1,0,0,1,vl)], z1_imag, vl);
    
    
        
    //C3
    C_real = vle32_v_f32m1(&H_real[GET_IDX(t_col, 2, 0, nr, vl)], vl);
    z2_real = vfmacc_vv_f32m1(z2_real, t_y_real, C_real, vl);

    C_imag = vle32_v_f32m1(&H_imag[GET_IDX(t_col, 2, 0, nr, vl)], vl);
    z2_real = vfmacc_vv_f32m1(z2_real, t_y_imag, C_imag, vl);

    z2_imag = vfmacc_vv_f32m1(z2_imag, t_y_imag, C_real, vl);

    vse32_v_f32m1(&z_real[GET_IDX(2,0,0,1,vl)], z2_real, vl);

    z2_imag = vfnmsac_vv_f32m1(z2_imag, t_y_real, C_imag, vl);

    vse32_v_f32m1(&z_imag[GET_IDX(2,0,0,1,vl)], z2_imag, vl);


        
    //D3
    D_real = vle32_v_f32m1(&H_real[GET_IDX(t_col, 3, 0, nr, vl)], vl);
    z3_real = vfmacc_vv_f32m1(z3_real, t_y_real, D_real, vl);

    D_imag = vle32_v_f32m1(&H_imag[GET_IDX(t_col, 3, 0, nr, vl)], vl);
    z3_real = vfmacc_vv_f32m1(z3_real, t_y_imag, D_imag, vl);

    z3_imag = vfmacc_vv_f32m1(z3_imag, t_y_imag, D_real, vl);

    vse32_v_f32m1(&z_real[GET_IDX(3,0,0,1,vl)], z3_real, vl);

    z3_imag = vfnmsac_vv_f32m1(z3_imag, t_y_real, D_imag, vl);

    vse32_v_f32m1(&z_imag[GET_IDX(3,0,0,1,vl)], z3_imag, vl);



        
}
