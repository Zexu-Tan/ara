// Copyright 2025 Vodafone Sponsored Chair of Communication System, TU Dresden and Synopsys

// Author: Zexu Tan:zexu.tan@tu-dresden.de, Vodafone Chair, TU Dresden

#include <riscv_vector.h>
#include "../lib.h"


void v_compute_z_cfxp16(
    size_t nt, 
    size_t nr, 
    size_t vl,
    size_t offset,
    size_t n_subcarrier,
    int16_t *H_real,
    int16_t *H_imag,
    int16_t *Y_real,
    int16_t *Y_imag,
    int16_t *z_real,
    int16_t *z_imag) 
{

    vint16m1_t t_y_real, t_y_imag;
    vint16m1_t A_real, A_imag, B_real, B_imag, C_real, C_imag, D_real, D_imag;


    vint32m2_t acc_z0_real = vmv_v_x_i32m2(0, vl);
    vint32m2_t acc_z0_imag = vmv_v_x_i32m2(0, vl);
    vint32m2_t acc_z1_real = vmv_v_x_i32m2(0, vl);
    vint32m2_t acc_z1_imag = vmv_v_x_i32m2(0, vl);
    vint32m2_t acc_z2_real = vmv_v_x_i32m2(0, vl);
    vint32m2_t acc_z2_imag = vmv_v_x_i32m2(0, vl);
    vint32m2_t acc_z3_real = vmv_v_x_i32m2(0, vl);
    vint32m2_t acc_z3_imag = vmv_v_x_i32m2(0, vl);


    for (int ptr_col = 0; ptr_col < nr - 1; ptr_col++) {
        //A0
        t_y_real = vle16_v_i16m1(&Y_real[GET_IDX(ptr_col, 0, offset, 1, n_subcarrier)], vl);
        A_real = vle16_v_i16m1(&H_real[GET_IDX(ptr_col, 0, offset, nr, n_subcarrier)], vl);
        acc_z0_real = vwmacc_vv_i32m2(acc_z0_real, t_y_real, A_real, vl);

        t_y_imag = vle16_v_i16m1(&Y_imag[GET_IDX(ptr_col, 0, offset, 1, n_subcarrier)], vl);
        acc_z0_imag = vwmacc_vv_i32m2(acc_z0_imag, t_y_imag, A_real, vl);
        
        A_imag = vle16_v_i16m1(&H_imag[GET_IDX(ptr_col, 0, offset, nr, n_subcarrier)], vl);
        acc_z0_real = vwmacc_vv_i32m2(acc_z0_real, t_y_imag, A_imag, vl);
        acc_z0_imag = vsub_vv_i32m2(acc_z0_imag, vwmul_vv_i32m2(t_y_real, A_imag, vl),vl);


        //B0
        B_real = vle16_v_i16m1(&H_real[GET_IDX(ptr_col, 1, offset, nr, n_subcarrier)], vl);
        acc_z1_real = vwmacc_vv_i32m2(acc_z1_real, t_y_real, B_real, vl);
        acc_z1_imag = vwmacc_vv_i32m2(acc_z1_imag, t_y_imag, B_real, vl);
        
        B_imag = vle16_v_i16m1(&H_imag[GET_IDX(ptr_col, 1, offset, nr, n_subcarrier)], vl);
        acc_z1_real = vwmacc_vv_i32m2(acc_z1_real, t_y_imag, B_imag, vl);
        acc_z1_imag = vsub_vv_i32m2(acc_z1_imag, vwmul_vv_i32m2(t_y_real, B_imag, vl),vl);

        //C0
        C_real = vle16_v_i16m1(&H_real[GET_IDX(ptr_col, 2, offset, nr, n_subcarrier)], vl);
        acc_z2_real = vwmacc_vv_i32m2(acc_z2_real, t_y_real, C_real, vl);
        acc_z2_imag = vwmacc_vv_i32m2(acc_z2_imag, t_y_imag, C_real, vl);
        
        C_imag = vle16_v_i16m1(&H_imag[GET_IDX(ptr_col, 2, offset, nr, n_subcarrier)], vl);
        acc_z2_real = vwmacc_vv_i32m2(acc_z2_real, t_y_imag, C_imag, vl);
        acc_z2_imag = vsub_vv_i32m2(acc_z2_imag, vwmul_vv_i32m2(t_y_real, C_imag, vl),vl);


        //D0
        D_real = vle16_v_i16m1(&H_real[GET_IDX(ptr_col, 3, offset, nr, n_subcarrier)], vl);
        acc_z3_real = vwmacc_vv_i32m2(acc_z3_real, t_y_real, D_real, vl);
        acc_z3_imag = vwmacc_vv_i32m2(acc_z3_imag, t_y_imag, D_real, vl);
        
        D_imag = vle16_v_i16m1(&H_imag[GET_IDX(ptr_col, 3, offset, nr, n_subcarrier)], vl);
        acc_z3_real = vwmacc_vv_i32m2(acc_z3_real, t_y_imag, D_imag, vl);
        acc_z3_imag = vsub_vv_i32m2(acc_z3_imag, vwmul_vv_i32m2(t_y_real, D_imag, vl),vl);
        
        
    }

    int16_t t_col = nt - 1;

    //A3
    t_y_real = vle16_v_i16m1(&Y_real[GET_IDX(t_col, 0, offset, 1, n_subcarrier)], vl);
    A_real = vle16_v_i16m1(&H_real[GET_IDX(t_col, 0, offset, nr, n_subcarrier)], vl);
    acc_z0_real = vwmacc_vv_i32m2(acc_z0_real, t_y_real, A_real, vl);

    t_y_imag = vle16_v_i16m1(&Y_imag[GET_IDX(t_col, 0, offset, 1, n_subcarrier)], vl);
    acc_z0_imag = vwmacc_vv_i32m2(acc_z0_imag, t_y_imag, A_real, vl);
        
    A_imag = vle16_v_i16m1(&H_imag[GET_IDX(t_col, 0, offset, nr, n_subcarrier)], vl);
    acc_z0_real = vwmacc_vv_i32m2(acc_z0_real, t_y_imag, A_imag, vl);

    vse16_v_i16m1(&z_real[GET_IDX(0,0,offset, 1, n_subcarrier)], vnclip_wx_i16m1(acc_z0_real, 10, vl), vl);
    acc_z0_imag = vsub_vv_i32m2(acc_z0_imag, vwmul_vv_i32m2(t_y_real, A_imag, vl),vl);
    vse16_v_i16m1(&z_imag[GET_IDX(0,0,offset, 1, n_subcarrier)], vnclip_wx_i16m1(acc_z0_imag, 10, vl), vl);



    //B3
    B_real = vle16_v_i16m1(&H_real[GET_IDX(t_col, 1, offset, nr, n_subcarrier)], vl);
    acc_z1_real = vwmacc_vv_i32m2(acc_z1_real, t_y_real, B_real, vl);
    acc_z1_imag = vwmacc_vv_i32m2(acc_z1_imag, t_y_imag, B_real, vl);
    B_imag = vle16_v_i16m1(&H_imag[GET_IDX(t_col, 1, offset, nr, n_subcarrier)], vl);
    acc_z1_real = vwmacc_vv_i32m2(acc_z1_real, t_y_imag, B_imag, vl);
    vse16_v_i16m1(&z_real[GET_IDX(1,0,offset, 1, n_subcarrier)], vnclip_wx_i16m1(acc_z1_real, 10, vl), vl);
    acc_z1_imag = vsub_vv_i32m2(acc_z1_imag, vwmul_vv_i32m2(t_y_real, B_imag, vl),vl);
    vse16_v_i16m1(&z_imag[GET_IDX(1,0,offset, 1, n_subcarrier)], vnclip_wx_i16m1(acc_z1_imag, 10, vl), vl);
    
    
    //C3
    C_real = vle16_v_i16m1(&H_real[GET_IDX(t_col, 2, offset, nr, n_subcarrier)], vl);
    acc_z2_real = vwmacc_vv_i32m2(acc_z2_real, t_y_real, C_real, vl);
    acc_z2_imag = vwmacc_vv_i32m2(acc_z2_imag, t_y_imag, C_real, vl);
    C_imag = vle16_v_i16m1(&H_imag[GET_IDX(t_col, 2, offset, nr, n_subcarrier)], vl);
    acc_z2_real = vwmacc_vv_i32m2(acc_z2_real, t_y_imag, C_imag, vl);
    vse16_v_i16m1(&z_real[GET_IDX(2,0,offset, 1, n_subcarrier)], vnclip_wx_i16m1(acc_z2_real, 10, vl), vl);
    acc_z2_imag = vsub_vv_i32m2(acc_z2_imag, vwmul_vv_i32m2(t_y_real, C_imag, vl),vl);
    vse16_v_i16m1(&z_imag[GET_IDX(2,0,offset, 1, n_subcarrier)], vnclip_wx_i16m1(acc_z2_imag, 10, vl), vl);


    //D3
    D_real = vle16_v_i16m1(&H_real[GET_IDX(t_col, 3, offset, nr, n_subcarrier)], vl);
    acc_z3_real = vwmacc_vv_i32m2(acc_z3_real, t_y_real, D_real, vl);
    acc_z3_imag = vwmacc_vv_i32m2(acc_z3_imag, t_y_imag, D_real, vl);
    D_imag = vle16_v_i16m1(&H_imag[GET_IDX(t_col, 3, offset, nr, n_subcarrier)], vl);
    acc_z3_real = vwmacc_vv_i32m2(acc_z3_real, t_y_imag, D_imag, vl);
    vse16_v_i16m1(&z_real[GET_IDX(3,0,offset, 1, n_subcarrier)], vnclip_wx_i16m1(acc_z3_real, 10, vl), vl);
    acc_z3_imag = vsub_vv_i32m2(acc_z3_imag, vwmul_vv_i32m2(t_y_real, D_imag, vl),vl);
    vse16_v_i16m1(&z_imag[GET_IDX(3,0,offset, 1, n_subcarrier)], vnclip_wx_i16m1(acc_z3_imag, 10, vl), vl);
        
}
