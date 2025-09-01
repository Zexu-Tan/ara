#include <riscv_vector.h>
#include "arith_lib.h"

void v_forward_substitution(
    size_t nt, 
    size_t nr, 
    size_t vl, 
    float *z_real,
    float *z_imag,
    float *L_real,
    float *L_imag,
    float *u_real,
    float *u_imag
){

    for (int ptr_row = 0; ptr_row < nt; ptr_row++){
        vfloat32m1_t t_zk_real, t_zk_imag;
        vfloat32m1_t t_uk_real, t_uk_imag;
        vfloat32m1_t t_Lij_real, t_Lij_imag;
        vfloat32m1_t t_fr_Lii_real;
        vfloat32m1_t p0_real = vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t p0_imag = vfmv_v_f_f32m1(0.0f, vl);

        t_zk_real = vle32_v_f32m1(&z_real[GET_IDX(ptr_row, 0, 0, 1, vl)], vl);
        t_zk_imag = vle32_v_f32m1(&z_imag[GET_IDX(ptr_row, 0, 0, 1, vl)], vl);
        
        t_fr_Lii_real = vle32_v_f32m1(&L_real[GET_IDX(ptr_row, ptr_row, 0, nt, vl)], vl);
        //t_fr_Lii_real = vfrsqrt7_v_f32m1(t_fr_Lii_real, vl);
        //t_fr_Lii_real = vfmul_vv_f32m1(t_fr_Lii_real, t_fr_Lii_real, vl);


        for (int ptr_col = 0; ptr_col < ptr_row; ptr_col++){
            t_uk_real = vle32_v_f32m1(&u_real[GET_IDX(ptr_col, 0, 0, 1, vl)], vl);
            t_uk_imag = vle32_v_f32m1(&u_imag[GET_IDX(ptr_col, 0, 0, 1, vl)], vl);

            t_Lij_real = vle32_v_f32m1(&L_real[GET_IDX(ptr_row, ptr_col, 0, nr, vl)], vl);
            t_Lij_imag = vle32_v_f32m1(&L_imag[GET_IDX(ptr_row, ptr_col, 0, nr, vl)], vl);

            vcfmac_vv_f32m1_sim(t_Lij_real, t_Lij_imag, t_uk_real, t_uk_imag, p0_real, p0_imag, &p0_real, &p0_imag, vl);

        }
        
        t_zk_real = vfsub_vv_f32m1(t_zk_real, p0_real, vl);
        t_zk_imag = vfsub_vv_f32m1(t_zk_imag, p0_imag, vl);

        t_zk_real = vfdiv_vv_f32m1(t_zk_real, t_fr_Lii_real, vl);
        t_zk_imag = vfdiv_vv_f32m1(t_zk_imag, t_fr_Lii_real, vl);

        vse32_v_f32m1(&u_real[GET_IDX(ptr_row, 0, 0, 1, vl)], t_zk_real, vl);
        vse32_v_f32m1(&u_imag[GET_IDX(ptr_row, 0, 0, 1, vl)], t_zk_imag, vl);

    }





}