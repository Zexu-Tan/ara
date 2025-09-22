#include <riscv_vector.h>
#include "arith_lib.h"

#include "printf.h"

void v_backward_substitution(
    size_t nt, 
    size_t nr, 
    size_t vl, 
    float *u_real,
    float *u_imag,
    float *L_real,
    float *L_imag,
    float *x_real,
    float *x_imag
){

    
    for (int ptr_row = nt - 1; ptr_row > -1; ptr_row--){
        vfloat32m1_t t_uk_real, t_uk_imag;
        vfloat32m1_t t_xk_real, t_xk_imag;
        vfloat32m1_t t_Lij_real, t_Lij_imag;
        vfloat32m1_t t_fr_Lii_real;
        vfloat32m1_t p0_real = vfmv_v_f_f32m1(0.0f, vl);
        vfloat32m1_t p0_imag = vfmv_v_f_f32m1(0.0f, vl);

        t_uk_real = vle32_v_f32m1(&u_real[GET_IDX(ptr_row, 0, 0, 1, vl)], vl);
        t_uk_imag = vle32_v_f32m1(&u_imag[GET_IDX(ptr_row, 0, 0, 1, vl)], vl);
        
        t_fr_Lii_real = vle32_v_f32m1(&L_real[GET_IDX(ptr_row, ptr_row, 0, nt, vl)], vl);

        for (int ptr_col = ptr_row + 1; ptr_col < nt; ptr_col++){
            t_xk_real = vle32_v_f32m1(&x_real[GET_IDX(ptr_col, 0, 0, 1, vl)], vl);
            t_xk_imag = vle32_v_f32m1(&x_imag[GET_IDX(ptr_col, 0, 0, 1, vl)], vl);

            t_Lij_real = vle32_v_f32m1(&L_real[GET_IDX(ptr_col, ptr_row, 0, nr, vl)], vl);
            t_Lij_imag = vle32_v_f32m1(&L_imag[GET_IDX(ptr_col, ptr_row, 0, nr, vl)], vl);
            t_Lij_imag = vfneg_v_f32m1(t_Lij_imag, vl);

            vcfmac_vv_f32m1_sim(t_Lij_real, t_Lij_imag, t_xk_real, t_xk_imag, p0_real, p0_imag, &p0_real, &p0_imag, vl);
        }


        t_uk_real = vfsub_vv_f32m1(t_uk_real, p0_real, vl);
        t_uk_imag = vfsub_vv_f32m1(t_uk_imag, p0_imag, vl);

        t_uk_real = vfmul_vv_f32m1(t_uk_real, t_fr_Lii_real, vl);
        t_uk_imag = vfmul_vv_f32m1(t_uk_imag, t_fr_Lii_real, vl);

        vse32_v_f32m1(&x_real[GET_IDX(ptr_row, 0, 0, 1, vl)], t_uk_real, vl);
        vse32_v_f32m1(&x_imag[GET_IDX(ptr_row, 0, 0, 1, vl)], t_uk_imag, vl);


    }

        





}