#include <riscv_vector.h>
#include "../lib.h"

void v_forward_substitution_cfxp16(
  size_t nt, 
  size_t nr, 
  size_t vl,
  size_t offset,
  size_t n_subcarrier,
    int16_t *z_real,
    int16_t *z_imag,
    int16_t *L_real,
    int16_t *L_imag,
    int16_t *u_real,
    int16_t *u_imag
){  

    vint32m2_t t_zk_real, t_zk_imag;
    vint16m1_t t_uk_real, t_uk_imag;
    vint16m1_t t_Lij_real, t_Lij_imag;
    vint16m1_t t_fr_Lii_real;
    vint32m2_t p0_real, p0_imag;

    for (int ptr_row = 0; ptr_row < nt; ptr_row++){
        t_zk_real = vmv_v_x_i32m2(0, vl);
        t_zk_imag = vmv_v_x_i32m2(0, vl);

        t_zk_real = vwadd_vx_i32m2(vle16_v_i16m1(&z_real[GET_IDX(ptr_row, 0, offset, nr, n_subcarrier)], vl), 0, vl);
        t_zk_imag = vwadd_vx_i32m2(vle16_v_i16m1(&z_imag[GET_IDX(ptr_row, 0, offset, nr, n_subcarrier)], vl), 0, vl);
        
        t_fr_Lii_real = vle16_v_i16m1(&L_real[GET_IDX(ptr_row, ptr_row, offset, nr, n_subcarrier)], vl);


        for (int ptr_col = 0; ptr_col < ptr_row; ptr_col++){
            t_uk_real = vle16_v_i16m1(&u_real[GET_IDX(ptr_col, 0, offset, nr, n_subcarrier)], vl);
            t_Lij_real = vle16_v_i16m1(&L_real[GET_IDX(ptr_row, ptr_col, offset, nr, n_subcarrier)], vl);
            p0_real = vwmacc_vv_i32m2(p0_real, t_Lij_real, t_uk_real, vl);

            t_uk_imag = vle16_v_i16m1(&u_imag[GET_IDX(ptr_col, 0, offset, nr, n_subcarrier)], vl);
            p0_imag = vwmacc_vv_i32m2(p0_imag, t_Lij_real,t_uk_imag, vl);

            t_Lij_imag = vle16_v_i16m1(&L_imag[GET_IDX(ptr_row, ptr_col, offset, nr, n_subcarrier)], vl);
            p0_real = vsub_vv_i32m2(p0_real, vwmul_vv_i32m2(t_Lij_imag, t_uk_imag, vl), vl);
            p0_imag = vwmacc_vv_i32m2(p0_imag, t_Lij_imag, t_uk_real, vl);


        }
        
        t_zk_real = vsub_vv_i32m2(t_zk_real, p0_real, vl);
        t_zk_real = vsub_vv_i32m2(t_zk_real, t_fr_Lii_real, vl);

        vse32_v_f32m1(&u_real[GET_IDX(ptr_row, 0, 0, 1, vl)], t_zk_real, vl);

        t_zk_imag = vfsub_vv_f32m1(t_zk_imag, p0_imag, vl);
        t_zk_imag = vfmul_vv_f32m1(t_zk_imag, t_fr_Lii_real, vl);
        vse32_v_f32m1(&u_imag[GET_IDX(ptr_row, 0, 0, 1, vl)], t_zk_imag, vl);

    }





}