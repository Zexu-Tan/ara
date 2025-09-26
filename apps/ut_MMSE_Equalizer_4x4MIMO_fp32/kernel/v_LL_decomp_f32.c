#include <riscv_vector.h>
#include "arith_lib.h"


void v_LL_decomp_f32(  
  size_t nt, 
  size_t nr, 
  size_t vl,
  float *Gram_real,
  float *Gram_imag,
  float *L_real,
  float *L_imag
){
  //L_00_real
  vfloat32m1_t L00_real = vle32_v_f32m1(&Gram_real[GET_IDX(0, 0, 0, nr, vl)],vl);
  L00_real = vfrsqrt7_v_f32m1(L00_real, vl);
  vse32_v_f32m1(&L_real[GET_IDX(0, 0, 0, nr, vl)], L00_real, vl);
  
  //L_10
  vfloat32m1_t L10_real = vle32_v_f32m1(&Gram_real[GET_IDX(1, 0, 0, nr, vl)],vl);
  vfloat32m1_t L10_imag = vle32_v_f32m1(&Gram_imag[GET_IDX(1, 0, 0, nr, vl)],vl);
  L10_real = vfmul_vv_f32m1(L10_real, L00_real, vl);
  L10_imag = vfmul_vv_f32m1(L10_imag, L00_real, vl);
  vse32_v_f32m1(&L_real[GET_IDX(1, 0, 0, nr, vl)], L10_real, vl);
  vse32_v_f32m1(&L_imag[GET_IDX(1, 0, 0, nr, vl)], L10_imag, vl);
  L10_imag = vfneg_v_f32m1(L10_imag, vl);

  //L_20
  vfloat32m1_t L20_real = vle32_v_f32m1(&Gram_real[GET_IDX(2, 0, 0, nr, vl)],vl);
  vfloat32m1_t L20_imag = vle32_v_f32m1(&Gram_imag[GET_IDX(2, 0, 0, nr, vl)],vl);
  L20_real = vfmul_vv_f32m1(L20_real, L00_real, vl);
  L20_imag = vfmul_vv_f32m1(L20_imag, L00_real, vl);
  vse32_v_f32m1(&L_real[GET_IDX(2, 0, 0, nr, vl)], L20_real, vl);
  vse32_v_f32m1(&L_imag[GET_IDX(2, 0, 0, nr, vl)], L20_imag, vl);

  //L_30
  vfloat32m1_t L30_real = vle32_v_f32m1(&Gram_real[GET_IDX(3, 0, 0, nr, vl)],vl);
  vfloat32m1_t L30_imag = vle32_v_f32m1(&Gram_imag[GET_IDX(3, 0, 0, nr, vl)],vl);
  L30_real = vfmul_vv_f32m1(L30_real, L00_real, vl);
  L30_imag = vfmul_vv_f32m1(L30_imag, L00_real, vl);
  vse32_v_f32m1(&L_real[GET_IDX(3, 0, 0, nr, vl)], L30_real, vl);
  vse32_v_f32m1(&L_imag[GET_IDX(3, 0, 0, nr, vl)], L30_imag, vl);
  
  
  //L_11_real
  vfloat32m1_t L11_real = vle32_v_f32m1(&Gram_real[GET_IDX(1, 1, 0, nr, vl)],vl);
  L11_real = vfnmsac_vv_f32m1(L11_real, L10_real, L10_real, vl);
  L11_real = vfnmsac_vv_f32m1(L11_real, L10_imag, L10_imag, vl);
  L11_real = vfrsqrt7_v_f32m1(L11_real, vl);
  vse32_v_f32m1(&L_real[GET_IDX(1, 1, 0, nr, vl)], L11_real, vl);

  
  //L_21
  
  vfloat32m1_t L21_real = vle32_v_f32m1(&Gram_real[GET_IDX(2, 1, 0, nr, vl)],vl);
  vfloat32m1_t L21_imag = vle32_v_f32m1(&Gram_imag[GET_IDX(2, 1, 0, nr, vl)],vl);
  
  L21_real = vfnmsac_vv_f32m1(L21_real, L20_real, L10_real, vl);
  L21_real = vfmacc_vv_f32m1(L21_real, L20_imag, L10_imag, vl);
  L21_imag = vfnmsac_vv_f32m1(L21_imag, L20_real, L10_imag, vl);
  L21_imag = vfnmsac_vv_f32m1(L21_imag, L20_imag, L10_real, vl);
  L21_real = vfmul_vv_f32m1(L21_real, L11_real, vl);
  L21_imag = vfmul_vv_f32m1(L21_imag, L11_real, vl);
  vse32_v_f32m1(&L_real[GET_IDX(2, 1, 0, nr, vl)], L21_real, vl);
  vse32_v_f32m1(&L_imag[GET_IDX(2, 1, 0, nr, vl)], L21_imag, vl);


  //L_31
  vfloat32m1_t L31_real = vle32_v_f32m1(&Gram_real[GET_IDX(3, 1, 0, nr, vl)],vl);
  vfloat32m1_t L31_imag = vle32_v_f32m1(&Gram_imag[GET_IDX(3, 1, 0, nr, vl)],vl);
  //vcfnmsac_vv_f32m1_sim(L30_real, L30_imag, L10_real, L10_imag, L31_real, L31_imag, &L31_real, &L31_imag, vl);
  
  L31_real = vfnmsac_vv_f32m1(L31_real, L30_real, L10_real, vl);
  L31_real = vfmacc_vv_f32m1(L31_real, L30_imag, L10_imag, vl);

  L31_imag = vfnmsac_vv_f32m1(L31_imag, L30_real, L10_imag, vl);
  L31_imag = vfnmsac_vv_f32m1(L31_imag, L30_imag, L10_real, vl);

  L31_real = vfmul_vv_f32m1(L31_real, L11_real, vl);
  L31_imag = vfmul_vv_f32m1(L31_imag, L11_real, vl);
  vse32_v_f32m1(&L_real[GET_IDX(3, 1, 0, nr, vl)], L31_real, vl);
  vse32_v_f32m1(&L_imag[GET_IDX(3, 1, 0, nr, vl)], L31_imag, vl);


  L20_imag = vfneg_v_f32m1(L20_imag, vl);
  L21_imag = vfneg_v_f32m1(L21_imag, vl);

  //L_22_real
  vfloat32m1_t L22_real = vle32_v_f32m1(&Gram_real[GET_IDX(2, 2, 0, nr, vl)],vl);
  L22_real = vfnmsac_vv_f32m1(L22_real, L20_real, L20_real, vl);
  L22_real = vfnmsac_vv_f32m1(L22_real, L20_imag, L20_imag, vl);
  L22_real = vfnmsac_vv_f32m1(L22_real, L21_real, L21_real, vl);
  L22_real = vfnmsac_vv_f32m1(L22_real, L21_imag, L21_imag, vl);
  L22_real = vfrsqrt7_v_f32m1(L22_real, vl);
  vse32_v_f32m1(&L_real[GET_IDX(2, 2, 0, nr, vl)], L22_real, vl);

  //L_32
  vfloat32m1_t L32_real = vle32_v_f32m1(&Gram_real[GET_IDX(3, 2, 0, nr, vl)],vl);
  vfloat32m1_t L32_imag = vle32_v_f32m1(&Gram_imag[GET_IDX(3, 2, 0, nr, vl)],vl);
  vcfnmsac_vv_f32m1_sim(L30_real, L30_imag, L20_real, L20_imag, L32_real, L32_imag, &L32_real, &L32_imag, vl);
  vcfnmsac_vv_f32m1_sim(L31_real, L31_imag, L21_real, L21_imag, L32_real, L32_imag, &L32_real, &L32_imag, vl);
  L32_real = vfmul_vv_f32m1(L32_real, L22_real, vl);
  L32_imag = vfmul_vv_f32m1(L32_imag, L22_real, vl);
  vse32_v_f32m1(&L_real[GET_IDX(3, 2, 0, nr, vl)], L32_real, vl);
  vse32_v_f32m1(&L_imag[GET_IDX(3, 2, 0, nr, vl)], L32_imag, vl);

  //L_33_real
  vfloat32m1_t L33_real = vle32_v_f32m1(&Gram_real[GET_IDX(3, 3, 0, nr, vl)],vl); 
  L33_real = vfnmsac_vv_f32m1(L33_real, L30_real, L30_real, vl);
  L33_real = vfnmsac_vv_f32m1(L33_real, L30_imag, L30_imag, vl);
  L33_real = vfnmsac_vv_f32m1(L33_real, L31_real, L31_real, vl);
  L33_real = vfnmsac_vv_f32m1(L33_real, L31_imag, L31_imag, vl);
  L33_real = vfnmsac_vv_f32m1(L33_real, L32_real, L32_real, vl);
  L33_real = vfnmsac_vv_f32m1(L33_real, L32_imag, L32_imag, vl);
  L33_real = vfrsqrt7_v_f32m1(L33_real, vl);
  vse32_v_f32m1(&L_real[GET_IDX(3, 3, 0, nr, vl)], L33_real, vl);

}

