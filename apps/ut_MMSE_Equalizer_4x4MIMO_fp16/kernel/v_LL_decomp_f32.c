#include <riscv_vector.h>
#include "arith_lib.h"



void v_LL_decomp_f32(size_t nt, size_t nr, size_t vl, float Gram_real[nt][nr][vl], float Gram_imag[nt][nr][vl], float L_real[nt][nr][vl], float L_imag[nt][nr][vl]) {
  vfloat32m1_t A00_real, frsqrt_L00, L00_real;
  vfloat32m1_t L10_real, L10_imag, L20_real, L20_imag, L30_real, L30_imag;
  
  A00_real = vle32_v_f32m1(&Gram_real[0][0][0],vl);
  frsqrt_L00 = vfrsqrt7_v_f32m1(A00_real, vl);
  L00_real  = vfmul_vv_f32m1(A00_real, frsqrt_L00,vl);

  vse32_v_f32m1(&L_real[0][0][0], L00_real, vl);
  

  //L10
  L10_real = vle32_v_f32m1(&Gram_real[1][0][0],vl);
  L10_imag = vle32_v_f32m1(&Gram_imag[1][0][0],vl);

  L10_real = vfmul_vv_f32m1(L10_real, frsqrt_L00,vl);
  L10_imag = vfmul_vv_f32m1(L10_imag, frsqrt_L00,vl); 

  vse32_v_f32m1(&L_real[1][0][0], L10_real, vl);
  vse32_v_f32m1(&L_imag[1][0][0], L10_imag, vl);

  //L20
  L20_real = vle32_v_f32m1(&Gram_real[2][0][0],vl);
  L20_imag = vle32_v_f32m1(&Gram_imag[2][0][0],vl);

  L20_real = vfmul_vv_f32m1(L20_real, frsqrt_L00,vl);
  L20_imag = vfmul_vv_f32m1(L20_imag, frsqrt_L00,vl);

  vse32_v_f32m1(&L_real[2][0][0], L20_real, vl);
  vse32_v_f32m1(&L_imag[2][0][0], L20_imag, vl); 

  //L30
  L30_real = vle32_v_f32m1(&Gram_real[3][0][0],vl);
  L30_imag = vle32_v_f32m1(&Gram_imag[3][0][0],vl);

  L30_real = vfmul_vv_f32m1(L30_real, frsqrt_L00,vl);
  L30_imag = vfmul_vv_f32m1(L30_imag, frsqrt_L00,vl);

  vse32_v_f32m1(&L_real[3][0][0], L30_real, vl);
  vse32_v_f32m1(&L_imag[3][0][0], L30_imag, vl);

  //L11
  vfloat32m1_t A11_real, p_sqr_L10, frsqrt_L11;
  A11_real = vle32_v_f32m1(&Gram_real[1][1][0],vl);

  //conjugate L10
  L10_imag = vfneg_v_f32m1(L10_imag, vl);

  vcsqrdmag_vv_f32m1_sim(L10_real, L10_imag, &p_sqr_L10, vl);

  A11_real = vfsub_vv_f32m1(A11_real, p_sqr_L10, vl);
  frsqrt_L11 = vfrsqrt7_v_f32m1(A11_real, vl);
  A11_real = vfmul_vv_f32m1(A11_real, frsqrt_L11, vl);

  vse32_v_f32m1(&L_real[1][1][0], A11_real, vl);

  //L21
  vfloat32m1_t L21_real, L21_imag, p0_L21_real, p0_L21_imag;
  L21_real = vle32_v_f32m1(&Gram_real[2][1][0],vl);
  L21_imag = vle32_v_f32m1(&Gram_imag[2][1][0],vl);

  vcfmul_vv_f32m1_sim(L20_real, L20_imag, L10_real, L10_imag, &p0_L21_real, &p0_L21_imag, vl);

  L21_real = vfsub_vv_f32m1(L21_real, p0_L21_real, vl);
  L21_imag = vfsub_vv_f32m1(L21_imag, p0_L21_imag, vl);

  L21_real = vfmul_vv_f32m1(L21_real, frsqrt_L11, vl);
  L21_imag = vfmul_vv_f32m1(L21_imag, frsqrt_L11, vl);

  vse32_v_f32m1(&L_real[2][1][0], L21_real, vl);
  vse32_v_f32m1(&L_imag[2][1][0], L21_imag, vl);


  //L31

  vfloat32m1_t L31_real, L31_imag, p0_L31_real, p0_L31_imag;
  L31_real = vle32_v_f32m1(&Gram_real[3][1][0],vl);
  L31_imag = vle32_v_f32m1(&Gram_imag[3][1][0],vl);

  vcfmul_vv_f32m1_sim(L30_real, L30_imag, L10_real, L10_imag, &p0_L31_real, &p0_L31_imag, vl);

  L31_real = vfsub_vv_f32m1(L31_real, p0_L31_real, vl);
  L31_imag = vfsub_vv_f32m1(L31_imag, p0_L31_imag, vl);

  L31_real = vfmul_vv_f32m1(L31_real, frsqrt_L11, vl);
  L31_imag = vfmul_vv_f32m1(L31_imag, frsqrt_L11, vl);

  vse32_v_f32m1(&L_real[3][1][0], L31_real, vl);
  vse32_v_f32m1(&L_imag[3][1][0], L31_imag, vl);


  //L22
  vfloat32m1_t L22_real, p0_L22_real, p1_L22_real, frsqrt_L22;
  L22_real = vle32_v_f32m1(&Gram_real[2][2][0],vl);

  //conjugate L20, L21
  L20_imag = vfneg_v_f32m1(L20_imag, vl);
  L21_imag = vfneg_v_f32m1(L21_imag, vl);

  vcsqrdmag_vv_f32m1_sim(L20_real, L20_imag, &p0_L22_real, vl);
  L22_real = vfsub_vv_f32m1(L22_real, p0_L22_real, vl);

  vcsqrdmag_vv_f32m1_sim(L21_real, L21_imag, &p1_L22_real, vl);

  L22_real = vfsub_vv_f32m1(L22_real, p1_L22_real, vl);

  frsqrt_L22 = vfrsqrt7_v_f32m1(L22_real, vl);

  L22_real = vfmul_vv_f32m1(L22_real, frsqrt_L22, vl);

  vse32_v_f32m1(&L_real[2][2][0], L22_real, vl);

  //L32
  vfloat32m1_t L32_real, L32_imag, p0_L32_real, p0_L32_imag, p1_L32_real, p1_L32_imag;
  L32_real = vle32_v_f32m1(&Gram_real[3][2][0],vl);
  L32_imag = vle32_v_f32m1(&Gram_imag[3][2][0],vl);

  vcfmul_vv_f32m1_sim(L30_real, L30_imag, L20_real, L20_imag, &p0_L32_real, &p0_L32_imag, vl);


  L32_real = vfsub_vv_f32m1(L32_real, p0_L32_real, vl);
  L32_imag = vfsub_vv_f32m1(L32_imag, p0_L32_imag, vl);

  vcfmul_vv_f32m1_sim(L31_real, L31_imag, L21_real, L21_imag, &p1_L32_real, &p1_L32_imag, vl);

  L32_real = vfsub_vv_f32m1(L32_real, p1_L32_real, vl);
  L32_imag = vfsub_vv_f32m1(L32_imag, p1_L32_imag, vl);
  
  
  L32_real = vfmul_vv_f32m1(L32_real, frsqrt_L22, vl);
  L32_imag = vfmul_vv_f32m1(L32_imag, frsqrt_L22, vl);

  vse32_v_f32m1(&L_real[3][2][0], L32_real, vl);
  vse32_v_f32m1(&L_imag[3][2][0], L32_imag, vl);


  //L33
  vfloat32m1_t A33_real, L33_real;
  A33_real = vle32_v_f32m1(&Gram_real[3][3][0],vl);

  vfloat32m1_t p3_real, p4_real, p5_real;
  vcsqrdmag_vv_f32m1_sim(L30_real, L30_imag, &p3_real, vl);
  A33_real = vfsub_vv_f32m1(A33_real, p3_real, vl);

  vcsqrdmag_vv_f32m1_sim(L31_real, L31_imag, &p4_real, vl);
  A33_real = vfsub_vv_f32m1(A33_real, p4_real, vl);

  vcsqrdmag_vv_f32m1_sim(L32_real, L32_imag, &p5_real, vl);
  A33_real = vfsub_vv_f32m1(A33_real, p5_real, vl);

  vfloat32m1_t frsqrt_L33;
  frsqrt_L33 = vfrsqrt7_v_f32m1(A33_real, vl);

  L33_real = vfmul_vv_f32m1(A33_real, frsqrt_L33, vl);

  vse32_v_f32m1(&L_real[3][3][0], L33_real, vl);
 
}