// Copyright 2025 versionodafone Sponsored Chair of Communication System, TU Dresden and Synopsys

// Author: Zexu Tan:zexu.tan@tu-dresden.de, versionodafone Chair, TU Dresden


#include <riscv_vector.h>
#include "arith_lib.h"

/*

version 1: V1*

*/


#define version 1

#if version == 1


void v_compute_gram_fp32(
  size_t nt, 
  size_t nr, 
  size_t vl, 
  float sigma_n2, 
  float *H_real,
  float *H_imag,
  float *Gram_real, 
  float *Gram_imag) {

    //gram_real
    vfloat32m1_t G_00_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_10_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_20_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_30_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_40_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_50_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_60_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_70_real = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_11_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_21_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_30_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_40_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_50_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_60_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_70_real = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_22_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_32_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_42_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_52_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_62_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_72_real = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_22_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_32_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_42_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_52_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_62_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_72_real = vfmv_v_f_f32m1(0.0f, vl);

    // + sigma_n
    G_00_real = vfadd_vf_f32m1(G_00_real, sigma_n2, vl);
    G_11_real = vfadd_vf_f32m1(G_11_real, sigma_n2, vl);
    G_22_real = vfadd_vf_f32m1(G_22_real, sigma_n2, vl);
    G_33_real = vfadd_vf_f32m1(G_33_real, sigma_n2, vl);


    for (int ptr_row = 0; ptr_row < 4; ptr_row++){
      //G00
      vfloat32m1_t t0_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 0, 0, nr, vl)], vl);
      vfloat32m1_t t0_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 0, 0, nr, vl)], vl);
      vfloat32m1_t t0_imag_neg = vfneg_v_f32m1(t0_imag, vl);
      vcfmac_vv_f32m1_sim(t0_real, t0_imag_neg, t0_real, t0_imag, G_00_real, G_00_imag, &G_00_real, &G_00_imag, vl);

      //G10
      vfloat32m1_t t1_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 1, 0, nr, vl)], vl);
      vfloat32m1_t t1_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 1, 0, nr, vl)], vl);
      vfloat32m1_t t1_imag_neg = vfneg_v_f32m1(t1_imag, vl);
      vcfmac_vv_f32m1_sim(t1_real, t1_imag_neg, t0_real, t0_imag, G_10_real, G_10_imag, &G_10_real, &G_10_imag, vl);


      //G20
      vfloat32m1_t t2_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 2, 0, nr, vl)], vl);
      vfloat32m1_t t2_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 2, 0, nr, vl)], vl);
      vfloat32m1_t t2_imag_neg = vfneg_v_f32m1(t2_imag, vl);
      vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t0_real, t0_imag, G_20_real, G_20_imag, &G_20_real, &G_20_imag, vl);

      //G30
      vfloat32m1_t t3_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 3, 0, nr, vl)], vl);
      vfloat32m1_t t3_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 3, 0, nr, vl)], vl);
      vfloat32m1_t t3_imag_neg = vfneg_v_f32m1(t3_imag, vl);
      vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t0_real, t0_imag, G_30_real, G_30_imag, &G_30_real, &G_30_imag, vl);

      //G11
      vcfmac_vv_f32m1_sim(t1_real, t1_imag_neg, t1_real, t1_imag, G_11_real, G_11_imag, &G_11_real, &G_11_imag, vl);
      //G21
      vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t1_real, t1_imag, G_21_real, G_21_imag, &G_21_real, &G_21_imag, vl);
      //G31
      vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t1_real, t1_imag, G_31_real, G_31_imag, &G_31_real, &G_31_imag, vl);
      //G22
      vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t2_real, t2_imag, G_22_real, G_22_imag, &G_22_real, &G_22_imag, vl);
      //G32
      vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t2_real, t2_imag, G_32_real, G_32_imag, &G_32_real, &G_32_imag, vl);
      //G33
      vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t3_real, t3_imag, G_33_real, G_33_imag, &G_33_real, &G_33_imag, vl);

    }
    //gram_real
    vse32_v_f32m1(&Gram_real[GET_IDX(0,0,0,nr,vl)], G_00_real, vl);
    vse32_v_f32m1(&Gram_real[GET_IDX(1,0,0,nr,vl)], G_10_real, vl);
    vse32_v_f32m1(&Gram_real[GET_IDX(2,0,0,nr,vl)], G_20_real, vl);
    vse32_v_f32m1(&Gram_real[GET_IDX(3,0,0,nr,vl)], G_30_real, vl);

    vse32_v_f32m1(&Gram_real[GET_IDX(1,1,0,nr,vl)], G_11_real, vl);
    vse32_v_f32m1(&Gram_real[GET_IDX(2,1,0,nr,vl)], G_21_real, vl);
    vse32_v_f32m1(&Gram_real[GET_IDX(3,1,0,nr,vl)], G_31_real, vl);

    vse32_v_f32m1(&Gram_real[GET_IDX(2,2,0,nr,vl)], G_22_real, vl);
    vse32_v_f32m1(&Gram_real[GET_IDX(3,2,0,nr,vl)], G_32_real, vl);

    vse32_v_f32m1(&Gram_real[GET_IDX(3,3,0,nr,vl)], G_33_real, vl);

    //gram_imag
    vse32_v_f32m1(&Gram_imag[GET_IDX(0,0,0,nr,vl)], G_00_imag, vl);
    vse32_v_f32m1(&Gram_imag[GET_IDX(1,0,0,nr,vl)], G_10_imag, vl);
    vse32_v_f32m1(&Gram_imag[GET_IDX(2,0,0,nr,vl)], G_20_imag, vl);
    vse32_v_f32m1(&Gram_imag[GET_IDX(3,0,0,nr,vl)], G_30_imag, vl);

    vse32_v_f32m1(&Gram_imag[GET_IDX(1,1,0,nr,vl)], G_11_imag, vl);
    vse32_v_f32m1(&Gram_imag[GET_IDX(2,1,0,nr,vl)], G_21_imag, vl);
    vse32_v_f32m1(&Gram_imag[GET_IDX(3,1,0,nr,vl)], G_31_imag, vl);

    vse32_v_f32m1(&Gram_imag[GET_IDX(2,2,0,nr,vl)], G_22_imag, vl);
    vse32_v_f32m1(&Gram_imag[GET_IDX(3,2,0,nr,vl)], G_32_imag, vl);

    vse32_v_f32m1(&Gram_imag[GET_IDX(3,3,0,nr,vl)], G_33_imag, vl);

}

#else


void v_compute_gram_fp32(
  size_t nt, 
  size_t nr, 
  size_t vl, 
  float sigma_n2, 
  float *H_real,
  float *H_imag,
  float *Gram_real, 
  float *Gram_imag) {
    for (int ptr_row = 0; ptr_row < 4; ptr_row++){
      //G00
      vfloat32m1_t t0_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 0, 0, nr, vl)], vl);
      vfloat32m1_t t0_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 0, 0, nr, vl)], vl);
      vfloat32m1_t t0_imag_neg = vfneg_v_f32m1(t0_imag, vl);

      vfloat32m1_t G_00_real = vle32_v_f32m1(&Gram_real[GET_IDX(0, 0, 0, nr, vl)], vl);
      if (ptr_row == 0){
        G_00_real = vfadd_vf_f32m1(G_00_real, sigma_n2, vl);
      }
      vfloat32m1_t G_00_imag = vle32_v_f32m1(&Gram_imag[GET_IDX(0, 0, 0, nr, vl)], vl);
      
      vcfmac_vv_f32m1_sim(t0_real, t0_imag_neg, t0_real, t0_imag, G_00_real, G_00_imag, &G_00_real, &G_00_imag, vl);

      vse32_v_f32m1(&Gram_real[GET_IDX(0,0,0,nr,vl)], G_00_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(0,0,0,nr,vl)], G_00_imag, vl);


      //G10
      vfloat32m1_t t1_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 1, 0, nr, vl)], vl);
      vfloat32m1_t t1_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 1, 0, nr, vl)], vl);
      vfloat32m1_t t1_imag_neg = vfneg_v_f32m1(t1_imag, vl);

      vfloat32m1_t G_10_real = vle32_v_f32m1(&Gram_real[GET_IDX(1, 0, 0, nr, vl)], vl);
      vfloat32m1_t G_10_imag = vle32_v_f32m1(&Gram_imag[GET_IDX(1, 0, 0, nr, vl)], vl);

      vcfmac_vv_f32m1_sim(t1_real, t1_imag_neg, t0_real, t0_imag, G_10_real, G_10_imag, &G_10_real, &G_10_imag, vl);

      vse32_v_f32m1(&Gram_real[GET_IDX(1,0,0,nr,vl)], G_10_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(1,0,0,nr,vl)], G_10_imag, vl);


      //G20
      vfloat32m1_t t2_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 2, 0, nr, vl)], vl);
      vfloat32m1_t t2_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 2, 0, nr, vl)], vl);
      vfloat32m1_t t2_imag_neg = vfneg_v_f32m1(t2_imag, vl);

      vfloat32m1_t G_20_real = vle32_v_f32m1(&Gram_real[GET_IDX(2, 0, 0, nr, vl)], vl);
      vfloat32m1_t G_20_imag = vle32_v_f32m1(&Gram_imag[GET_IDX(2, 0, 0, nr, vl)], vl);
      vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t0_real, t0_imag, G_20_real, G_20_imag, &G_20_real, &G_20_imag, vl);

      vse32_v_f32m1(&Gram_real[GET_IDX(2,0,0,nr,vl)], G_20_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(2,0,0,nr,vl)], G_20_imag, vl);


      //G30
      vfloat32m1_t t3_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 3, 0, nr, vl)], vl);
      vfloat32m1_t t3_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 3, 0, nr, vl)], vl);
      vfloat32m1_t t3_imag_neg = vfneg_v_f32m1(t3_imag, vl);

      vfloat32m1_t G_30_real = vle32_v_f32m1(&Gram_real[GET_IDX(3, 0, 0, nr, vl)], vl);
      vfloat32m1_t G_30_imag = vle32_v_f32m1(&Gram_imag[GET_IDX(3, 0, 0, nr, vl)], vl);
      vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t0_real, t0_imag, G_30_real, G_30_imag, &G_30_real, &G_30_imag, vl);

      vse32_v_f32m1(&Gram_real[GET_IDX(3,0,0,nr,vl)], G_30_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(3,0,0,nr,vl)], G_30_imag, vl);


      //G11
      vfloat32m1_t G_11_real = vle32_v_f32m1(&Gram_real[GET_IDX(1, 1, 0, nr, vl)], vl);
      if (ptr_row == 0){
        G_11_real = vfadd_vf_f32m1(G_11_real, sigma_n2, vl);
      }
      vfloat32m1_t G_11_imag = vle32_v_f32m1(&Gram_imag[GET_IDX(1, 1, 0, nr, vl)], vl);
      vcfmac_vv_f32m1_sim(t1_real, t1_imag_neg, t1_real, t1_imag, G_11_real, G_11_imag, &G_11_real, &G_11_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(1,1,0,nr,vl)], G_11_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(1,1,0,nr,vl)], G_11_imag, vl);

      //G21
      vfloat32m1_t G_21_real = vle32_v_f32m1(&Gram_real[GET_IDX(2, 1, 0, nr, vl)], vl);
      vfloat32m1_t G_21_imag = vle32_v_f32m1(&Gram_imag[GET_IDX(2, 1, 0, nr, vl)], vl);
      vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t1_real, t1_imag, G_21_real, G_21_imag, &G_21_real, &G_21_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(2,1,0,nr,vl)], G_21_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(2,1,0,nr,vl)], G_21_imag, vl);

      //G31
      vfloat32m1_t G_31_real = vle32_v_f32m1(&Gram_real[GET_IDX(3, 1, 0, nr, vl)], vl);
      vfloat32m1_t G_31_imag = vle32_v_f32m1(&Gram_imag[GET_IDX(3, 1, 0, nr, vl)], vl);
      vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t1_real, t1_imag, G_31_real, G_31_imag, &G_31_real, &G_31_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(3,1,0,nr,vl)], G_31_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(3,1,0,nr,vl)], G_31_imag, vl);

      //G22
      vfloat32m1_t G_22_real = vle32_v_f32m1(&Gram_real[GET_IDX(2, 2, 0, nr, vl)], vl);
      if (ptr_row == 0){
        G_22_real = vfadd_vf_f32m1(G_22_real, sigma_n2, vl);
      }
      vfloat32m1_t G_22_imag = vle32_v_f32m1(&Gram_imag[GET_IDX(2, 2, 0, nr, vl)], vl);
      vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t2_real, t2_imag, G_22_real, G_22_imag, &G_22_real, &G_22_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(2,2,0,nr,vl)], G_22_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(2,2,0,nr,vl)], G_22_imag, vl);

      //G32
      vfloat32m1_t G_32_real = vle32_v_f32m1(&Gram_real[GET_IDX(3, 2, 0, nr, vl)], vl);
      vfloat32m1_t G_32_imag = vle32_v_f32m1(&Gram_imag[GET_IDX(3, 2, 0, nr, vl)], vl);
      vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t2_real, t2_imag, G_32_real, G_32_imag, &G_32_real, &G_32_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(3,2,0,nr,vl)], G_22_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(3,2,0,nr,vl)], G_22_imag, vl);

      //G33
      vfloat32m1_t G_33_real = vle32_v_f32m1(&Gram_real[GET_IDX(3, 3, 0, nr, vl)], vl);
      if (ptr_row == 0){
        G_33_real = vfadd_vf_f32m1(G_33_real, sigma_n2, vl);
      }
      vfloat32m1_t G_33_imag = vle32_v_f32m1(&Gram_imag[GET_IDX(3, 3, 0, nr, vl)], vl);
      vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t3_real, t3_imag, G_33_real, G_33_imag, &G_33_real, &G_33_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(3,3,0,nr,vl)], G_33_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(3,3,0,nr,vl)], G_33_imag, vl);

    }
}



#endif

































