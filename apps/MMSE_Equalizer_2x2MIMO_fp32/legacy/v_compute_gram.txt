// Copyright 2025 Vodafone Sponsored Chair of Communication System, TU Dresden and Synopsys

// Author: Zexu Tan:zexu.tan@tu-dresden.de, Vodafone Chair, TU Dresden


#include <riscv_vector.h>
#include "arith_lib.h"


#define CASE 3

//case 0 (LEGACY)v1
//CASE 2 (LEGACY)v2
//case 3 (LEGACY)v2_interleaved
//case 4 (LEGACY)v1_interleaved
//case 5 (LEGACY)vl blocked
//case 6 (LEGACY)v1_perfect
//case 7 undertest



#if CASE == 0



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

    vfloat32m1_t G_11_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_21_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_31_real = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_22_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_32_real = vfmv_v_f_f32m1(0.0f, vl);
    
    vfloat32m1_t G_33_real = vfmv_v_f_f32m1(0.0f, vl);

    //gram_imag
    vfloat32m1_t G_00_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_10_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_20_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_30_imag = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_11_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_21_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_31_imag = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_22_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_32_imag = vfmv_v_f_f32m1(0.0f, vl);
    
    vfloat32m1_t G_33_imag = vfmv_v_f_f32m1(0.0f, vl);

    // + sigma_n
    G_00_real = vfadd_vf_f32m1(G_00_real, sigma_n2, vl);
    G_11_real = vfadd_vf_f32m1(G_11_real, sigma_n2, vl);
    G_22_real = vfadd_vf_f32m1(G_22_real, sigma_n2, vl);
    G_33_real = vfadd_vf_f32m1(G_33_real, sigma_n2, vl);

    vfloat32m1_t t0_real, t0_imag, t0_imag_neg;
    vfloat32m1_t t1_real, t1_imag, t1_imag_neg;
    vfloat32m1_t t2_real, t2_imag, t2_imag_neg;
    vfloat32m1_t t3_real, t3_imag, t3_imag_neg;


    for (int ptr_row = 0; ptr_row < 4; ptr_row++){
      //G00
      t0_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 0, 0, nr, vl)], vl);
      t0_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 0, 0, nr, vl)], vl);
      t0_imag_neg = vfneg_v_f32m1(t0_imag, vl);
      vcfmac_vv_f32m1_sim(t0_real, t0_imag_neg, t0_real, t0_imag, G_00_real, G_00_imag, &G_00_real, &G_00_imag, vl);

      //G10
      t1_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 1, 0, nr, vl)], vl);
      t1_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 1, 0, nr, vl)], vl);
      t1_imag_neg = vfneg_v_f32m1(t1_imag, vl);
      vcfmac_vv_f32m1_sim(t1_real, t1_imag_neg, t0_real, t0_imag, G_10_real, G_10_imag, &G_10_real, &G_10_imag, vl);


      //G20
      t2_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 2, 0, nr, vl)], vl);
      t2_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 2, 0, nr, vl)], vl);
      t2_imag_neg = vfneg_v_f32m1(t2_imag, vl);
      vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t0_real, t0_imag, G_20_real, G_20_imag, &G_20_real, &G_20_imag, vl);

      //G30
      t3_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 3, 0, nr, vl)], vl);
      t3_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 3, 0, nr, vl)], vl);
      t3_imag_neg = vfneg_v_f32m1(t3_imag, vl);
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



#elif CASE == 1 


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

    vfloat32m1_t G_11_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_21_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_31_real = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_22_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_32_real = vfmv_v_f_f32m1(0.0f, vl);
    
    vfloat32m1_t G_33_real = vfmv_v_f_f32m1(0.0f, vl);

    //gram_imag
    vfloat32m1_t G_00_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_10_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_20_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_30_imag = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_11_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_21_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_31_imag = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_22_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_32_imag = vfmv_v_f_f32m1(0.0f, vl);
    
    vfloat32m1_t G_33_imag = vfmv_v_f_f32m1(0.0f, vl);

    // + sigma_n
    G_00_real = vfadd_vf_f32m1(G_00_real, sigma_n2, vl);
    G_11_real = vfadd_vf_f32m1(G_11_real, sigma_n2, vl);
    G_22_real = vfadd_vf_f32m1(G_22_real, sigma_n2, vl);
    G_33_real = vfadd_vf_f32m1(G_33_real, sigma_n2, vl);

    vfloat32m1_t t0_real, t0_imag, t0_imag_neg;
    vfloat32m1_t t1_real, t1_imag, t1_imag_neg;
    vfloat32m1_t t2_real, t2_imag, t2_imag_neg;
    vfloat32m1_t t3_real, t3_imag, t3_imag_neg;


    for (int ptr_row = 0; ptr_row < 3; ptr_row++){
      //G00
      t0_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 0, 0, nr, vl)], vl);
      t0_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 0, 0, nr, vl)], vl);
      t0_imag_neg = vfneg_v_f32m1(t0_imag, vl);
      vcfmac_vv_f32m1_sim(t0_real, t0_imag_neg, t0_real, t0_imag, G_00_real, G_00_imag, &G_00_real, &G_00_imag, vl);

      //G10
      t1_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 1, 0, nr, vl)], vl);
      t1_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 1, 0, nr, vl)], vl);
      t1_imag_neg = vfneg_v_f32m1(t1_imag, vl);
      vcfmac_vv_f32m1_sim(t1_real, t1_imag_neg, t0_real, t0_imag, G_10_real, G_10_imag, &G_10_real, &G_10_imag, vl);


      //G20
      t2_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 2, 0, nr, vl)], vl);
      t2_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 2, 0, nr, vl)], vl);
      t2_imag_neg = vfneg_v_f32m1(t2_imag, vl);
      vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t0_real, t0_imag, G_20_real, G_20_imag, &G_20_real, &G_20_imag, vl);

      //G30
      t3_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 3, 0, nr, vl)], vl);
      t3_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 3, 0, nr, vl)], vl);
      t3_imag_neg = vfneg_v_f32m1(t3_imag, vl);
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


    t0_real = vle32_v_f32m1(&H_real[GET_IDX(3, 0, 0, nr, vl)], vl);
    t0_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 0, 0, nr, vl)], vl);
    t0_imag_neg = vfneg_v_f32m1(t0_imag, vl);
    vcfmac_vv_f32m1_sim(t0_real, t0_imag_neg, t0_real, t0_imag, G_00_real, G_00_imag, &G_00_real, &G_00_imag, vl);

    vse32_v_f32m1(&Gram_real[GET_IDX(0,0,0,nr,vl)], G_00_real, vl);
    vse32_v_f32m1(&Gram_imag[GET_IDX(0,0,0,nr,vl)], G_00_imag, vl);

    //G10
    t1_real = vle32_v_f32m1(&H_real[GET_IDX(3, 1, 0, nr, vl)], vl);
    t1_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 1, 0, nr, vl)], vl);
    t1_imag_neg = vfneg_v_f32m1(t1_imag, vl);
    vcfmac_vv_f32m1_sim(t1_real, t1_imag_neg, t0_real, t0_imag, G_10_real, G_10_imag, &G_10_real, &G_10_imag, vl);
  
    vse32_v_f32m1(&Gram_real[GET_IDX(1,0,0,nr,vl)], G_10_real, vl);
    vse32_v_f32m1(&Gram_imag[GET_IDX(1,0,0,nr,vl)], G_10_imag, vl);

    //G20
    t2_real = vle32_v_f32m1(&H_real[GET_IDX(3, 2, 0, nr, vl)], vl);
    t2_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 2, 0, nr, vl)], vl);
    t2_imag_neg = vfneg_v_f32m1(t2_imag, vl);
    vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t0_real, t0_imag, G_20_real, G_20_imag, &G_20_real, &G_20_imag, vl);

    vse32_v_f32m1(&Gram_real[GET_IDX(2,0,0,nr,vl)], G_20_real, vl);
    vse32_v_f32m1(&Gram_imag[GET_IDX(2,0,0,nr,vl)], G_20_imag, vl);



    //G30
    t3_real = vle32_v_f32m1(&H_real[GET_IDX(3, 3, 0, nr, vl)], vl);
    t3_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 3, 0, nr, vl)], vl);
    t3_imag_neg = vfneg_v_f32m1(t3_imag, vl);
    vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t0_real, t0_imag, G_30_real, G_30_imag, &G_30_real, &G_30_imag, vl);

    vse32_v_f32m1(&Gram_real[GET_IDX(3,0,0,nr,vl)], G_30_real, vl);
    vse32_v_f32m1(&Gram_imag[GET_IDX(3,0,0,nr,vl)], G_30_imag, vl);
    
    

    //G11
    vcfmac_vv_f32m1_sim(t1_real, t1_imag_neg, t1_real, t1_imag, G_11_real, G_11_imag, &G_11_real, &G_11_imag, vl);

    vse32_v_f32m1(&Gram_real[GET_IDX(1,1,0,nr,vl)], G_11_real, vl);
    vse32_v_f32m1(&Gram_imag[GET_IDX(1,1,0,nr,vl)], G_11_imag, vl);

    //G21
    vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t1_real, t1_imag, G_21_real, G_21_imag, &G_21_real, &G_21_imag, vl);

    vse32_v_f32m1(&Gram_real[GET_IDX(2,1,0,nr,vl)], G_21_real, vl);
    vse32_v_f32m1(&Gram_imag[GET_IDX(2,1,0,nr,vl)], G_21_imag, vl);

    //G31
    vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t1_real, t1_imag, G_31_real, G_31_imag, &G_31_real, &G_31_imag, vl);

    vse32_v_f32m1(&Gram_real[GET_IDX(3,1,0,nr,vl)], G_31_real, vl);
    vse32_v_f32m1(&Gram_imag[GET_IDX(3,1,0,nr,vl)], G_31_imag, vl);

    //G22
    vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t2_real, t2_imag, G_22_real, G_22_imag, &G_22_real, &G_22_imag, vl);

    vse32_v_f32m1(&Gram_real[GET_IDX(2,2,0,nr,vl)], G_22_real, vl);
    vse32_v_f32m1(&Gram_imag[GET_IDX(2,2,0,nr,vl)], G_22_imag, vl);

    //G32
    vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t2_real, t2_imag, G_32_real, G_32_imag, &G_32_real, &G_32_imag, vl);

    vse32_v_f32m1(&Gram_real[GET_IDX(3,2,0,nr,vl)], G_32_real, vl);
    vse32_v_f32m1(&Gram_imag[GET_IDX(3,2,0,nr,vl)], G_32_imag, vl);

    //G33
    vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t3_real, t3_imag, G_33_real, G_33_imag, &G_33_real, &G_33_imag, vl);

    vse32_v_f32m1(&Gram_real[GET_IDX(3,3,0,nr,vl)], G_33_real, vl);
    vse32_v_f32m1(&Gram_imag[GET_IDX(3,3,0,nr,vl)], G_33_imag, vl);

}

#elif CASE == 2

void v_compute_gram_fp32(
  size_t nt, 
  size_t nr, 
  size_t vl, 
  float sigma_n2, 
  float *H_real,
  float *H_imag,
  float *Gram_real, 
  float *Gram_imag) {
    vfloat32m1_t t0_real, t0_imag, t1_real, t1_imag, t2_real, t2_imag, t3_real, t3_imag;
    vfloat32m1_t ta_real, ta_imag, tb_real, tb_imag, tc_real, tc_imag, td_real, td_imag;
    vfloat32m1_t acc_0_real;
    vfloat32m1_t acc_0_imag;
    
    for (int j = 0; j < 4; j++){
      for (int i = j; i >-1; i--){
        acc_0_real = vfmv_v_f_f32m1(0.0f, vl);
        acc_0_imag = vfmv_v_f_f32m1(0.0f, vl);
        if (i == j){
          
          t0_real = vle32_v_f32m1(&H_real[GET_IDX(0, i, 0, nr, vl)], vl);
          ta_imag = vle32_v_f32m1(&H_imag[GET_IDX(0, i, 0, nr, vl)], vl);
          t0_imag = vfneg_v_f32m1(ta_imag, vl);
          vcfmac_vv_f32m1_sim(t0_real, t0_imag, t0_real, ta_imag, acc_0_real, acc_0_imag, &acc_0_real, &acc_0_imag, vl);
          acc_0_real = vfadd_vf_f32m1(acc_0_real, sigma_n2, vl);

          t1_real = vle32_v_f32m1(&H_real[GET_IDX(1, i, 0, nr, vl)], vl);
          tb_imag = vle32_v_f32m1(&H_imag[GET_IDX(1, i, 0, nr, vl)], vl);
          t1_imag = vfneg_v_f32m1(tb_imag, vl);
          vcfmac_vv_f32m1_sim(t1_real, t1_imag, t1_real, tb_imag, acc_0_real, acc_0_imag, &acc_0_real, &acc_0_imag, vl);
  
          t2_real = vle32_v_f32m1(&H_real[GET_IDX(2, i, 0, nr, vl)], vl);
          tc_imag = vle32_v_f32m1(&H_imag[GET_IDX(2, i, 0, nr, vl)], vl);
          t2_imag = vfneg_v_f32m1(tc_imag, vl);
          vcfmac_vv_f32m1_sim(t2_real, t2_imag, t2_real, tc_imag, acc_0_real, acc_0_imag, &acc_0_real, &acc_0_imag, vl);
  
          t3_real = vle32_v_f32m1(&H_real[GET_IDX(3, i, 0, nr, vl)], vl);
          td_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, i, 0, nr, vl)], vl);
          t3_imag = vfneg_v_f32m1(td_imag, vl);
          vcfmac_vv_f32m1_sim(t3_real, t3_imag, t3_real, td_imag, acc_0_real, acc_0_imag, &acc_0_real, &acc_0_imag, vl);
  

        }
        else{
          
          ta_real = vle32_v_f32m1(&H_real[GET_IDX(0, i, 0, nr, vl)], vl);
          ta_imag = vle32_v_f32m1(&H_imag[GET_IDX(0, i, 0, nr, vl)], vl);
          vcfmac_vv_f32m1_sim(t0_real, t0_imag, ta_real, ta_imag, acc_0_real, acc_0_imag, &acc_0_real, &acc_0_imag, vl);

          tb_real = vle32_v_f32m1(&H_real[GET_IDX(1, i, 0, nr, vl)], vl);
          tb_imag = vle32_v_f32m1(&H_imag[GET_IDX(1, i, 0, nr, vl)], vl);
          vcfmac_vv_f32m1_sim(t1_real, t1_imag, tb_real, tb_imag, acc_0_real, acc_0_imag, &acc_0_real, &acc_0_imag, vl);

          tc_real = vle32_v_f32m1(&H_real[GET_IDX(2, i, 0, nr, vl)], vl);
          tc_imag = vle32_v_f32m1(&H_imag[GET_IDX(2, i, 0, nr, vl)], vl);
          vcfmac_vv_f32m1_sim(t2_real, t2_imag, tc_real, tc_imag, acc_0_real, acc_0_imag, &acc_0_real, &acc_0_imag, vl);

          td_real = vle32_v_f32m1(&H_real[GET_IDX(3, i, 0, nr, vl)], vl);
          td_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, i, 0, nr, vl)], vl);
          vcfmac_vv_f32m1_sim(t3_real, t3_imag, td_real, td_imag, acc_0_real, acc_0_imag, &acc_0_real, &acc_0_imag, vl);
          
          
          
          
        }

        vse32_v_f32m1(&Gram_real[GET_IDX(j,i,0,nr,vl)], acc_0_real, vl);
        vse32_v_f32m1(&Gram_imag[GET_IDX(j,i,0,nr,vl)], acc_0_imag, vl);
        
      }


    }
  }

#elif CASE == 3

void v_compute_gram_fp32(
  size_t nt, 
  size_t nr, 
  size_t vl, 
  float sigma_n2, 
  float *H_real,
  float *H_imag,
  float *Gram_real, 
  float *Gram_imag) {
    vfloat32m1_t t0_real, t0_imag, t1_real, t1_imag, t2_real, t2_imag, t3_real, t3_imag;
    vfloat32m1_t ta_real, ta_imag, tb_real, tb_imag, tc_real, tc_imag, td_real, td_imag;
    vfloat32m1_t acc_0_real;
    vfloat32m1_t acc_0_imag;

    
    for (int i = 0; i < 4; i++) {
      acc_0_real = vfmv_v_f_f32m1(sigma_n2, vl);

    //G_ii
      //0a
      t0_real = vle32_v_f32m1(&H_real[GET_IDX(0, i, 0, nr, vl)], vl);
      acc_0_real = vfmacc_vv_f32m1(acc_0_real, t0_real, t0_real, vl);

      ta_imag = vle32_v_f32m1(&H_imag[GET_IDX(0, i, 0, nr, vl)], vl);
      t0_imag = vfneg_v_f32m1(ta_imag, vl);

      acc_0_real = vfnmsac_vv_f32m1(acc_0_real, t0_imag, ta_imag, vl);
      
      //1b
      t1_real = vle32_v_f32m1(&H_real[GET_IDX(1, i, 0, nr, vl)], vl);
      acc_0_real = vfmacc_vv_f32m1(acc_0_real, t1_real, t1_real, vl);

      tb_imag = vle32_v_f32m1(&H_imag[GET_IDX(1, i, 0, nr, vl)], vl);
      t1_imag = vfneg_v_f32m1(tb_imag, vl);
      acc_0_real = vfnmsac_vv_f32m1(acc_0_real, t1_imag, tb_imag, vl);


      //2c
      t2_real = vle32_v_f32m1(&H_real[GET_IDX(2, i, 0, nr, vl)], vl);
      acc_0_real = vfmacc_vv_f32m1(acc_0_real, t2_real, t2_real, vl);

      tc_imag = vle32_v_f32m1(&H_imag[GET_IDX(2, i, 0, nr, vl)], vl);
      t2_imag = vfneg_v_f32m1(tc_imag, vl);
      acc_0_real = vfnmsac_vv_f32m1(acc_0_real, t2_imag, tc_imag, vl);


          

      //3d
      t3_real = vle32_v_f32m1(&H_real[GET_IDX(3, i, 0, nr, vl)], vl);
      acc_0_real = vfmacc_vv_f32m1(acc_0_real, t3_real, t3_real, vl);

      td_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, i, 0, nr, vl)], vl);
      t3_imag = vfneg_v_f32m1(td_imag, vl);
      acc_0_real = vfnmsac_vv_f32m1(acc_0_real, t3_imag, td_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(i,i,0,nr,vl)], acc_0_real, vl);
      
      
      for (int j = i - 1; j > -1; j--) {
        acc_0_real = vfmv_v_f_f32m1(0.0f, vl);
        acc_0_imag = vfmv_v_f_f32m1(0.0f, vl);

        ta_real = vle32_v_f32m1(&H_real[GET_IDX(0, j, 0, nr, vl)], vl);
        acc_0_real = vfmacc_vv_f32m1(acc_0_real, t0_real, ta_real, vl);

        ta_imag = vle32_v_f32m1(&H_imag[GET_IDX(0, j, 0, nr, vl)], vl);

        acc_0_real = vfnmsac_vv_f32m1(acc_0_real, t0_imag, ta_imag, vl);
        acc_0_imag = vfmacc_vv_f32m1(acc_0_imag, t0_real, ta_imag, vl);
        acc_0_imag = vfmacc_vv_f32m1(acc_0_imag, t0_imag, ta_real, vl);
      
        //1b
        tb_real = vle32_v_f32m1(&H_real[GET_IDX(1, j, 0, nr, vl)], vl);
        acc_0_real = vfmacc_vv_f32m1(acc_0_real, t1_real, tb_real, vl);

        tb_imag = vle32_v_f32m1(&H_imag[GET_IDX(1, j, 0, nr, vl)], vl);
        acc_0_real = vfnmsac_vv_f32m1(acc_0_real, t1_imag, tb_imag, vl);

        acc_0_imag = vfmacc_vv_f32m1(acc_0_imag, t1_real, tb_imag, vl);
        acc_0_imag = vfmacc_vv_f32m1(acc_0_imag, t1_imag, tb_real, vl);


        //2c
        tc_real = vle32_v_f32m1(&H_real[GET_IDX(2, j, 0, nr, vl)], vl);
        acc_0_real = vfmacc_vv_f32m1(acc_0_real, t2_real, tc_real, vl);

        tc_imag = vle32_v_f32m1(&H_imag[GET_IDX(2, j, 0, nr, vl)], vl);
        acc_0_real = vfnmsac_vv_f32m1(acc_0_real, t2_imag, tc_imag, vl);

        acc_0_imag = vfmacc_vv_f32m1(acc_0_imag, t2_real, tc_imag, vl);
        acc_0_imag = vfmacc_vv_f32m1(acc_0_imag, t2_imag, tc_real, vl);

          

        //3d
        td_real = vle32_v_f32m1(&H_real[GET_IDX(3, j, 0, nr, vl)], vl);
        acc_0_real = vfmacc_vv_f32m1(acc_0_real, t3_real, td_real, vl);

        td_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, j, 0, nr, vl)], vl);
        acc_0_real = vfnmsac_vv_f32m1(acc_0_real, t3_imag, td_imag, vl);
        vse32_v_f32m1(&Gram_real[GET_IDX(i,j,0,nr,vl)], acc_0_real, vl);

        acc_0_imag = vfmacc_vv_f32m1(acc_0_imag, t3_real, td_imag, vl);
        acc_0_imag = vfmacc_vv_f32m1(acc_0_imag, t3_imag, td_real, vl);
        vse32_v_f32m1(&Gram_imag[GET_IDX(i,j,0,nr,vl)], acc_0_imag, vl); 




      }


    }



    
  }
#elif CASE == 4

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
    vfloat32m1_t G_00_real = vfmv_v_f_f32m1(sigma_n2, vl);
    vfloat32m1_t G_10_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_20_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_30_real = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_11_real = vfmv_v_f_f32m1(sigma_n2, vl);
    vfloat32m1_t G_21_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_31_real = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_22_real = vfmv_v_f_f32m1(sigma_n2, vl);
    vfloat32m1_t G_32_real = vfmv_v_f_f32m1(0.0f, vl);
    
    vfloat32m1_t G_33_real = vfmv_v_f_f32m1(sigma_n2, vl);

    //gram_imag
    vfloat32m1_t G_00_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_10_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_20_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_30_imag = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_11_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_21_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_31_imag = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_22_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_32_imag = vfmv_v_f_f32m1(0.0f, vl);
    
    vfloat32m1_t G_33_imag = vfmv_v_f_f32m1(0.0f, vl);

    // + sigma_n
    //G_00_real = vfadd_vf_f32m1(G_00_real, sigma_n2, vl);
    //G_11_real = vfadd_vf_f32m1(G_11_real, sigma_n2, vl);
    //G_22_real = vfadd_vf_f32m1(G_22_real, sigma_n2, vl);
    //G_33_real = vfadd_vf_f32m1(G_33_real, sigma_n2, vl);

    vfloat32m1_t t0_real, t0_imag, t0_imag_neg;
    vfloat32m1_t t1_real, t1_imag, t1_imag_neg;
    vfloat32m1_t t2_real, t2_imag, t2_imag_neg;
    vfloat32m1_t t3_real, t3_imag, t3_imag_neg;


    for (int ptr_row = 0; ptr_row < 3; ptr_row++){
      //G_00_interleaved

      t0_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 0, 0, nr, vl)], vl);
      G_00_real = vfmacc_vv_f32m1(G_00_real, t0_real, t0_real, vl);

      t0_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 0, 0, nr, vl)], vl);
      t0_imag_neg = vfneg_v_f32m1(t0_imag, vl);
      
      G_00_real = vfnmsac_vv_f32m1(G_00_real, t0_imag, t0_imag_neg, vl);
      G_00_imag = vfmacc_vv_f32m1(G_00_imag, t0_real, t0_imag, vl);
      G_00_imag = vfmacc_vv_f32m1(G_00_imag, t0_imag_neg, t0_real, vl);


      //G_10_interleaved
      t1_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 1, 0, nr, vl)], vl);
      G_10_real = vfmacc_vv_f32m1(G_10_real, t1_real, t0_real, vl);

      t1_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 1, 0, nr, vl)], vl);
      t1_imag_neg = vfneg_v_f32m1(t1_imag, vl);
  
      G_10_real = vfnmsac_vv_f32m1(G_10_real, t1_imag_neg, t0_imag, vl);
      G_10_imag = vfmacc_vv_f32m1(G_10_imag, t1_real, t0_imag, vl);
      G_10_imag = vfmacc_vv_f32m1(G_10_imag, t1_imag_neg, t0_real, vl);



      //G_20_interleaved
      t2_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 2, 0, nr, vl)], vl);
      G_20_real = vfmacc_vv_f32m1(G_20_real, t2_real, t0_real, vl);

      t2_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 2, 0, nr, vl)], vl);
      t2_imag_neg = vfneg_v_f32m1(t2_imag, vl);

      G_20_real = vfnmsac_vv_f32m1(G_20_real, t2_imag_neg, t0_imag, vl);
      G_20_imag = vfmacc_vv_f32m1(G_20_imag, t2_real, t0_imag, vl);
      G_20_imag = vfmacc_vv_f32m1(G_20_imag, t2_imag_neg, t0_real, vl);


      //G_30_interleaved
      t3_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 3, 0, nr, vl)], vl);
      G_30_real = vfmacc_vv_f32m1(G_30_real, t3_real, t0_real, vl);
      
      t3_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 3, 0, nr, vl)], vl);
      t3_imag_neg = vfneg_v_f32m1(t3_imag, vl);

      G_30_real = vfnmsac_vv_f32m1(G_30_real, t3_imag_neg, t0_imag, vl);
      G_30_imag = vfmacc_vv_f32m1(G_30_imag, t3_real, t0_imag, vl);
      G_30_imag = vfmacc_vv_f32m1(G_30_imag, t3_imag_neg, t0_real, vl);


      //G_11_interleaved
      //vcfmac_vv_f32m1_sim(t1_real, t1_imag_neg, t1_real, t1_imag, G_11_real, G_11_imag, &G_11_real, &G_11_imag, vl);
      
      G_11_real = vfmacc_vv_f32m1(G_11_real, t1_real, t1_real, vl);
      G_11_real = vfnmsac_vv_f32m1(G_11_real, t1_imag_neg, t1_imag, vl);
      G_11_imag = vfmacc_vv_f32m1(G_11_imag, t1_imag_neg, t1_real, vl);
      G_11_imag = vfmacc_vv_f32m1(G_11_imag, t1_real, t1_imag, vl);

      //G_21_interleaved
      //vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t1_real, t1_imag, G_21_real, G_21_imag, &G_21_real, &G_21_imag, vl);
      G_21_real = vfmacc_vv_f32m1(G_21_real, t2_real, t1_real, vl);
      G_21_real = vfnmsac_vv_f32m1(G_21_real, t2_imag_neg, t1_imag, vl);
      G_21_imag = vfmacc_vv_f32m1(G_21_imag, t2_imag_neg, t1_real, vl);
      G_21_imag = vfmacc_vv_f32m1(G_21_imag, t2_real, t1_imag, vl);


      //G_31_interleaved
      //vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t1_real, t1_imag, G_31_real, G_31_imag, &G_31_real, &G_31_imag, vl);
      G_31_real = vfmacc_vv_f32m1(G_31_real, t3_real, t1_real, vl);
      G_31_real = vfnmsac_vv_f32m1(G_31_real, t3_imag_neg, t1_imag, vl);
      G_31_imag = vfmacc_vv_f32m1(G_31_imag, t3_imag_neg, t1_real, vl);
      G_31_imag = vfmacc_vv_f32m1(G_31_imag, t3_real, t1_imag, vl);


      //G_22_interleaved
      //vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t2_real, t2_imag, G_22_real, G_22_imag, &G_22_real, &G_22_imag, vl);

      G_22_real = vfmacc_vv_f32m1(G_22_real, t2_real, t2_real, vl);
      G_22_real = vfnmsac_vv_f32m1(G_22_real, t2_imag_neg, t2_imag, vl);
      G_22_imag = vfmacc_vv_f32m1(G_22_imag, t2_imag_neg, t2_real, vl);
      G_22_imag = vfmacc_vv_f32m1(G_22_imag, t2_real, t2_imag, vl);


      //G_32_interleaved
      //vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t2_real, t2_imag, G_32_real, G_32_imag, &G_32_real, &G_32_imag, vl);

      G_32_real = vfmacc_vv_f32m1(G_32_real, t3_real, t2_real, vl);
      G_32_real = vfnmsac_vv_f32m1(G_32_real, t3_imag_neg, t2_imag, vl);
      G_32_imag = vfmacc_vv_f32m1(G_32_imag, t3_imag_neg, t2_real, vl);
      G_32_imag = vfmacc_vv_f32m1(G_32_imag, t3_real, t2_imag, vl);

      //G33
      //vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t3_real, t3_imag, G_33_real, G_33_imag, &G_33_real, &G_33_imag, vl);
      
      G_33_real = vfmacc_vv_f32m1(G_33_real, t3_real, t3_real, vl);
      G_33_real = vfnmsac_vv_f32m1(G_33_real, t3_imag_neg, t3_imag, vl);
      G_33_imag = vfmacc_vv_f32m1(G_33_imag, t3_imag_neg, t3_real, vl);
      G_33_imag = vfmacc_vv_f32m1(G_33_imag, t3_real, t3_imag, vl);

    }
      //G_00_interleaved

      t0_real = vle32_v_f32m1(&H_real[GET_IDX(3, 0, 0, nr, vl)], vl);
      G_00_real = vfmacc_vv_f32m1(G_00_real, t0_real, t0_real, vl);

      t0_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 0, 0, nr, vl)], vl);
      t0_imag_neg = vfneg_v_f32m1(t0_imag, vl);
      
      G_00_real = vfnmsac_vv_f32m1(G_00_real, t0_imag, t0_imag_neg, vl);

      vse32_v_f32m1(&Gram_real[GET_IDX(0,0,0,nr,vl)], G_00_real, vl);

    
      G_00_imag = vfmacc_vv_f32m1(G_00_imag, t0_real, t0_imag, vl);
      G_00_imag = vfmacc_vv_f32m1(G_00_imag, t0_imag_neg, t0_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(0,0,0,nr,vl)], G_00_imag, vl);


      //G_10_interleaved
      t1_real = vle32_v_f32m1(&H_real[GET_IDX(3, 1, 0, nr, vl)], vl);
      G_10_real = vfmacc_vv_f32m1(G_10_real, t1_real, t0_real, vl);

      t1_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 1, 0, nr, vl)], vl);
      t1_imag_neg = vfneg_v_f32m1(t1_imag, vl);
  
      G_10_real = vfnmsac_vv_f32m1(G_10_real, t1_imag_neg, t0_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(1,0,0,nr,vl)], G_10_real, vl);

      G_10_imag = vfmacc_vv_f32m1(G_10_imag, t1_real, t0_imag, vl);
      G_10_imag = vfmacc_vv_f32m1(G_10_imag, t1_imag_neg, t0_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(1,0,0,nr,vl)], G_10_imag, vl);



      //G_20_interleaved
      t2_real = vle32_v_f32m1(&H_real[GET_IDX(3, 2, 0, nr, vl)], vl);
      G_20_real = vfmacc_vv_f32m1(G_20_real, t2_real, t0_real, vl);

      t2_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 2, 0, nr, vl)], vl);
      t2_imag_neg = vfneg_v_f32m1(t2_imag, vl);

      G_20_real = vfnmsac_vv_f32m1(G_20_real, t2_imag_neg, t0_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(2,0,0,nr,vl)], G_20_real, vl);

      G_20_imag = vfmacc_vv_f32m1(G_20_imag, t2_real, t0_imag, vl);
      G_20_imag = vfmacc_vv_f32m1(G_20_imag, t2_imag_neg, t0_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(2,0,0,nr,vl)], G_20_imag, vl);


      //G_30_interleaved
      t3_real = vle32_v_f32m1(&H_real[GET_IDX(3, 3, 0, nr, vl)], vl);
      G_30_real = vfmacc_vv_f32m1(G_30_real, t3_real, t0_real, vl);
      
      t3_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 3, 0, nr, vl)], vl);
      t3_imag_neg = vfneg_v_f32m1(t3_imag, vl);

      G_30_real = vfnmsac_vv_f32m1(G_30_real, t3_imag_neg, t0_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(3,0,0,nr,vl)], G_30_real, vl);

      G_30_imag = vfmacc_vv_f32m1(G_30_imag, t3_real, t0_imag, vl);
      G_30_imag = vfmacc_vv_f32m1(G_30_imag, t3_imag_neg, t0_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(3,0,0,nr,vl)], G_30_imag, vl);

      //G_11_interleaved
      //vcfmac_vv_f32m1_sim(t1_real, t1_imag_neg, t1_real, t1_imag, G_11_real, G_11_imag, &G_11_real, &G_11_imag, vl);
      
      G_11_real = vfmacc_vv_f32m1(G_11_real, t1_real, t1_real, vl);
      G_11_real = vfnmsac_vv_f32m1(G_11_real, t1_imag_neg, t1_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(1,1,0,nr,vl)], G_11_real, vl);

      G_11_imag = vfmacc_vv_f32m1(G_11_imag, t1_imag_neg, t1_real, vl);
      G_11_imag = vfmacc_vv_f32m1(G_11_imag, t1_real, t1_imag, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(1,1,0,nr,vl)], G_11_imag, vl);

      //G_21_interleaved
      //vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t1_real, t1_imag, G_21_real, G_21_imag, &G_21_real, &G_21_imag, vl);
      G_21_real = vfmacc_vv_f32m1(G_21_real, t2_real, t1_real, vl);
      G_21_real = vfnmsac_vv_f32m1(G_21_real, t2_imag_neg, t1_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(2,1,0,nr,vl)], G_21_real, vl);

      G_21_imag = vfmacc_vv_f32m1(G_21_imag, t2_imag_neg, t1_real, vl);
      G_21_imag = vfmacc_vv_f32m1(G_21_imag, t2_real, t1_imag, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(2,1,0,nr,vl)], G_21_imag, vl);


      //G_31_interleaved
      //vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t1_real, t1_imag, G_31_real, G_31_imag, &G_31_real, &G_31_imag, vl);
      G_31_real = vfmacc_vv_f32m1(G_31_real, t3_real, t1_real, vl);
      G_31_real = vfnmsac_vv_f32m1(G_31_real, t3_imag_neg, t1_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(3,1,0,nr,vl)], G_31_real, vl);

      G_31_imag = vfmacc_vv_f32m1(G_31_imag, t3_imag_neg, t1_real, vl);
      G_31_imag = vfmacc_vv_f32m1(G_31_imag, t3_real, t1_imag, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(3,1,0,nr,vl)], G_31_imag, vl);


      //G_22_interleaved
      //vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t2_real, t2_imag, G_22_real, G_22_imag, &G_22_real, &G_22_imag, vl);

      G_22_real = vfmacc_vv_f32m1(G_22_real, t2_real, t2_real, vl);
      G_22_real = vfnmsac_vv_f32m1(G_22_real, t2_imag_neg, t2_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(2,2,0,nr,vl)], G_22_real, vl);
    

      G_22_imag = vfmacc_vv_f32m1(G_22_imag, t2_imag_neg, t2_real, vl);
      G_22_imag = vfmacc_vv_f32m1(G_22_imag, t2_real, t2_imag, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(2,2,0,nr,vl)], G_22_imag, vl);


      //G_32_interleaved
      //vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t2_real, t2_imag, G_32_real, G_32_imag, &G_32_real, &G_32_imag, vl);

      G_32_real = vfmacc_vv_f32m1(G_32_real, t3_real, t2_real, vl);
      G_32_real = vfnmsac_vv_f32m1(G_32_real, t3_imag_neg, t2_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(3,2,0,nr,vl)], G_32_real, vl);

      G_32_imag = vfmacc_vv_f32m1(G_32_imag, t3_imag_neg, t2_real, vl);
      G_32_imag = vfmacc_vv_f32m1(G_32_imag, t3_real, t2_imag, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(3,2,0,nr,vl)], G_32_imag, vl);

      //G33
      //vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t3_real, t3_imag, G_33_real, G_33_imag, &G_33_real, &G_33_imag, vl);
      
      G_33_real = vfmacc_vv_f32m1(G_33_real, t3_real, t3_real, vl);
      G_33_real = vfnmsac_vv_f32m1(G_33_real, t3_imag_neg, t3_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(3,3,0,nr,vl)], G_33_real, vl);

      G_33_imag = vfmacc_vv_f32m1(G_33_imag, t3_imag_neg, t3_real, vl);
      G_33_imag = vfmacc_vv_f32m1(G_33_imag, t3_real, t3_imag, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(3,3,0,nr,vl)], G_33_imag, vl);

}


#elif CASE == 5

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
    vfloat32m1_t G_00_real = vfmv_v_f_f32m1(sigma_n2, vl);
    vfloat32m1_t G_10_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_20_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_30_real = vfmv_v_f_f32m1(0.0f, vl);


    //gram_imag
    vfloat32m1_t G_00_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_10_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_20_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_30_imag = vfmv_v_f_f32m1(0.0f, vl);


    // + sigma_n
    //G_00_real = vfadd_vf_f32m1(G_00_real, sigma_n2, vl);
    //G_11_real = vfadd_vf_f32m1(G_11_real, sigma_n2, vl);
    //G_22_real = vfadd_vf_f32m1(G_22_real, sigma_n2, vl);
    //G_33_real = vfadd_vf_f32m1(G_33_real, sigma_n2, vl);

    vfloat32m1_t t0_real, t0_imag, t0_imag_neg;
    vfloat32m1_t t1_real, t1_imag, t1_imag_neg;
    vfloat32m1_t t2_real, t2_imag, t2_imag_neg;
    vfloat32m1_t t3_real, t3_imag, t3_imag_neg;


    for (int ptr_row = 0; ptr_row < 3; ptr_row++){
      //G_00_interleaved

      t0_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 0, 0, nr, vl)], vl);
      G_00_real = vfmacc_vv_f32m1(G_00_real, t0_real, t0_real, vl);

      t0_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 0, 0, nr, vl)], vl);
      t0_imag_neg = vfneg_v_f32m1(t0_imag, vl);
      
      G_00_real = vfnmsac_vv_f32m1(G_00_real, t0_imag, t0_imag_neg, vl);
      G_00_imag = vfmacc_vv_f32m1(G_00_imag, t0_real, t0_imag, vl);
      G_00_imag = vfmacc_vv_f32m1(G_00_imag, t0_imag_neg, t0_real, vl);


      //G_10_interleaved
      t1_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 1, 0, nr, vl)], vl);
      G_10_real = vfmacc_vv_f32m1(G_10_real, t1_real, t0_real, vl);

      t1_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 1, 0, nr, vl)], vl);
      t1_imag_neg = vfneg_v_f32m1(t1_imag, vl);
  
      G_10_real = vfnmsac_vv_f32m1(G_10_real, t1_imag_neg, t0_imag, vl);
      G_10_imag = vfmacc_vv_f32m1(G_10_imag, t1_real, t0_imag, vl);
      G_10_imag = vfmacc_vv_f32m1(G_10_imag, t1_imag_neg, t0_real, vl);



      //G_20_interleaved
      t2_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 2, 0, nr, vl)], vl);
      G_20_real = vfmacc_vv_f32m1(G_20_real, t2_real, t0_real, vl);

      t2_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 2, 0, nr, vl)], vl);
      t2_imag_neg = vfneg_v_f32m1(t2_imag, vl);

      G_20_real = vfnmsac_vv_f32m1(G_20_real, t2_imag_neg, t0_imag, vl);
      G_20_imag = vfmacc_vv_f32m1(G_20_imag, t2_real, t0_imag, vl);
      G_20_imag = vfmacc_vv_f32m1(G_20_imag, t2_imag_neg, t0_real, vl);


      //G_30_interleaved
      t3_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 3, 0, nr, vl)], vl);
      G_30_real = vfmacc_vv_f32m1(G_30_real, t3_real, t0_real, vl);
      
      t3_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 3, 0, nr, vl)], vl);
      t3_imag_neg = vfneg_v_f32m1(t3_imag, vl);

      G_30_real = vfnmsac_vv_f32m1(G_30_real, t3_imag_neg, t0_imag, vl);
      G_30_imag = vfmacc_vv_f32m1(G_30_imag, t3_real, t0_imag, vl);
      G_30_imag = vfmacc_vv_f32m1(G_30_imag, t3_imag_neg, t0_real, vl);


      

    }
      //G_00_interleaved

      t0_real = vle32_v_f32m1(&H_real[GET_IDX(3, 0, 0, nr, vl)], vl);
      G_00_real = vfmacc_vv_f32m1(G_00_real, t0_real, t0_real, vl);

      t0_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 0, 0, nr, vl)], vl);
      t0_imag_neg = vfneg_v_f32m1(t0_imag, vl);
      
      G_00_real = vfnmsac_vv_f32m1(G_00_real, t0_imag, t0_imag_neg, vl);

      vse32_v_f32m1(&Gram_real[GET_IDX(0,0,0,nr,vl)], G_00_real, vl);

    
      G_00_imag = vfmacc_vv_f32m1(G_00_imag, t0_real, t0_imag, vl);
      G_00_imag = vfmacc_vv_f32m1(G_00_imag, t0_imag_neg, t0_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(0,0,0,nr,vl)], G_00_imag, vl);


      //G_10_interleaved
      t1_real = vle32_v_f32m1(&H_real[GET_IDX(3, 1, 0, nr, vl)], vl);
      G_10_real = vfmacc_vv_f32m1(G_10_real, t1_real, t0_real, vl);

      t1_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 1, 0, nr, vl)], vl);
      t1_imag_neg = vfneg_v_f32m1(t1_imag, vl);
  
      G_10_real = vfnmsac_vv_f32m1(G_10_real, t1_imag_neg, t0_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(1,0,0,nr,vl)], G_10_real, vl);

      G_10_imag = vfmacc_vv_f32m1(G_10_imag, t1_real, t0_imag, vl);
      G_10_imag = vfmacc_vv_f32m1(G_10_imag, t1_imag_neg, t0_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(1,0,0,nr,vl)], G_10_imag, vl);



      //G_20_interleaved
      t2_real = vle32_v_f32m1(&H_real[GET_IDX(3, 2, 0, nr, vl)], vl);
      G_20_real = vfmacc_vv_f32m1(G_20_real, t2_real, t0_real, vl);

      t2_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 2, 0, nr, vl)], vl);
      t2_imag_neg = vfneg_v_f32m1(t2_imag, vl);

      G_20_real = vfnmsac_vv_f32m1(G_20_real, t2_imag_neg, t0_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(2,0,0,nr,vl)], G_20_real, vl);

      G_20_imag = vfmacc_vv_f32m1(G_20_imag, t2_real, t0_imag, vl);
      G_20_imag = vfmacc_vv_f32m1(G_20_imag, t2_imag_neg, t0_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(2,0,0,nr,vl)], G_20_imag, vl);


      //G_30_interleaved
      t3_real = vle32_v_f32m1(&H_real[GET_IDX(3, 3, 0, nr, vl)], vl);
      G_30_real = vfmacc_vv_f32m1(G_30_real, t3_real, t0_real, vl);
      
      t3_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 3, 0, nr, vl)], vl);
      t3_imag_neg = vfneg_v_f32m1(t3_imag, vl);

      G_30_real = vfnmsac_vv_f32m1(G_30_real, t3_imag_neg, t0_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(3,0,0,nr,vl)], G_30_real, vl);

      G_30_imag = vfmacc_vv_f32m1(G_30_imag, t3_real, t0_imag, vl);
      G_30_imag = vfmacc_vv_f32m1(G_30_imag, t3_imag_neg, t0_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(3,0,0,nr,vl)], G_30_imag, vl);

      //G_11_interleaved
      //vcfmac_vv_f32m1_sim(t1_real, t1_imag_neg, t1_real, t1_imag, G_11_real, G_11_imag, &G_11_real, &G_11_imag, vl);
      
      

}


#elif CASE == 6

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
    vfloat32m1_t G_00_real = vfmv_v_f_f32m1(sigma_n2, vl);
    vfloat32m1_t G_10_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_20_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_30_real = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_11_real = vfmv_v_f_f32m1(sigma_n2, vl);
    vfloat32m1_t G_21_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_31_real = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_22_real = vfmv_v_f_f32m1(sigma_n2, vl);
    vfloat32m1_t G_32_real = vfmv_v_f_f32m1(0.0f, vl);
    
    vfloat32m1_t G_33_real = vfmv_v_f_f32m1(sigma_n2, vl);

    //gram_imag
    vfloat32m1_t G_00_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_10_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_20_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_30_imag = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_11_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_21_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_31_imag = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_22_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_32_imag = vfmv_v_f_f32m1(0.0f, vl);
    
    vfloat32m1_t G_33_imag = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t t0_real, t0_imag;
    vfloat32m1_t t1_real, t1_imag;
    vfloat32m1_t t2_real, t2_imag;
    vfloat32m1_t t3_real, t3_imag;


    for (int ptr_row = 0; ptr_row < 3; ptr_row++){
      //G_00_interleaved

      t0_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 0, 0, nr, vl)], vl);
      G_00_real = vfmacc_vv_f32m1(G_00_real, t0_real, t0_real, vl);

      t0_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 0, 0, nr, vl)], vl);
      //t0_imag_neg = vfneg_v_f32m1(t0_imag, vl);
      
      G_00_real = vfmacc_vv_f32m1(G_00_real, t0_imag, t0_imag, vl);
      G_00_imag = vfmacc_vv_f32m1(G_00_imag, t0_real, t0_imag, vl);
      G_00_imag = vfnmsac_vv_f32m1(G_00_imag, t0_imag, t0_real, vl);


      //G_10_interleaved
      t1_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 1, 0, nr, vl)], vl);
      G_10_real = vfmacc_vv_f32m1(G_10_real, t1_real, t0_real, vl);

      t1_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 1, 0, nr, vl)], vl);
  
      G_10_real = vfmacc_vv_f32m1(G_10_real, t1_imag, t0_imag, vl);
      G_10_imag = vfmacc_vv_f32m1(G_10_imag, t1_real, t0_imag, vl);
      G_10_imag = vfnmsac_vv_f32m1(G_10_imag, t1_imag, t0_real, vl);



      //G_20_interleaved
      t2_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 2, 0, nr, vl)], vl);
      G_20_real = vfmacc_vv_f32m1(G_20_real, t2_real, t0_real, vl);

      t2_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 2, 0, nr, vl)], vl);

      G_20_real = vfmacc_vv_f32m1(G_20_real, t2_imag, t0_imag, vl);
      G_20_imag = vfmacc_vv_f32m1(G_20_imag, t2_real, t0_imag, vl);
      G_20_imag = vfnmsac_vv_f32m1(G_20_imag, t2_imag, t0_real, vl);


      //G_30_interleaved
      t3_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 3, 0, nr, vl)], vl);
      G_30_real = vfmacc_vv_f32m1(G_30_real, t3_real, t0_real, vl);
      
      t3_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 3, 0, nr, vl)], vl);

      G_30_real = vfmacc_vv_f32m1(G_30_real, t3_imag, t0_imag, vl);
      G_30_imag = vfmacc_vv_f32m1(G_30_imag, t3_real, t0_imag, vl);
      G_30_imag = vfnmsac_vv_f32m1(G_30_imag, t3_imag, t0_real, vl);


      //G_11_interleaved
      //vcfmac_vv_f32m1_sim(t1_real, t1_imag_neg, t1_real, t1_imag, G_11_real, G_11_imag, &G_11_real, &G_11_imag, vl);
      
      G_11_real = vfmacc_vv_f32m1(G_11_real, t1_real, t1_real, vl);
      G_11_real = vfmacc_vv_f32m1(G_11_real, t1_imag, t1_imag, vl);
      G_11_imag = vfnmsac_vv_f32m1(G_11_imag, t1_imag, t1_real, vl);
      G_11_imag = vfmacc_vv_f32m1(G_11_imag, t1_real, t1_imag, vl);

      //G_21_interleaved
      //vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t1_real, t1_imag, G_21_real, G_21_imag, &G_21_real, &G_21_imag, vl);
      G_21_real = vfmacc_vv_f32m1(G_21_real, t2_real, t1_real, vl);
      G_21_real = vfmacc_vv_f32m1(G_21_real, t2_imag, t1_imag, vl);
      G_21_imag = vfnmsac_vv_f32m1(G_21_imag, t2_imag, t1_real, vl);
      G_21_imag = vfmacc_vv_f32m1(G_21_imag, t2_real, t1_imag, vl);


      //G_31_interleaved
      //vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t1_real, t1_imag, G_31_real, G_31_imag, &G_31_real, &G_31_imag, vl);
      G_31_real = vfmacc_vv_f32m1(G_31_real, t3_real, t1_real, vl);
      G_31_real = vfmacc_vv_f32m1(G_31_real, t3_imag, t1_imag, vl);
      G_31_imag = vfnmsac_vv_f32m1(G_31_imag, t3_imag, t1_real, vl);
      G_31_imag = vfmacc_vv_f32m1(G_31_imag, t3_real, t1_imag, vl);


      //G_22_interleaved
      //vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t2_real, t2_imag, G_22_real, G_22_imag, &G_22_real, &G_22_imag, vl);

      G_22_real = vfmacc_vv_f32m1(G_22_real, t2_real, t2_real, vl);
      G_22_real = vfmacc_vv_f32m1(G_22_real, t2_imag, t2_imag, vl);
      G_22_imag = vfnmsac_vv_f32m1(G_22_imag, t2_imag, t2_real, vl);
      G_22_imag = vfmacc_vv_f32m1(G_22_imag, t2_real, t2_imag, vl);


      //G_32_interleaved
      //vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t2_real, t2_imag, G_32_real, G_32_imag, &G_32_real, &G_32_imag, vl);

      G_32_real = vfmacc_vv_f32m1(G_32_real, t3_real, t2_real, vl);
      G_32_real = vfmacc_vv_f32m1(G_32_real, t3_imag, t2_imag, vl);
      G_32_imag = vfnmsac_vv_f32m1(G_32_imag, t3_imag, t2_real, vl);
      G_32_imag = vfmacc_vv_f32m1(G_32_imag, t3_real, t2_imag, vl);

      //G33
      //vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t3_real, t3_imag, G_33_real, G_33_imag, &G_33_real, &G_33_imag, vl);
      
      G_33_real = vfmacc_vv_f32m1(G_33_real, t3_real, t3_real, vl);
      G_33_real = vfmacc_vv_f32m1(G_33_real, t3_imag, t3_imag, vl);
      G_33_imag = vfnmsac_vv_f32m1(G_33_imag, t3_imag, t3_real, vl);
      G_33_imag = vfmacc_vv_f32m1(G_33_imag, t3_real, t3_imag, vl);

      

    }
      //G_00_interleaved

      t0_real = vle32_v_f32m1(&H_real[GET_IDX(3, 0, 0, nr, vl)], vl);
      G_00_real = vfmacc_vv_f32m1(G_00_real, t0_real, t0_real, vl);

      t0_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 0, 0, nr, vl)], vl);
      G_00_real = vfmacc_vv_f32m1(G_00_real, t0_imag, t0_imag, vl);

      vse32_v_f32m1(&Gram_real[GET_IDX(0,0,0,nr,vl)], G_00_real, vl);

    
      G_00_imag = vfmacc_vv_f32m1(G_00_imag, t0_real, t0_imag, vl);
      G_00_imag = vfnmsac_vv_f32m1(G_00_imag, t0_imag, t0_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(0,0,0,nr,vl)], G_00_imag, vl);


      //G_10_interleaved
      t1_real = vle32_v_f32m1(&H_real[GET_IDX(3, 1, 0, nr, vl)], vl);
      G_10_real = vfmacc_vv_f32m1(G_10_real, t1_real, t0_real, vl);

      t1_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 1, 0, nr, vl)], vl);
  
      G_10_real = vfmacc_vv_f32m1(G_10_real, t1_imag, t0_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(1,0,0,nr,vl)], G_10_real, vl);

      G_10_imag = vfmacc_vv_f32m1(G_10_imag, t1_real, t0_imag, vl);
      G_10_imag = vfnmsac_vv_f32m1(G_10_imag, t1_imag, t0_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(1,0,0,nr,vl)], G_10_imag, vl);



      //G_20_interleaved
      t2_real = vle32_v_f32m1(&H_real[GET_IDX(3, 2, 0, nr, vl)], vl);
      G_20_real = vfmacc_vv_f32m1(G_20_real, t2_real, t0_real, vl);

      t2_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 2, 0, nr, vl)], vl);
      //t2_imag_neg = vfneg_v_f32m1(t2_imag, vl);

      G_20_real = vfmacc_vv_f32m1(G_20_real, t2_imag, t0_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(2,0,0,nr,vl)], G_20_real, vl);

      G_20_imag = vfmacc_vv_f32m1(G_20_imag, t2_real, t0_imag, vl);
      G_20_imag = vfnmsac_vv_f32m1(G_20_imag, t2_imag, t0_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(2,0,0,nr,vl)], G_20_imag, vl);


      //G_30_interleaved
      t3_real = vle32_v_f32m1(&H_real[GET_IDX(3, 3, 0, nr, vl)], vl);
      G_30_real = vfmacc_vv_f32m1(G_30_real, t3_real, t0_real, vl);
      
      t3_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 3, 0, nr, vl)], vl);
      //t3_imag_neg = vfneg_v_f32m1(t3_imag, vl);

      G_30_real = vfmacc_vv_f32m1(G_30_real, t3_imag, t0_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(3,0,0,nr,vl)], G_30_real, vl);

      G_30_imag = vfmacc_vv_f32m1(G_30_imag, t3_real, t0_imag, vl);
      G_30_imag = vfnmsac_vv_f32m1(G_30_imag, t3_imag, t0_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(3,0,0,nr,vl)], G_30_imag, vl);

      //G_11_interleaved
      //vcfmac_vv_f32m1_sim(t1_real, t1_imag_neg, t1_real, t1_imag, G_11_real, G_11_imag, &G_11_real, &G_11_imag, vl);
      
      G_11_real = vfmacc_vv_f32m1(G_11_real, t1_real, t1_real, vl);
      G_11_real = vfmacc_vv_f32m1(G_11_real, t1_imag, t1_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(1,1,0,nr,vl)], G_11_real, vl);

      G_11_imag = vfnmsac_vv_f32m1(G_11_imag, t1_imag, t1_real, vl);
      G_11_imag = vfmacc_vv_f32m1(G_11_imag, t1_real, t1_imag, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(1,1,0,nr,vl)], G_11_imag, vl);

      //G_21_interleaved
      //vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t1_real, t1_imag, G_21_real, G_21_imag, &G_21_real, &G_21_imag, vl);
      G_21_real = vfmacc_vv_f32m1(G_21_real, t2_real, t1_real, vl);
      G_21_real = vfmacc_vv_f32m1(G_21_real, t2_imag, t1_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(2,1,0,nr,vl)], G_21_real, vl);

      G_21_imag = vfnmsac_vv_f32m1(G_21_imag, t2_imag, t1_real, vl);
      G_21_imag = vfmacc_vv_f32m1(G_21_imag, t2_real, t1_imag, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(2,1,0,nr,vl)], G_21_imag, vl);


      //G_31_interleaved
      //vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t1_real, t1_imag, G_31_real, G_31_imag, &G_31_real, &G_31_imag, vl);
      G_31_real = vfmacc_vv_f32m1(G_31_real, t3_real, t1_real, vl);
      G_31_real = vfmacc_vv_f32m1(G_31_real, t3_imag, t1_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(3,1,0,nr,vl)], G_31_real, vl);

      G_31_imag = vfnmsac_vv_f32m1(G_31_imag, t3_imag, t1_real, vl);
      G_31_imag = vfmacc_vv_f32m1(G_31_imag, t3_real, t1_imag, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(3,1,0,nr,vl)], G_31_imag, vl);


      //G_22_interleaved
      //vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t2_real, t2_imag, G_22_real, G_22_imag, &G_22_real, &G_22_imag, vl);

      G_22_real = vfmacc_vv_f32m1(G_22_real, t2_real, t2_real, vl);
      G_22_real = vfmacc_vv_f32m1(G_22_real, t2_imag, t2_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(2,2,0,nr,vl)], G_22_real, vl);
    

      G_22_imag = vfnmsac_vv_f32m1(G_22_imag, t2_imag, t2_real, vl);
      G_22_imag = vfmacc_vv_f32m1(G_22_imag, t2_real, t2_imag, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(2,2,0,nr,vl)], G_22_imag, vl);


      //G_32_interleaved
      //vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t2_real, t2_imag, G_32_real, G_32_imag, &G_32_real, &G_32_imag, vl);

      G_32_real = vfmacc_vv_f32m1(G_32_real, t3_real, t2_real, vl);
      G_32_real = vfmacc_vv_f32m1(G_32_real, t3_imag, t2_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(3,2,0,nr,vl)], G_32_real, vl);

      G_32_imag = vfnmsac_vv_f32m1(G_32_imag, t3_imag, t2_real, vl);
      G_32_imag = vfmacc_vv_f32m1(G_32_imag, t3_real, t2_imag, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(3,2,0,nr,vl)], G_32_imag, vl);

      //G33
      //vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t3_real, t3_imag, G_33_real, G_33_imag, &G_33_real, &G_33_imag, vl);
      
      G_33_real = vfmacc_vv_f32m1(G_33_real, t3_real, t3_real, vl);
      G_33_real = vfmacc_vv_f32m1(G_33_real, t3_imag, t3_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(3,3,0,nr,vl)], G_33_real, vl);

      G_33_imag = vfnmsac_vv_f32m1(G_33_imag, t3_imag, t3_real, vl);
      G_33_imag = vfmacc_vv_f32m1(G_33_imag, t3_real, t3_imag, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(3,3,0,nr,vl)], G_33_imag, vl);

}

#elif CASE == 7

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
    vfloat32m1_t G_00_real = vfmv_v_f_f32m1(sigma_n2, vl);
    vfloat32m1_t G_10_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_20_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_30_real = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_11_real = vfmv_v_f_f32m1(sigma_n2, vl);
    vfloat32m1_t G_21_real = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_31_real = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_22_real = vfmv_v_f_f32m1(sigma_n2, vl);
    vfloat32m1_t G_32_real = vfmv_v_f_f32m1(0.0f, vl);
    
    vfloat32m1_t G_33_real = vfmv_v_f_f32m1(sigma_n2, vl);

    //gram_imag
    vfloat32m1_t G_10_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_20_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_30_imag = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_21_imag = vfmv_v_f_f32m1(0.0f, vl);
    vfloat32m1_t G_31_imag = vfmv_v_f_f32m1(0.0f, vl);

    vfloat32m1_t G_32_imag = vfmv_v_f_f32m1(0.0f, vl);
    
    vfloat32m1_t t0_real, t0_imag;
    vfloat32m1_t t1_real, t1_imag;
    vfloat32m1_t t2_real, t2_imag;
    vfloat32m1_t t3_real, t3_imag;


    for (int ptr_row = 0; ptr_row < 3; ptr_row++){
      //G_00_interleaved

      t0_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 0, 0, nr, vl)], vl);
      G_00_real = vfmacc_vv_f32m1(G_00_real, t0_real, t0_real, vl);

      t0_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 0, 0, nr, vl)], vl);
      //t0_imag_neg = vfneg_v_f32m1(t0_imag, vl);
      
      G_00_real = vfmacc_vv_f32m1(G_00_real, t0_imag, t0_imag, vl);


      //G_10_interleaved
      t1_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 1, 0, nr, vl)], vl);
      G_10_real = vfmacc_vv_f32m1(G_10_real, t1_real, t0_real, vl);

      t1_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 1, 0, nr, vl)], vl);
  
      G_10_real = vfmacc_vv_f32m1(G_10_real, t1_imag, t0_imag, vl);
      G_10_imag = vfmacc_vv_f32m1(G_10_imag, t1_real, t0_imag, vl);
      G_10_imag = vfnmsac_vv_f32m1(G_10_imag, t1_imag, t0_real, vl);



      //G_20_interleaved
      t2_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 2, 0, nr, vl)], vl);
      G_20_real = vfmacc_vv_f32m1(G_20_real, t2_real, t0_real, vl);

      t2_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 2, 0, nr, vl)], vl);

      G_20_real = vfmacc_vv_f32m1(G_20_real, t2_imag, t0_imag, vl);
      G_20_imag = vfmacc_vv_f32m1(G_20_imag, t2_real, t0_imag, vl);
      G_20_imag = vfnmsac_vv_f32m1(G_20_imag, t2_imag, t0_real, vl);


      //G_30_interleaved
      t3_real = vle32_v_f32m1(&H_real[GET_IDX(ptr_row, 3, 0, nr, vl)], vl);
      G_30_real = vfmacc_vv_f32m1(G_30_real, t3_real, t0_real, vl);
      
      t3_imag = vle32_v_f32m1(&H_imag[GET_IDX(ptr_row, 3, 0, nr, vl)], vl);

      G_30_real = vfmacc_vv_f32m1(G_30_real, t3_imag, t0_imag, vl);
      G_30_imag = vfmacc_vv_f32m1(G_30_imag, t3_real, t0_imag, vl);
      G_30_imag = vfnmsac_vv_f32m1(G_30_imag, t3_imag, t0_real, vl);


      //G_11_interleaved
      //vcfmac_vv_f32m1_sim(t1_real, t1_imag_neg, t1_real, t1_imag, G_11_real, G_11_imag, &G_11_real, &G_11_imag, vl);
      
      G_11_real = vfmacc_vv_f32m1(G_11_real, t1_real, t1_real, vl);
      G_21_real = vfmacc_vv_f32m1(G_21_real, t2_real, t1_real, vl);
      G_31_imag = vfnmsac_vv_f32m1(G_31_imag, t3_imag, t1_real, vl);
      G_31_real = vfmacc_vv_f32m1(G_31_real, t3_real, t1_real, vl);
      G_21_imag = vfnmsac_vv_f32m1(G_21_imag, t2_imag, t1_real, vl);

      G_21_imag = vfmacc_vv_f32m1(G_21_imag, t2_real, t1_imag, vl);
      G_11_real = vfmacc_vv_f32m1(G_11_real, t1_imag, t1_imag, vl);

      //G_21_interleaved
      //vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t1_real, t1_imag, G_21_real, G_21_imag, &G_21_real, &G_21_imag, vl);
      
      G_21_real = vfmacc_vv_f32m1(G_21_real, t2_imag, t1_imag, vl);
      


      //G_31_interleaved
      //vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t1_real, t1_imag, G_31_real, G_31_imag, &G_31_real, &G_31_imag, vl);
      
      G_31_imag = vfmacc_vv_f32m1(G_31_imag, t3_real, t1_imag, vl);
      G_31_real = vfmacc_vv_f32m1(G_31_real, t3_imag, t1_imag, vl);
      
      

      //G_22_interleaved
      //vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t2_real, t2_imag, G_22_real, G_22_imag, &G_22_real, &G_22_imag, vl);

      G_22_real = vfmacc_vv_f32m1(G_22_real, t2_real, t2_real, vl);
      G_22_real = vfmacc_vv_f32m1(G_22_real, t2_imag, t2_imag, vl);


      //G_32_interleaved
      //vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t2_real, t2_imag, G_32_real, G_32_imag, &G_32_real, &G_32_imag, vl);
      G_32_imag = vfmacc_vv_f32m1(G_32_imag, t3_real, t2_imag, vl);
      G_32_real = vfmacc_vv_f32m1(G_32_real, t3_real, t2_real, vl);
      G_32_imag = vfnmsac_vv_f32m1(G_32_imag, t3_imag, t2_real, vl);
      G_32_real = vfmacc_vv_f32m1(G_32_real, t3_imag, t2_imag, vl);
      
     

      //G33
      //vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t3_real, t3_imag, G_33_real, G_33_imag, &G_33_real, &G_33_imag, vl);
      
      G_33_real = vfmacc_vv_f32m1(G_33_real, t3_real, t3_real, vl);
      G_33_real = vfmacc_vv_f32m1(G_33_real, t3_imag, t3_imag, vl);

      

    }
      //G_00_interleaved

      t0_real = vle32_v_f32m1(&H_real[GET_IDX(3, 0, 0, nr, vl)], vl);
      G_00_real = vfmacc_vv_f32m1(G_00_real, t0_real, t0_real, vl);

      t0_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 0, 0, nr, vl)], vl);
      G_00_real = vfmacc_vv_f32m1(G_00_real, t0_imag, t0_imag, vl);

      vse32_v_f32m1(&Gram_real[GET_IDX(0,0,0,nr,vl)], G_00_real, vl);

  


      //G_10_interleaved
      t1_real = vle32_v_f32m1(&H_real[GET_IDX(3, 1, 0, nr, vl)], vl);
      G_10_real = vfmacc_vv_f32m1(G_10_real, t1_real, t0_real, vl);

      t1_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 1, 0, nr, vl)], vl);
  
      G_10_real = vfmacc_vv_f32m1(G_10_real, t1_imag, t0_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(1,0,0,nr,vl)], G_10_real, vl);

      G_10_imag = vfmacc_vv_f32m1(G_10_imag, t1_real, t0_imag, vl);
      G_10_imag = vfnmsac_vv_f32m1(G_10_imag, t1_imag, t0_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(1,0,0,nr,vl)], G_10_imag, vl);



      //G_20_interleaved
      t2_real = vle32_v_f32m1(&H_real[GET_IDX(3, 2, 0, nr, vl)], vl);
      G_20_real = vfmacc_vv_f32m1(G_20_real, t2_real, t0_real, vl);

      t2_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 2, 0, nr, vl)], vl);
      //t2_imag_neg = vfneg_v_f32m1(t2_imag, vl);

      G_20_real = vfmacc_vv_f32m1(G_20_real, t2_imag, t0_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(2,0,0,nr,vl)], G_20_real, vl);

      G_20_imag = vfmacc_vv_f32m1(G_20_imag, t2_real, t0_imag, vl);
      G_20_imag = vfnmsac_vv_f32m1(G_20_imag, t2_imag, t0_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(2,0,0,nr,vl)], G_20_imag, vl);


      //G_30_interleaved
      t3_real = vle32_v_f32m1(&H_real[GET_IDX(3, 3, 0, nr, vl)], vl);
      G_30_real = vfmacc_vv_f32m1(G_30_real, t3_real, t0_real, vl);
      
      t3_imag = vle32_v_f32m1(&H_imag[GET_IDX(3, 3, 0, nr, vl)], vl);
      //t3_imag_neg = vfneg_v_f32m1(t3_imag, vl);

      G_30_real = vfmacc_vv_f32m1(G_30_real, t3_imag, t0_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(3,0,0,nr,vl)], G_30_real, vl);

      G_30_imag = vfmacc_vv_f32m1(G_30_imag, t3_real, t0_imag, vl);
      G_30_imag = vfnmsac_vv_f32m1(G_30_imag, t3_imag, t0_real, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(3,0,0,nr,vl)], G_30_imag, vl);

      //G_11_interleaved
      //vcfmac_vv_f32m1_sim(t1_real, t1_imag_neg, t1_real, t1_imag, G_11_real, G_11_imag, &G_11_real, &G_11_imag, vl);
      
      G_11_real = vfmacc_vv_f32m1(G_11_real, t1_real, t1_real, vl);
      G_11_real = vfmacc_vv_f32m1(G_11_real, t1_imag, t1_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(1,1,0,nr,vl)], G_11_real, vl);

      //G_21_interleaved
      //vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t1_real, t1_imag, G_21_real, G_21_imag, &G_21_real, &G_21_imag, vl);
      G_21_real = vfmacc_vv_f32m1(G_21_real, t2_real, t1_real, vl);
      G_21_real = vfmacc_vv_f32m1(G_21_real, t2_imag, t1_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(2,1,0,nr,vl)], G_21_real, vl);

      G_21_imag = vfnmsac_vv_f32m1(G_21_imag, t2_imag, t1_real, vl);
      G_21_imag = vfmacc_vv_f32m1(G_21_imag, t2_real, t1_imag, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(2,1,0,nr,vl)], G_21_imag, vl);


      //G_31_interleaved
      //vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t1_real, t1_imag, G_31_real, G_31_imag, &G_31_real, &G_31_imag, vl);
      G_31_real = vfmacc_vv_f32m1(G_31_real, t3_real, t1_real, vl);
      G_31_real = vfmacc_vv_f32m1(G_31_real, t3_imag, t1_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(3,1,0,nr,vl)], G_31_real, vl);

      G_31_imag = vfnmsac_vv_f32m1(G_31_imag, t3_imag, t1_real, vl);
      G_31_imag = vfmacc_vv_f32m1(G_31_imag, t3_real, t1_imag, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(3,1,0,nr,vl)], G_31_imag, vl);


      //G_22_interleaved
      //vcfmac_vv_f32m1_sim(t2_real, t2_imag_neg, t2_real, t2_imag, G_22_real, G_22_imag, &G_22_real, &G_22_imag, vl);

      G_22_real = vfmacc_vv_f32m1(G_22_real, t2_real, t2_real, vl);
      G_22_real = vfmacc_vv_f32m1(G_22_real, t2_imag, t2_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(2,2,0,nr,vl)], G_22_real, vl);
    



      //G_32_interleaved
      //vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t2_real, t2_imag, G_32_real, G_32_imag, &G_32_real, &G_32_imag, vl);

      G_32_real = vfmacc_vv_f32m1(G_32_real, t3_real, t2_real, vl);
      G_32_real = vfmacc_vv_f32m1(G_32_real, t3_imag, t2_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(3,2,0,nr,vl)], G_32_real, vl);

      G_32_imag = vfnmsac_vv_f32m1(G_32_imag, t3_imag, t2_real, vl);
      G_32_imag = vfmacc_vv_f32m1(G_32_imag, t3_real, t2_imag, vl);
      vse32_v_f32m1(&Gram_imag[GET_IDX(3,2,0,nr,vl)], G_32_imag, vl);

      //G33
      //vcfmac_vv_f32m1_sim(t3_real, t3_imag_neg, t3_real, t3_imag, G_33_real, G_33_imag, &G_33_real, &G_33_imag, vl);
      
      G_33_real = vfmacc_vv_f32m1(G_33_real, t3_real, t3_real, vl);
      G_33_real = vfmacc_vv_f32m1(G_33_real, t3_imag, t3_imag, vl);
      vse32_v_f32m1(&Gram_real[GET_IDX(3,3,0,nr,vl)], G_33_real, vl);

}

#endif














