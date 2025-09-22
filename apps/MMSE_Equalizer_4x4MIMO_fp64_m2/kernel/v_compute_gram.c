// Copyright 2025 Vodafone Sponsored Chair of Communication System, TU Dresden and Synopsys

// Author: Zexu Tan:zexu.tan@tu-dresden.de, Vodafone Chair, TU Dresden


#include <riscv_vector.h>
#include "arith_lib.h"


#define CASE 1

//case 0 V1*
//case 1 V2*



#if CASE == 0

void v_compute_gram_fp32(
  size_t nt, 
  size_t nr, 
  size_t vl, 
  double sigma_n2, 
  double *H_real,
  double *H_imag,
  double *Gram_real, 
  double *Gram_imag) {

    //gram_real
    vfloat64m2_t G_00_real = vfmv_v_f_f64m2(sigma_n2, vl);
    vfloat64m2_t G_10_real = vfmv_v_f_f64m2(0.0f, vl);
    vfloat64m2_t G_20_real = vfmv_v_f_f64m2(0.0f, vl);
    vfloat64m2_t G_30_real = vfmv_v_f_f64m2(0.0f, vl);

    vfloat64m2_t G_11_real = vfmv_v_f_f64m2(sigma_n2, vl);
    vfloat64m2_t G_21_real = vfmv_v_f_f64m2(0.0f, vl);
    vfloat64m2_t G_31_real = vfmv_v_f_f64m2(0.0f, vl);

    vfloat64m2_t G_22_real = vfmv_v_f_f64m2(sigma_n2, vl);
    vfloat64m2_t G_32_real = vfmv_v_f_f64m2(0.0f, vl);
    
    vfloat64m2_t G_33_real = vfmv_v_f_f64m2(sigma_n2, vl);

    //gram_imag
    vfloat64m2_t G_10_imag = vfmv_v_f_f64m2(0.0f, vl);
    vfloat64m2_t G_20_imag = vfmv_v_f_f64m2(0.0f, vl);
    vfloat64m2_t G_30_imag = vfmv_v_f_f64m2(0.0f, vl);

    vfloat64m2_t G_21_imag = vfmv_v_f_f64m2(0.0f, vl);
    vfloat64m2_t G_31_imag = vfmv_v_f_f64m2(0.0f, vl);

    vfloat64m2_t G_32_imag = vfmv_v_f_f64m2(0.0f, vl);
    
    vfloat64m2_t t0_real, t0_imag;
    vfloat64m2_t t1_real, t1_imag;
    vfloat64m2_t t2_real, t2_imag;
    vfloat64m2_t t3_real, t3_imag;


    for (int ptr_row = 0; ptr_row < 3; ptr_row++){
      //G_00_interleaved

      t0_real = vle64_v_f64m2(&H_real[GET_IDX(ptr_row, 0, 0, nr, vl)], vl);
      G_00_real = vfmacc_vv_f64m2(G_00_real, t0_real, t0_real, vl);

      t0_imag = vle64_v_f64m2(&H_imag[GET_IDX(ptr_row, 0, 0, nr, vl)], vl);
      //t0_imag_neg = vfneg_v_f64m2(t0_imag, vl);
      
      G_00_real = vfmacc_vv_f64m2(G_00_real, t0_imag, t0_imag, vl);


      //G_10_interleaved
      t1_real = vle64_v_f64m2(&H_real[GET_IDX(ptr_row, 1, 0, nr, vl)], vl);
      G_10_real = vfmacc_vv_f64m2(G_10_real, t1_real, t0_real, vl);

      t1_imag = vle64_v_f64m2(&H_imag[GET_IDX(ptr_row, 1, 0, nr, vl)], vl);
  
      G_10_real = vfmacc_vv_f64m2(G_10_real, t1_imag, t0_imag, vl);
      G_10_imag = vfmacc_vv_f64m2(G_10_imag, t1_real, t0_imag, vl);
      G_10_imag = vfnmsac_vv_f64m2(G_10_imag, t1_imag, t0_real, vl);



      //G_20_interleaved
      t2_real = vle64_v_f64m2(&H_real[GET_IDX(ptr_row, 2, 0, nr, vl)], vl);
      G_20_real = vfmacc_vv_f64m2(G_20_real, t2_real, t0_real, vl);

      t2_imag = vle64_v_f64m2(&H_imag[GET_IDX(ptr_row, 2, 0, nr, vl)], vl);

      G_20_real = vfmacc_vv_f64m2(G_20_real, t2_imag, t0_imag, vl);
      G_20_imag = vfmacc_vv_f64m2(G_20_imag, t2_real, t0_imag, vl);
      G_20_imag = vfnmsac_vv_f64m2(G_20_imag, t2_imag, t0_real, vl);


      //G_30_interleaved
      t3_real = vle64_v_f64m2(&H_real[GET_IDX(ptr_row, 3, 0, nr, vl)], vl);
      G_30_real = vfmacc_vv_f64m2(G_30_real, t3_real, t0_real, vl);
      
      t3_imag = vle64_v_f64m2(&H_imag[GET_IDX(ptr_row, 3, 0, nr, vl)], vl);

      G_30_real = vfmacc_vv_f64m2(G_30_real, t3_imag, t0_imag, vl);
      G_30_imag = vfmacc_vv_f64m2(G_30_imag, t3_real, t0_imag, vl);
      G_30_imag = vfnmsac_vv_f64m2(G_30_imag, t3_imag, t0_real, vl);


      //G_11_interleaved
      //vcfmac_vv_f64m2_sim(t1_real, t1_imag_neg, t1_real, t1_imag, G_11_real, G_11_imag, &G_11_real, &G_11_imag, vl);
      
      G_11_real = vfmacc_vv_f64m2(G_11_real, t1_real, t1_real, vl);
      G_21_real = vfmacc_vv_f64m2(G_21_real, t2_real, t1_real, vl);
      G_31_imag = vfnmsac_vv_f64m2(G_31_imag, t3_imag, t1_real, vl);
      G_31_real = vfmacc_vv_f64m2(G_31_real, t3_real, t1_real, vl);
      G_21_imag = vfnmsac_vv_f64m2(G_21_imag, t2_imag, t1_real, vl);

      G_21_imag = vfmacc_vv_f64m2(G_21_imag, t2_real, t1_imag, vl);
      G_11_real = vfmacc_vv_f64m2(G_11_real, t1_imag, t1_imag, vl);

      //G_21_interleaved
      //vcfmac_vv_f64m2_sim(t2_real, t2_imag_neg, t1_real, t1_imag, G_21_real, G_21_imag, &G_21_real, &G_21_imag, vl);
      
      G_21_real = vfmacc_vv_f64m2(G_21_real, t2_imag, t1_imag, vl);
      


      //G_31_interleaved
      //vcfmac_vv_f64m2_sim(t3_real, t3_imag_neg, t1_real, t1_imag, G_31_real, G_31_imag, &G_31_real, &G_31_imag, vl);
      
      G_31_imag = vfmacc_vv_f64m2(G_31_imag, t3_real, t1_imag, vl);
      G_31_real = vfmacc_vv_f64m2(G_31_real, t3_imag, t1_imag, vl);
      
      

      //G_22_interleaved
      //vcfmac_vv_f64m2_sim(t2_real, t2_imag_neg, t2_real, t2_imag, G_22_real, G_22_imag, &G_22_real, &G_22_imag, vl);

      G_22_real = vfmacc_vv_f64m2(G_22_real, t2_real, t2_real, vl);
      G_22_real = vfmacc_vv_f64m2(G_22_real, t2_imag, t2_imag, vl);


      //G_32_interleaved
      //vcfmac_vv_f64m2_sim(t3_real, t3_imag_neg, t2_real, t2_imag, G_32_real, G_32_imag, &G_32_real, &G_32_imag, vl);
      G_32_imag = vfmacc_vv_f64m2(G_32_imag, t3_real, t2_imag, vl);
      G_32_real = vfmacc_vv_f64m2(G_32_real, t3_real, t2_real, vl);
      G_32_imag = vfnmsac_vv_f64m2(G_32_imag, t3_imag, t2_real, vl);
      G_32_real = vfmacc_vv_f64m2(G_32_real, t3_imag, t2_imag, vl);
      
     

      //G33
      //vcfmac_vv_f64m2_sim(t3_real, t3_imag_neg, t3_real, t3_imag, G_33_real, G_33_imag, &G_33_real, &G_33_imag, vl);
      
      G_33_real = vfmacc_vv_f64m2(G_33_real, t3_real, t3_real, vl);
      G_33_real = vfmacc_vv_f64m2(G_33_real, t3_imag, t3_imag, vl);

      

    }
      //G_00_interleaved

      t0_real = vle64_v_f64m2(&H_real[GET_IDX(3, 0, 0, nr, vl)], vl);
      G_00_real = vfmacc_vv_f64m2(G_00_real, t0_real, t0_real, vl);

      t0_imag = vle64_v_f64m2(&H_imag[GET_IDX(3, 0, 0, nr, vl)], vl);
      G_00_real = vfmacc_vv_f64m2(G_00_real, t0_imag, t0_imag, vl);

      vse64_v_f64m2(&Gram_real[GET_IDX(0,0,0,nr,vl)], G_00_real, vl);

  


      //G_10_interleaved
      t1_real = vle64_v_f64m2(&H_real[GET_IDX(3, 1, 0, nr, vl)], vl);
      G_10_real = vfmacc_vv_f64m2(G_10_real, t1_real, t0_real, vl);

      t1_imag = vle64_v_f64m2(&H_imag[GET_IDX(3, 1, 0, nr, vl)], vl);
  
      G_10_real = vfmacc_vv_f64m2(G_10_real, t1_imag, t0_imag, vl);
      vse64_v_f64m2(&Gram_real[GET_IDX(1,0,0,nr,vl)], G_10_real, vl);

      G_10_imag = vfmacc_vv_f64m2(G_10_imag, t1_real, t0_imag, vl);
      G_10_imag = vfnmsac_vv_f64m2(G_10_imag, t1_imag, t0_real, vl);
      vse64_v_f64m2(&Gram_imag[GET_IDX(1,0,0,nr,vl)], G_10_imag, vl);



      //G_20_interleaved
      t2_real = vle64_v_f64m2(&H_real[GET_IDX(3, 2, 0, nr, vl)], vl);
      G_20_real = vfmacc_vv_f64m2(G_20_real, t2_real, t0_real, vl);

      t2_imag = vle64_v_f64m2(&H_imag[GET_IDX(3, 2, 0, nr, vl)], vl);
      //t2_imag_neg = vfneg_v_f64m2(t2_imag, vl);

      G_20_real = vfmacc_vv_f64m2(G_20_real, t2_imag, t0_imag, vl);
      vse64_v_f64m2(&Gram_real[GET_IDX(2,0,0,nr,vl)], G_20_real, vl);

      G_20_imag = vfmacc_vv_f64m2(G_20_imag, t2_real, t0_imag, vl);
      G_20_imag = vfnmsac_vv_f64m2(G_20_imag, t2_imag, t0_real, vl);
      vse64_v_f64m2(&Gram_imag[GET_IDX(2,0,0,nr,vl)], G_20_imag, vl);


      //G_30_interleaved
      t3_real = vle64_v_f64m2(&H_real[GET_IDX(3, 3, 0, nr, vl)], vl);
      G_30_real = vfmacc_vv_f64m2(G_30_real, t3_real, t0_real, vl);
      
      t3_imag = vle64_v_f64m2(&H_imag[GET_IDX(3, 3, 0, nr, vl)], vl);
      //t3_imag_neg = vfneg_v_f64m2(t3_imag, vl);

      G_30_real = vfmacc_vv_f64m2(G_30_real, t3_imag, t0_imag, vl);
      vse64_v_f64m2(&Gram_real[GET_IDX(3,0,0,nr,vl)], G_30_real, vl);

      G_30_imag = vfmacc_vv_f64m2(G_30_imag, t3_real, t0_imag, vl);
      G_30_imag = vfnmsac_vv_f64m2(G_30_imag, t3_imag, t0_real, vl);
      vse64_v_f64m2(&Gram_imag[GET_IDX(3,0,0,nr,vl)], G_30_imag, vl);

      //G_11_interleaved
      //vcfmac_vv_f64m2_sim(t1_real, t1_imag_neg, t1_real, t1_imag, G_11_real, G_11_imag, &G_11_real, &G_11_imag, vl);
      
      G_11_real = vfmacc_vv_f64m2(G_11_real, t1_real, t1_real, vl);
      G_11_real = vfmacc_vv_f64m2(G_11_real, t1_imag, t1_imag, vl);
      vse64_v_f64m2(&Gram_real[GET_IDX(1,1,0,nr,vl)], G_11_real, vl);

      //G_21_interleaved
      //vcfmac_vv_f64m2_sim(t2_real, t2_imag_neg, t1_real, t1_imag, G_21_real, G_21_imag, &G_21_real, &G_21_imag, vl);
      G_21_real = vfmacc_vv_f64m2(G_21_real, t2_real, t1_real, vl);
      G_21_real = vfmacc_vv_f64m2(G_21_real, t2_imag, t1_imag, vl);
      vse64_v_f64m2(&Gram_real[GET_IDX(2,1,0,nr,vl)], G_21_real, vl);

      G_21_imag = vfnmsac_vv_f64m2(G_21_imag, t2_imag, t1_real, vl);
      G_21_imag = vfmacc_vv_f64m2(G_21_imag, t2_real, t1_imag, vl);
      vse64_v_f64m2(&Gram_imag[GET_IDX(2,1,0,nr,vl)], G_21_imag, vl);


      //G_31_interleaved
      //vcfmac_vv_f64m2_sim(t3_real, t3_imag_neg, t1_real, t1_imag, G_31_real, G_31_imag, &G_31_real, &G_31_imag, vl);
      G_31_real = vfmacc_vv_f64m2(G_31_real, t3_real, t1_real, vl);
      G_31_real = vfmacc_vv_f64m2(G_31_real, t3_imag, t1_imag, vl);
      vse64_v_f64m2(&Gram_real[GET_IDX(3,1,0,nr,vl)], G_31_real, vl);

      G_31_imag = vfnmsac_vv_f64m2(G_31_imag, t3_imag, t1_real, vl);
      G_31_imag = vfmacc_vv_f64m2(G_31_imag, t3_real, t1_imag, vl);
      vse64_v_f64m2(&Gram_imag[GET_IDX(3,1,0,nr,vl)], G_31_imag, vl);


      //G_22_interleaved
      //vcfmac_vv_f64m2_sim(t2_real, t2_imag_neg, t2_real, t2_imag, G_22_real, G_22_imag, &G_22_real, &G_22_imag, vl);

      G_22_real = vfmacc_vv_f64m2(G_22_real, t2_real, t2_real, vl);
      G_22_real = vfmacc_vv_f64m2(G_22_real, t2_imag, t2_imag, vl);
      vse64_v_f64m2(&Gram_real[GET_IDX(2,2,0,nr,vl)], G_22_real, vl);
    



      //G_32_interleaved
      //vcfmac_vv_f64m2_sim(t3_real, t3_imag_neg, t2_real, t2_imag, G_32_real, G_32_imag, &G_32_real, &G_32_imag, vl);

      G_32_real = vfmacc_vv_f64m2(G_32_real, t3_real, t2_real, vl);
      G_32_real = vfmacc_vv_f64m2(G_32_real, t3_imag, t2_imag, vl);
      vse64_v_f64m2(&Gram_real[GET_IDX(3,2,0,nr,vl)], G_32_real, vl);

      G_32_imag = vfnmsac_vv_f64m2(G_32_imag, t3_imag, t2_real, vl);
      G_32_imag = vfmacc_vv_f64m2(G_32_imag, t3_real, t2_imag, vl);
      vse64_v_f64m2(&Gram_imag[GET_IDX(3,2,0,nr,vl)], G_32_imag, vl);

      //G33
      //vcfmac_vv_f64m2_sim(t3_real, t3_imag_neg, t3_real, t3_imag, G_33_real, G_33_imag, &G_33_real, &G_33_imag, vl);
      
      G_33_real = vfmacc_vv_f64m2(G_33_real, t3_real, t3_real, vl);
      G_33_real = vfmacc_vv_f64m2(G_33_real, t3_imag, t3_imag, vl);
      vse64_v_f64m2(&Gram_real[GET_IDX(3,3,0,nr,vl)], G_33_real, vl);

}

#elif CASE == 1

void v_compute_gram_fp32(
  size_t nt, 
  size_t nr, 
  size_t vl, 
  double sigma_n2, 
  double *H_real,
  double *H_imag,
  double *Gram_real, 
  double *Gram_imag) {
    vfloat64m2_t t0_real, t0_imag, t1_real, t1_imag, t2_real, t2_imag, t3_real, t3_imag;
    vfloat64m2_t ta_real, ta_imag, tb_real, tb_imag, tc_real, tc_imag, td_real, td_imag;
    vfloat64m2_t acc_0_real;
    vfloat64m2_t acc_0_imag;

    
    for (int i = 0; i < 4; i++) {
      acc_0_real = vfmv_v_f_f64m2(sigma_n2, vl);

    //G_ii
      //0a
      t0_real = vle64_v_f64m2(&H_real[GET_IDX(0, i, 0, nr, vl)], vl);
      acc_0_real = vfmacc_vv_f64m2(acc_0_real, t0_real, t0_real, vl);

      ta_imag = vle64_v_f64m2(&H_imag[GET_IDX(0, i, 0, nr, vl)], vl);
      t0_imag = vfneg_v_f64m2(ta_imag, vl);

      acc_0_real = vfnmsac_vv_f64m2(acc_0_real, t0_imag, ta_imag, vl);
      
      //1b
      t1_real = vle64_v_f64m2(&H_real[GET_IDX(1, i, 0, nr, vl)], vl);
      acc_0_real = vfmacc_vv_f64m2(acc_0_real, t1_real, t1_real, vl);

      tb_imag = vle64_v_f64m2(&H_imag[GET_IDX(1, i, 0, nr, vl)], vl);
      t1_imag = vfneg_v_f64m2(tb_imag, vl);
      acc_0_real = vfnmsac_vv_f64m2(acc_0_real, t1_imag, tb_imag, vl);


      //2c
      t2_real = vle64_v_f64m2(&H_real[GET_IDX(2, i, 0, nr, vl)], vl);
      acc_0_real = vfmacc_vv_f64m2(acc_0_real, t2_real, t2_real, vl);

      tc_imag = vle64_v_f64m2(&H_imag[GET_IDX(2, i, 0, nr, vl)], vl);
      t2_imag = vfneg_v_f64m2(tc_imag, vl);
      acc_0_real = vfnmsac_vv_f64m2(acc_0_real, t2_imag, tc_imag, vl);


          

      //3d
      t3_real = vle64_v_f64m2(&H_real[GET_IDX(3, i, 0, nr, vl)], vl);
      acc_0_real = vfmacc_vv_f64m2(acc_0_real, t3_real, t3_real, vl);

      td_imag = vle64_v_f64m2(&H_imag[GET_IDX(3, i, 0, nr, vl)], vl);
      t3_imag = vfneg_v_f64m2(td_imag, vl);
      acc_0_real = vfnmsac_vv_f64m2(acc_0_real, t3_imag, td_imag, vl);
      vse64_v_f64m2(&Gram_real[GET_IDX(i,i,0,nr,vl)], acc_0_real, vl);
      
      
      for (int j = i - 1; j > -1; j--) {
        acc_0_real = vfmv_v_f_f64m2(0.0f, vl);
        acc_0_imag = vfmv_v_f_f64m2(0.0f, vl);

        ta_real = vle64_v_f64m2(&H_real[GET_IDX(0, j, 0, nr, vl)], vl);
        acc_0_real = vfmacc_vv_f64m2(acc_0_real, t0_real, ta_real, vl);

        ta_imag = vle64_v_f64m2(&H_imag[GET_IDX(0, j, 0, nr, vl)], vl);

        acc_0_real = vfnmsac_vv_f64m2(acc_0_real, t0_imag, ta_imag, vl);
        acc_0_imag = vfmacc_vv_f64m2(acc_0_imag, t0_real, ta_imag, vl);
        acc_0_imag = vfmacc_vv_f64m2(acc_0_imag, t0_imag, ta_real, vl);
      
        //1b
        tb_real = vle64_v_f64m2(&H_real[GET_IDX(1, j, 0, nr, vl)], vl);
        acc_0_real = vfmacc_vv_f64m2(acc_0_real, t1_real, tb_real, vl);

        tb_imag = vle64_v_f64m2(&H_imag[GET_IDX(1, j, 0, nr, vl)], vl);
        acc_0_real = vfnmsac_vv_f64m2(acc_0_real, t1_imag, tb_imag, vl);

        acc_0_imag = vfmacc_vv_f64m2(acc_0_imag, t1_real, tb_imag, vl);
        acc_0_imag = vfmacc_vv_f64m2(acc_0_imag, t1_imag, tb_real, vl);


        //2c
        tc_real = vle64_v_f64m2(&H_real[GET_IDX(2, j, 0, nr, vl)], vl);
        acc_0_real = vfmacc_vv_f64m2(acc_0_real, t2_real, tc_real, vl);

        tc_imag = vle64_v_f64m2(&H_imag[GET_IDX(2, j, 0, nr, vl)], vl);
        acc_0_real = vfnmsac_vv_f64m2(acc_0_real, t2_imag, tc_imag, vl);

        acc_0_imag = vfmacc_vv_f64m2(acc_0_imag, t2_real, tc_imag, vl);
        acc_0_imag = vfmacc_vv_f64m2(acc_0_imag, t2_imag, tc_real, vl);

          

        //3d
        td_real = vle64_v_f64m2(&H_real[GET_IDX(3, j, 0, nr, vl)], vl);
        acc_0_real = vfmacc_vv_f64m2(acc_0_real, t3_real, td_real, vl);

        td_imag = vle64_v_f64m2(&H_imag[GET_IDX(3, j, 0, nr, vl)], vl);
        acc_0_real = vfnmsac_vv_f64m2(acc_0_real, t3_imag, td_imag, vl);
        vse64_v_f64m2(&Gram_real[GET_IDX(i,j,0,nr,vl)], acc_0_real, vl);

        acc_0_imag = vfmacc_vv_f64m2(acc_0_imag, t3_real, td_imag, vl);
        acc_0_imag = vfmacc_vv_f64m2(acc_0_imag, t3_imag, td_real, vl);
        vse64_v_f64m2(&Gram_imag[GET_IDX(i,j,0,nr,vl)], acc_0_imag, vl); 




      }
    }
  }


#endif














