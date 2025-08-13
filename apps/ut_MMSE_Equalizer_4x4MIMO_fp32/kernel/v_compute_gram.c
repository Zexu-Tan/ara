// Copyright 2025 Vodafone Sponsored Chair of Communication System, TU Dresden and Synopsys

// Author: Zexu Tan:zexu.tan@tu-dresden.de, Vodafone Chair, TU Dresden


#include <riscv_vector.h>
#include "arith_lib.h"



void v_compute_gram_f32(size_t nt, size_t nr, size_t vl, float sigma_n2, float H_real[nt][nr][vl], float H_imag[nt][nr][vl], float Gram_real[nt][nr][vl], float Gram_imag[nt][nr][vl]){
  vfloat32m1_t temp_v_A_real, temp_v_A_imag, temp_v_B_real, temp_v_B_imag, temp_v_C_real, temp_v_C_imag, temp_v_D_real, temp_v_D_imag;
  vfloat32m1_t temp_v_A_imag_conj, temp_v_B_imag_conj, temp_v_C_imag_conj, temp_v_D_imag_conj;
  vfloat32m1_t temp_v_acc_real, temp_v_acc_imag;
  vfloat32m1_t temp_v_i_real, temp_v_i_imag; 
  vfloat32m1_t temp_v_psum_real, temp_v_psum_imag;
  
  vfloat32m1_t temp_v_acc_t3_real, temp_v_acc_t3_imag;
  

  for (int ptr = 0; ptr < nr; ptr++) {
      //A0  
      temp_v_A_real = vle32_v_f32m1(&H_real[ptr][0][0], vl);
      temp_v_A_imag = vle32_v_f32m1(&H_imag[ptr][0][0], vl);
      temp_v_A_imag_conj = vfneg_v_f32m1(temp_v_A_imag, vl);

      vcfmul_vv_f32m1_sim(temp_v_A_real, temp_v_A_imag_conj, temp_v_A_real, temp_v_A_imag, &temp_v_psum_real, &temp_v_psum_imag, vl);
      
      if (ptr == 0){
        temp_v_psum_real = vfadd_vf_f32m1(temp_v_psum_real, sigma_n2, vl);
        //temp_v_psum_imag = vfadd_vf_f32m1(temp_v_psum_imag, sigma_n2, vl);
        vse32_v_f32m1(&Gram_real[0][0][0], temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_imag[0][0][0], temp_v_psum_imag, vl);
      } else {
        temp_v_acc_real = vle32_v_f32m1(&Gram_real[0][0][0], vl);
        temp_v_acc_t3_real = vfadd_vv_f32m1(temp_v_acc_real, temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_real[0][0][0], temp_v_acc_t3_real, vl);

        temp_v_acc_imag = vle32_v_f32m1(&Gram_imag[0][0][0], vl);
        temp_v_acc_t3_imag = vfadd_vv_f32m1(temp_v_acc_imag, temp_v_psum_imag, vl);
        vse32_v_f32m1(&Gram_imag[0][0][0], temp_v_acc_t3_imag, vl);
      }

      //B0
      temp_v_B_real = vle32_v_f32m1(&H_real[ptr][1][0], vl);
      temp_v_B_imag = vle32_v_f32m1(&H_imag[ptr][1][0], vl);
      temp_v_B_imag_conj = vfneg_v_f32m1(temp_v_B_imag, vl);

      vcfmul_vv_f32m1_sim(temp_v_B_real, temp_v_B_imag_conj, temp_v_A_real, temp_v_A_imag, &temp_v_psum_real, &temp_v_psum_imag, vl);
      
      if (ptr == 0){
        vse32_v_f32m1(&Gram_real[1][0][0], temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_imag[1][0][0], temp_v_psum_imag, vl);
      } else {
        temp_v_acc_real = vle32_v_f32m1(&Gram_real[1][0][0], vl);
        temp_v_acc_t3_real = vfadd_vv_f32m1(temp_v_acc_real, temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_real[1][0][0], temp_v_acc_t3_real, vl);

        temp_v_acc_imag = vle32_v_f32m1(&Gram_imag[1][0][0], vl);
        temp_v_acc_t3_imag = vfadd_vv_f32m1(temp_v_acc_imag, temp_v_psum_imag, vl);
        vse32_v_f32m1(&Gram_imag[1][0][0], temp_v_acc_t3_imag, vl);
      }

      //C0
      temp_v_C_real = vle32_v_f32m1(&H_real[ptr][2][0], vl);
      temp_v_C_imag = vle32_v_f32m1(&H_imag[ptr][2][0], vl);
      temp_v_C_imag_conj = vfneg_v_f32m1(temp_v_C_imag, vl);

      vcfmul_vv_f32m1_sim(temp_v_C_real, temp_v_C_imag_conj, temp_v_A_real, temp_v_A_imag, &temp_v_psum_real, &temp_v_psum_imag, vl);

      if (ptr == 0){
        vse32_v_f32m1(&Gram_real[2][0][0], temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_imag[2][0][0], temp_v_psum_imag, vl);
      } else {
        temp_v_acc_real = vle32_v_f32m1(&Gram_real[2][0][0], vl);
        temp_v_acc_t3_real = vfadd_vv_f32m1(temp_v_acc_real, temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_real[2][0][0], temp_v_acc_t3_real, vl);

        temp_v_acc_imag = vle32_v_f32m1(&Gram_imag[2][0][0], vl);
        temp_v_acc_t3_imag = vfadd_vv_f32m1(temp_v_acc_imag, temp_v_psum_imag, vl);
        vse32_v_f32m1(&Gram_imag[2][0][0], temp_v_acc_t3_imag, vl);
      }


      //D0
      temp_v_D_real = vle32_v_f32m1(&H_real[ptr][3][0], vl);
      temp_v_D_imag = vle32_v_f32m1(&H_imag[ptr][3][0], vl);
      temp_v_D_imag_conj = vfneg_v_f32m1(temp_v_D_imag, vl);

      vcfmul_vv_f32m1_sim(temp_v_D_real, temp_v_D_imag_conj, temp_v_A_real, temp_v_A_imag, &temp_v_psum_real, &temp_v_psum_imag, vl);

      if (ptr == 0){
        vse32_v_f32m1(&Gram_real[3][0][0], temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_imag[3][0][0], temp_v_psum_imag, vl);
      } else {
        temp_v_acc_real = vle32_v_f32m1(&Gram_real[3][0][0], vl);
        temp_v_acc_t3_real = vfadd_vv_f32m1(temp_v_acc_real, temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_real[3][0][0], temp_v_acc_t3_real, vl);

        temp_v_acc_imag = vle32_v_f32m1(&Gram_imag[3][0][0], vl);
        temp_v_acc_t3_imag = vfadd_vv_f32m1(temp_v_acc_imag, temp_v_psum_imag, vl);
        vse32_v_f32m1(&Gram_imag[3][0][0], temp_v_acc_t3_imag, vl);
      }
      

      //B1 
      vcfmul_vv_f32m1_sim(temp_v_B_real, temp_v_B_imag_conj, temp_v_B_real, temp_v_B_imag, &temp_v_psum_real, &temp_v_psum_imag, vl);

      if (ptr == 0){
        temp_v_psum_real = vfadd_vf_f32m1(temp_v_psum_real, sigma_n2, vl);
        //temp_v_psum_imag = vfadd_vf_f32m1(temp_v_psum_imag, sigma_n2, vl);

        vse32_v_f32m1(&Gram_real[1][1][0], temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_imag[1][1][0], temp_v_psum_imag, vl);
      } else {
        temp_v_acc_real = vle32_v_f32m1(&Gram_real[1][1][0], vl);
        temp_v_acc_t3_real = vfadd_vv_f32m1(temp_v_acc_real, temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_real[1][1][0], temp_v_acc_t3_real, vl);

        temp_v_acc_imag = vle32_v_f32m1(&Gram_imag[1][1][0], vl);
        temp_v_acc_t3_imag = vfadd_vv_f32m1(temp_v_acc_imag, temp_v_psum_imag, vl);
        vse32_v_f32m1(&Gram_imag[1][1][0], temp_v_acc_t3_imag, vl);
      }

      //C1
      vcfmul_vv_f32m1_sim(temp_v_C_real, temp_v_C_imag_conj, temp_v_B_real, temp_v_B_imag, &temp_v_psum_real, &temp_v_psum_imag, vl);
      if (ptr == 0){
        vse32_v_f32m1(&Gram_real[2][1][0], temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_imag[2][1][0], temp_v_psum_imag, vl);
      } else {
        temp_v_acc_real = vle32_v_f32m1(&Gram_real[2][1][0], vl);
        temp_v_acc_t3_real = vfadd_vv_f32m1(temp_v_acc_real, temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_real[2][1][0], temp_v_acc_t3_real, vl);

        temp_v_acc_imag = vle32_v_f32m1(&Gram_imag[2][1][0], vl);
        temp_v_acc_t3_imag = vfadd_vv_f32m1(temp_v_acc_imag, temp_v_psum_imag, vl);
        vse32_v_f32m1(&Gram_imag[2][1][0], temp_v_acc_t3_imag, vl);
      }

      //D1
      vcfmul_vv_f32m1_sim(temp_v_D_real, temp_v_D_imag_conj, temp_v_B_real, temp_v_B_imag, &temp_v_psum_real, &temp_v_psum_imag, vl);

      //still here
      if (ptr == 0){
        vse32_v_f32m1(&Gram_real[3][1][0], temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_imag[3][1][0], temp_v_psum_imag, vl);
      } else {
        temp_v_acc_real = vle32_v_f32m1(&Gram_real[3][1][0], vl);
        temp_v_acc_t3_real = vfadd_vv_f32m1(temp_v_acc_real, temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_real[3][1][0], temp_v_acc_t3_real, vl);

        temp_v_acc_imag = vle32_v_f32m1(&Gram_imag[3][1][0], vl);
        temp_v_acc_t3_imag = vfadd_vv_f32m1(temp_v_acc_imag, temp_v_psum_imag, vl);
        vse32_v_f32m1(&Gram_imag[3][1][0], temp_v_acc_t3_imag, vl);
      }

      //C2
      vcfmul_vv_f32m1_sim(temp_v_C_real, temp_v_C_imag_conj, temp_v_C_real, temp_v_C_imag, &temp_v_psum_real, &temp_v_psum_imag, vl);
      //still here
      if (ptr == 0){
        temp_v_psum_real = vfadd_vf_f32m1(temp_v_psum_real, sigma_n2, vl);
        //temp_v_psum_imag = vfadd_vf_f32m1(temp_v_psum_imag, sigma_n2, vl);

        vse32_v_f32m1(&Gram_real[2][2][0], temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_imag[2][2][0], temp_v_psum_imag, vl);
      } else {
        temp_v_acc_real = vle32_v_f32m1(&Gram_real[2][2][0], vl);
        temp_v_acc_t3_real = vfadd_vv_f32m1(temp_v_acc_real, temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_real[2][2][0], temp_v_acc_t3_real, vl);

        temp_v_acc_imag = vle32_v_f32m1(&Gram_imag[2][2][0], vl);
        temp_v_acc_t3_imag = vfadd_vv_f32m1(temp_v_acc_imag, temp_v_psum_imag, vl);
        vse32_v_f32m1(&Gram_imag[2][2][0], temp_v_acc_t3_imag, vl);
      }



      //D2
      vcfmul_vv_f32m1_sim(temp_v_D_real, temp_v_D_imag_conj, temp_v_C_real, temp_v_C_imag, &temp_v_psum_real, &temp_v_psum_imag, vl);

       //still here
      if (ptr == 0){
        vse32_v_f32m1(&Gram_real[3][2][0], temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_imag[3][2][0], temp_v_psum_imag, vl);
      } else {
        temp_v_acc_real = vle32_v_f32m1(&Gram_real[3][2][0], vl);
        temp_v_acc_t3_real = vfadd_vv_f32m1(temp_v_acc_real, temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_real[3][2][0], temp_v_acc_t3_real, vl);

        temp_v_acc_imag = vle32_v_f32m1(&Gram_imag[3][2][0], vl);
        temp_v_acc_t3_imag = vfadd_vv_f32m1(temp_v_acc_imag, temp_v_psum_imag, vl);
        vse32_v_f32m1(&Gram_imag[3][2][0], temp_v_acc_t3_imag, vl);
      }


      //D3

      vcfmul_vv_f32m1_sim(temp_v_D_real, temp_v_D_imag_conj, temp_v_D_real, temp_v_D_imag, &temp_v_psum_real, &temp_v_psum_imag, vl);

      if (ptr == 0){
        temp_v_psum_real = vfadd_vf_f32m1(temp_v_psum_real, sigma_n2, vl);
        //temp_v_psum_imag = vfadd_vf_f32m1(temp_v_psum_imag, sigma_n2, vl);
        vse32_v_f32m1(&Gram_real[3][3][0], temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_imag[3][3][0], temp_v_psum_imag, vl);
      } else {
        temp_v_acc_real = vle32_v_f32m1(&Gram_real[3][3][0], vl);
        temp_v_acc_t3_real = vfadd_vv_f32m1(temp_v_acc_real, temp_v_psum_real, vl);
        vse32_v_f32m1(&Gram_real[3][3][0], temp_v_acc_t3_real, vl);

        temp_v_acc_imag = vle32_v_f32m1(&Gram_imag[3][3][0], vl);
        temp_v_acc_t3_imag = vfadd_vv_f32m1(temp_v_acc_imag, temp_v_psum_imag, vl);
        vse32_v_f32m1(&Gram_imag[3][3][0], temp_v_acc_t3_imag, vl);
      }
  } 
}

