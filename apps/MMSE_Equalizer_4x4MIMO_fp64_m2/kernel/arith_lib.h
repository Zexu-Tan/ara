#ifndef _ARITH_LIB_H_
#define _ARITH_LIB_H_

#include <stdint.h>
#include <string.h>




#define GET_IDX(i, j, k, nr, vl)  (((i) * (nr) + (j)) * (vl) + (k))
//#define GET_IDX(i, j, k, nr, vl)  (((i) * (nr) + (j)) * (vl) + (k))
//i,j  k:subcarriers, nr: receiver size in MIMO, vl





//void v_compute_gram_fp32(size_t nt, size_t nr, size_t vl, float sigma_n2, float H_real[nt][nr][vl], float H_imag[nt][nr][vl], float Gram_real[nt][nr][vl], float Gram_imag[nt][nr][vl]);
void v_compute_gram_fp32(
  size_t nt, 
  size_t nr, 
  size_t vl, 
  double sigma_n2, 
  double *H_real,
  double *H_imag,
  double *Gram_real, 
  double *Gram_imag);
  
//void vfctrans_vv_f32m1_sim(vfloat64m1_t a_real, vfloat64m1_t a_imag, size_t vl);


#endif

