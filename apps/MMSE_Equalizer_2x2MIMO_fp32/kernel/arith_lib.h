#ifndef _ARITH_LIB_H_
#define _ARITH_LIB_H_

#include <stdint.h>
#include <string.h>

#define GET_IDX(i, j, k, nr, vl)  (((i) * (nr) + (j)) * (vl) + (k))
//#define GET_IDX(i, j, k, nr, vl)  (((i) * (nr) + (j)) * (vl) + (k))
//i,j  k:subcarriers, nr: receiver size in MIMO, vl


void v_compute_gram_fp32(
  size_t nt, 
  size_t nr, 
  size_t vl, 
  float sigma_n2, 
  float *H_real,
  float *H_imag,
  float *Gram_real, 
  float *Gram_imag);
  

void v_LL_decomp_fp32(  
  size_t nt, 
  size_t nr, 
  size_t vl,
  float *Gram_real,
  float *Gram_imag,
  float *L_real,
  float *L_imag

);


void v_compute_z_fp32(size_t nt, size_t nr, size_t vl, float *H_real,float *H_imag,float *Y_real,float *Y_imag,float *z_real,float *z_imag);

void v_forward_substitution(size_t nt, size_t nr, size_t vl, float *z_real,float *z_imag,float *L_real,float *L_imag,float *u_real,float *u_imag);

void v_backward_substitution(size_t nt, size_t nr, size_t vl, float *u_real,float *u_imag,float *L_real,float *L_imag,float *x_real,float *x_imag);


static inline void vcfnmsac_vv_f32m1_sim(  
    //c = -a*b + p0
    vfloat32m1_t a_real,
    vfloat32m1_t a_imag,
    vfloat32m1_t b_real,
    vfloat32m1_t b_imag,
    vfloat32m1_t p0_real,
    vfloat32m1_t p0_imag,
    vfloat32m1_t *c_real,
    vfloat32m1_t *c_imag,
    size_t vl){
    

    *c_real = vfnmsac_vv_f32m1(p0_real, a_real, b_real, vl);
    *c_real = vfmacc_vv_f32m1(*c_real, a_imag, b_imag, vl);

    *c_imag = vfnmsac_vv_f32m1(p0_imag, a_real, b_imag, vl);
    *c_imag = vfnmsac_vv_f32m1(*c_imag, a_imag, b_real, vl);

    
}







#endif

