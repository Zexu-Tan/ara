#ifndef _ARITH_LIB_H_
#define _ARITH_LIB_H_

#include <stdint.h>
#include <string.h>


#define VCFMUL_VV_SIM(ar, ai, br, bi, cr, ci, vl)        \
  asm volatile (                                            \
    "vsetvli zero, %0, e32,m1\n"                            \
    "vfmul.vv " #cr ", " #ar ", " #br "\n"                  \
    "vfmul.vv v1, "   #ai ", " #bi "\n"                     \
    "vfsub.vv " #cr ", " #cr ", v1\n"                       \
    "vfmul.vv " #ci ", " #ar ", " #bi "\n"                  \
    "vfmul.vv v1, "   #ai ", " #br "\n"                     \
    "vfadd.vv " #ci ", " #ci ", v1\n"                       \
    : /* no outputs */                                       \
    : "r"(vl)                                               \
    : "v1"                                                  \
  )

#define GET_IDX(i, j, k, nr, vl)  (((i) * (nr) + (j)) * (vl) + (k))
//#define GET_IDX(i, j, k, nr, vl)  (((i) * (nr) + (j)) * (vl) + (k))
//i,j  k:subcarriers, nr: receiver size in MIMO, vl


#define CASE 0

void vcfmul_vv_f32m1_sim(
    vfloat32m1_t ar, vfloat32m1_t ai,
    vfloat32m1_t br, vfloat32m1_t bi,
    vfloat32m1_t *cr,
    vfloat32m1_t *ci,
    size_t vl);


//void v_compute_gram_fp32(size_t nt, size_t nr, size_t vl, float sigma_n2, float H_real[nt][nr][vl], float H_imag[nt][nr][vl], float Gram_real[nt][nr][vl], float Gram_imag[nt][nr][vl]);
void v_compute_gram_fp32(
  size_t nt, 
  size_t nr, 
  size_t vl, 
  float sigma_n2, 
  float *H_real,
  float *H_imag,
  float *Gram_real, 
  float *Gram_imag);
  
//void vfctrans_vv_f32m1_sim(vfloat32m1_t a_real, vfloat32m1_t a_imag, size_t vl);

void vcsqrdmag_vv_f32m1_sim(vfloat32m1_t vr_ar, vfloat32m1_t vr_ai, vfloat32m1_t *vr_cr, size_t vl);

void v_LL_decomp_f32(  
  size_t nt, 
  size_t nr, 
  size_t vl,
  float *Gram_real,
  float *Gram_imag,
  float *L_real,
  float *L_imag

);
void vcfnmsac_vv_f32m1_sim(  
    //c = -a*b + p0
    vfloat32m1_t a_real,
    vfloat32m1_t a_imag,
    vfloat32m1_t b_real,
    vfloat32m1_t b_imag,
    vfloat32m1_t p0_real,
    vfloat32m1_t p0_imag,
    vfloat32m1_t *c_real,
    vfloat32m1_t *c_imag,
    size_t vl);

//void v_LL_decomp_f32(size_t nt, size_t nr, size_t vl, float Gram_real[nt][nr][vl], float Gram_imag[nt][nr][vl], float L_real[nt][nr][vl], float L_imag[nt][nr][vl]);

void vcfmac_vv_f32m1_sim(vfloat32m1_t a_real,vfloat32m1_t a_imag,vfloat32m1_t b_real,vfloat32m1_t b_imag,vfloat32m1_t p0_real,vfloat32m1_t p0_imag,vfloat32m1_t *c_real,vfloat32m1_t *c_imag,size_t vl);

void v_compute_z(size_t nt, size_t nr, size_t vl, float *H_real,float *H_imag,float *Y_real,float *Y_imag,float *z_real,float *z_imag);

void v_forward_substitution(size_t nt, size_t nr, size_t vl, float *z_real,float *z_imag,float *L_real,float *L_imag,float *u_real,float *u_imag);

void v_backward_substitution(size_t nt, size_t nr, size_t vl, float *u_real,float *u_imag,float *L_real,float *L_imag,float *x_real,float *x_imag);


#endif

