#ifndef LIB_H_
#define LIB_H_

#include <stdint.h>
#include <string.h>

#define GET_IDX(i, j, k, nr, n_subcarier)  (((i) * (nr) + (j)) * (n_subcarier) + (k))

static inline void __attribute__((always_inline)) vcwmul_vv_i32m2_sim(
    vint16m1_t ar, vint16m1_t ai,
    vint16m1_t br, vint16m1_t bi,
    vint32m2_t *cr,
    vint32m2_t *ci,
    size_t vl)
{
    // real: ar*br - ai*bi
    *cr = vsub_vv_i32m2(vwmul_vv_i32m2(ar, br, vl), vwmul_vv_i32m2(ai, bi, vl), vl);   // r -= ai*bi
    // imag: ar*bi + ai*br
    *ci = vadd_vv_i32m2(vwmul_vv_i32m2(ar, bi, vl), vwmul_vv_i32m2(ai, br, vl), vl);    // i += ai*br
};

static inline void __attribute__((always_inline)) vcwmacc_vv_i32m2_sim(  
    //c = a * b + p0  
    vint16m1_t a_real,
    vint16m1_t a_imag,
    vint16m1_t b_real,
    vint16m1_t b_imag,
    vint32m2_t p0_real,
    vint32m2_t p0_imag,
    vint32m2_t *c_real,
    vint32m2_t *c_imag,
    size_t vl){

    vint32m2_t v0_real, v0_imag;
    vcwmul_vv_i32m2_sim(a_real, a_imag, b_real, b_imag, &v0_real, &v0_imag, vl);
        
    *c_real = vadd_vv_i32m2(p0_real, v0_real, vl);
    *c_imag = vadd_vv_i32m2(p0_imag, v0_imag, vl);
};




void v_compute_gram_cfxp16(
  size_t nt, 
  size_t nr, 
  size_t vl,
  size_t offset,
  size_t n_subcarrier,
  int16_t sigma_n2, 
  int16_t  *H_real,
  int16_t  *H_imag,
  int16_t  *Gram_real, 
  int16_t  *Gram_imag); 

void v_compute_gram_cfxp16_unscaled(
  size_t nt, 
  size_t nr, 
  size_t vl,
  size_t offset,
  size_t n_subcarrier,
  int16_t sigma_n2, 
  int16_t  *H_real,
  int16_t  *H_imag,
  int32_t  *Gram_real, 
  int32_t  *Gram_imag);







#endif

