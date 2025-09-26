#ifndef LIB_H_
#define LIB_H_

#include <stdint.h>
#include <string.h>

#define GET_IDX(i, j, k, nr, n_subcarier)  (((i) * (nr) + (j)) * (n_subcarier) + (k))



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

void v_LL_decomp_cfxp16(  
  size_t nt, 
  size_t nr, 
  size_t vl,
  size_t offset,
  size_t n_subcarrier,
  int16_t *Gram_real,
  int16_t *Gram_imag,
  int16_t *L_real,
  int16_t *L_imag
);


void v_compute_z_cfxp16(
    size_t nt, 
    size_t nr, 
    size_t vl,
    size_t offset,
    size_t n_subcarrier,
    int16_t *H_real,
    int16_t *H_imag,
    int16_t *Y_real,
    int16_t *Y_imag,
    int16_t *z_real,
    int16_t *z_imag); 



void v_forward_substitution_cfxp16(
    size_t nt, 
    size_t nr, 
    size_t vl,
    size_t offset,
    size_t n_subcarrier,
    int16_t *z_real,
    int16_t *z_imag,
    int16_t *L_real,
    int16_t *L_imag,
    int16_t *u_real,
    int16_t *u_imag
);


#endif

