#include <riscv_vector.h>
#include "../lib.h"

static inline vint16m1_t vfxrsqrt_v_i16m1(vint16m1_t a, vint16m1_t t, size_t vl) {
  return vdiv_vv_i16m1(t, a, vl); //Imitate the fixed point reciprocal square root 
}

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
){
  vint16m1_t v1 = vmv_v_x_i16m1(1024, vl);

  //L_00
  vint16m1_t L00_real = vle16_v_i16m1(&Gram_real[GET_IDX(0, 0, offset, nr, n_subcarrier)], vl);
  L00_real = vfxrsqrt_v_i16m1(L00_real, v1, vl);
  vse16_v_i16m1(&L_real[GET_IDX(0, 0, offset, nr, n_subcarrier)], L00_real, vl);
  

  //L_10
  vint16m1_t L10_real = vle16_v_i16m1(&Gram_real[GET_IDX(1, 0, offset, nr, n_subcarrier)],vl);

  L10_real = vnclip_wx_i16m1(vwmul_vv_i32m2(L10_real, L00_real, vl), 10, vl);

  vint16m1_t L10_imag = vle16_v_i16m1(&Gram_imag[GET_IDX(1, 0, offset, nr, n_subcarrier)],vl);

  L10_imag = vnclip_wx_i16m1(vwmul_vv_i32m2(L10_imag, L00_real, vl), 10, vl);

  vse16_v_i16m1(&L_real[GET_IDX(1, 0, offset, nr, n_subcarrier)], L10_real, vl);
  vse16_v_i16m1(&L_imag[GET_IDX(1, 0, offset, nr, n_subcarrier)], L10_imag, vl);
  L10_imag = vneg_v_i16m1(L10_imag, vl);



  //L_20
  vint16m1_t L20_real = vle16_v_i16m1(&Gram_real[GET_IDX(2, 0, offset, nr, n_subcarrier)],vl);

  L20_real = vnclip_wx_i16m1(vwmul_vv_i32m2(L20_real, L00_real, vl), 10, vl);

  vint16m1_t L20_imag = vle16_v_i16m1(&Gram_imag[GET_IDX(2, 0, offset, nr, n_subcarrier)],vl);

  L20_imag = vnclip_wx_i16m1(vwmul_vv_i32m2(L20_imag, L00_real, vl), 10, vl);

  vse16_v_i16m1(&L_real[GET_IDX(2, 0, offset, nr, n_subcarrier)], L20_real, vl);
  vse16_v_i16m1(&L_imag[GET_IDX(2, 0, offset, nr, n_subcarrier)], L20_imag, vl);

  //L_30
  vint16m1_t L30_real = vle16_v_i16m1(&Gram_real[GET_IDX(3, 0, offset, nr, n_subcarrier)],vl);

  L30_real = vnclip_wx_i16m1(vwmul_vv_i32m2(L30_real, L00_real, vl), 10, vl);

  vint16m1_t L30_imag = vle16_v_i16m1(&Gram_imag[GET_IDX(3, 0, offset, nr, n_subcarrier)],vl);

  L30_imag = vnclip_wx_i16m1(vwmul_vv_i32m2(L30_imag, L00_real, vl), 10, vl);

  vse16_v_i16m1(&L_real[GET_IDX(3, 0, offset, nr, n_subcarrier)], L30_real, vl);
  vse16_v_i16m1(&L_imag[GET_IDX(3, 0, offset, nr, n_subcarrier)], L30_imag, vl);
  

  //L_11_real
  vint16m1_t L11_real = vle16_v_i16m1(&Gram_real[GET_IDX(1, 1, offset, nr, n_subcarrier)],vl);

  //vint32m2_t acc_L11_real = vsll_vx_i32m2(vwadd_vx_i32m2(L11_real, vl), 10, vl);
  
  vint32m2_t acc_L11_real = vsll_vx_i32m2(vwadd_vx_i32m2(L11_real, 0, vl), 10, vl);
  

  acc_L11_real = vsll_vx_i32m2(acc_L11_real, 10, vl);

  acc_L11_real = vsub_vv_i32m2(acc_L11_real, vwmul_vv_i32m2(L10_real, L10_real, vl), vl);
  acc_L11_real = vsub_vv_i32m2(acc_L11_real, vwmul_vv_i32m2(L10_imag, L10_imag, vl), vl);

          vint16m1_t L21_real = vle16_v_i16m1(&Gram_real[GET_IDX(2, 1, offset, nr, n_subcarrier)],vl);

  L11_real = vfxrsqrt_v_i16m1(vnclip_wx_i16m1(acc_L11_real, 10, vl), v1, vl);

  vse16_v_i16m1(&L_real[GET_IDX(1, 1, offset, nr, n_subcarrier)], L11_real, vl);

  
 
  //L_21
  vint32m2_t acc_L21_real = vsll_vx_i32m2(vwadd_vx_i32m2(L21_real,0, vl), 10 ,vl);
  acc_L21_real = vsub_vv_i32m2(acc_L21_real, vwmul_vv_i32m2(L20_real, L10_real, vl), vl);
  vint16m1_t L21_imag = vle16_v_i16m1(&Gram_imag[GET_IDX(2, 1, offset, nr, n_subcarrier)],vl);
  acc_L21_real = vwmacc_vv_i32m2(acc_L21_real, L20_imag, L10_imag, vl);

  vint32m2_t acc_L21_imag = vsll_vx_i32m2(vwadd_vx_i32m2(L21_imag,0, vl), 10 ,vl);
  acc_L21_imag = vsub_vv_i32m2(acc_L21_imag, vwmul_vv_i32m2(L20_real, L10_imag, vl), vl);

      vint16m1_t L31_real = vle16_v_i16m1(&Gram_real[GET_IDX(3, 1, offset, nr, n_subcarrier)],vl);

  acc_L21_imag = vsub_vv_i32m2(acc_L21_imag, vwmul_vv_i32m2(L20_imag, L10_real, vl), vl);

  acc_L21_real = vwmul_vv_i32m2(vnclip_wx_i16m1(acc_L21_real, 10, vl), L11_real, vl);

  L21_real = vnclip_wx_i16m1(acc_L21_real, 10, vl);

  vse16_v_i16m1(&L_real[GET_IDX(2, 1, offset, nr, n_subcarrier)], L21_real, vl);

  acc_L21_imag = vwmul_vv_i32m2(vnclip_wx_i16m1(acc_L21_imag, 10, vl), L11_real, vl);

  L21_imag = vnclip_wx_i16m1(acc_L21_imag, 10, vl);
  vse16_v_i16m1(&L_imag[GET_IDX(2, 1, offset, nr, n_subcarrier)], L21_imag, vl);


 
  //L_31

  vint32m2_t acc_L31_real = vsll_vx_i32m2(vwadd_vx_i32m2(L31_real,0, vl), 10, vl);
  acc_L31_real = vsub_vv_i32m2(acc_L31_real, vwmul_vv_i32m2(L30_real, L10_real, vl), vl);

  vint16m1_t L31_imag = vle16_v_i16m1(&Gram_imag[GET_IDX(3, 1, offset, nr, n_subcarrier)],vl);
  acc_L31_real = vwmacc_vv_i32m2(acc_L31_real, L30_imag, L10_imag, vl);

  vint32m2_t acc_L31_imag = vsll_vx_i32m2(vwadd_vx_i32m2(L31_imag,0, vl), 10, vl);
  acc_L31_imag = vsub_vv_i32m2(acc_L31_imag, vwmul_vv_i32m2(L30_real, L10_imag,vl), vl);
  acc_L31_imag = vsub_vv_i32m2(acc_L31_imag, vwmul_vv_i32m2(L30_imag, L10_real,vl), vl);

  acc_L31_real = vwmul_vv_i32m2(vnclip_wx_i16m1(acc_L31_real, 10, vl), L11_real, vl);
  L31_real = vnclip_wx_i16m1(acc_L31_real, 10, vl);

  vse16_v_i16m1(&L_real[GET_IDX(3, 1, offset, nr, n_subcarrier)], L31_real, vl);

  acc_L31_imag = vwmul_vv_i32m2(vnclip_wx_i16m1(acc_L31_imag, 10, vl), L11_real, vl);
  L31_imag = vnclip_wx_i16m1(acc_L31_imag,  10, vl);
  vse16_v_i16m1(&L_imag[GET_IDX(3, 1, offset, nr, n_subcarrier)], L31_imag, vl);

  L20_imag = vneg_v_i16m1(L20_imag, vl);
  L21_imag = vneg_v_i16m1(L21_imag, vl);
  

  //L_22_real
  vint16m1_t L22_real = vle16_v_i16m1(&Gram_real[GET_IDX(2, 2, offset, nr, n_subcarrier)],vl);

  vint32m2_t acc_L22_real = vsll_vx_i32m2(vwadd_vx_i32m2(L22_real, 0, vl), 10, vl);

  acc_L22_real = vsub_vv_i32m2(acc_L22_real, vwmul_vv_i32m2(L20_real, L20_real, vl), vl);
  acc_L22_real = vsub_vv_i32m2(acc_L22_real, vwmul_vv_i32m2(L20_imag, L20_imag, vl), vl);

  vint16m1_t L32_real = vle16_v_i16m1(&Gram_real[GET_IDX(3, 2, offset, nr, n_subcarrier)], vl);

  acc_L22_real = vsub_vv_i32m2(acc_L22_real, vwmul_vv_i32m2(L21_real, L21_real, vl), vl);
  acc_L22_real = vsub_vv_i32m2(acc_L22_real, vwmul_vv_i32m2(L21_imag, L21_imag, vl), vl);

  L22_real = vfxrsqrt_v_i16m1(vnclip_wx_i16m1(acc_L22_real, 10, vl), v1, vl);
  vse16_v_i16m1(&L_real[GET_IDX(2, 2, offset, nr, n_subcarrier)], L22_real, vl);

  

  



  //L_32
  
  vint32m2_t acc_L32_real = vsll_vx_i32m2(vwadd_vx_i32m2(L32_real,0, vl), 10, vl);
  acc_L32_real = vsub_vv_i32m2(acc_L32_real, vwmul_vv_i32m2(L30_real, L20_real, vl), vl);
  vint16m1_t L32_imag = vle16_v_i16m1(&Gram_imag[GET_IDX(3, 2, offset, nr, n_subcarrier)], vl);
  vint32m2_t acc_L32_imag = vsll_vx_i32m2(vwadd_vx_i32m2(L32_imag,0, vl), 10, vl);
  acc_L32_real = vwmacc_vv_i32m2(acc_L32_real, L30_imag, L20_imag, vl);

  acc_L32_imag = vsub_vv_i32m2(acc_L32_imag, vwmul_vv_i32m2(L30_real, L20_imag, vl), vl);
  acc_L32_imag = vsub_vv_i32m2(acc_L32_imag, vwmul_vv_i32m2(L30_imag, L20_real, vl), vl);

  acc_L32_real = vsub_vv_i32m2(acc_L32_real, vwmul_vv_i32m2(L31_real, L21_real, vl), vl);
  acc_L32_real = vadd_vv_i32m2(acc_L32_real, vwmul_vv_i32m2(L31_imag, L21_imag, vl), vl);

  vint16m1_t L33_real = vle16_v_i16m1(&Gram_real[GET_IDX(3, 3, offset, nr, n_subcarrier )],vl);
  

  acc_L32_imag = vsub_vv_i32m2(acc_L32_imag, vwmul_vv_i32m2(L31_real, L21_imag, vl), vl);
  acc_L32_imag = vsub_vv_i32m2(acc_L32_imag, vwmul_vv_i32m2(L31_imag, L21_real, vl), vl);

  acc_L32_real = vwmul_vv_i32m2(vnclip_wx_i16m1(acc_L32_real, 10, vl), L22_real, vl);

  L32_real = vnclip_wx_i16m1(acc_L32_real,  10, vl);

  vse16_v_i16m1(&L_real[GET_IDX(3, 2, offset, nr, n_subcarrier)], L32_real, vl);

  acc_L32_imag = vwmul_vv_i32m2(vnclip_wx_i16m1(acc_L32_imag, 10, vl), L22_real, vl);

  L32_imag = vnclip_wx_i16m1(acc_L32_imag,  10, vl);

  vse16_v_i16m1(&L_imag[GET_IDX(3, 2, offset, nr, n_subcarrier)], L32_imag, vl);



  //L_33_real
  vint32m2_t acc_L33_real = vsll_vx_i32m2(vwadd_vx_i32m2(L33_real, 0, vl), 10, vl);
  acc_L33_real = vsub_vv_i32m2(acc_L33_real, vwmul_vv_i32m2(L30_real, L30_real, vl), vl);
  acc_L33_real = vsub_vv_i32m2(acc_L33_real, vwmul_vv_i32m2(L30_imag, L30_imag, vl), vl);
  acc_L33_real = vsub_vv_i32m2(acc_L33_real, vwmul_vv_i32m2(L31_real, L31_real, vl), vl);
  acc_L33_real = vsub_vv_i32m2(acc_L33_real, vwmul_vv_i32m2(L31_imag, L31_imag, vl), vl);
  acc_L33_real = vsub_vv_i32m2(acc_L33_real, vwmul_vv_i32m2(L32_real, L32_real, vl), vl);
  acc_L33_real = vsub_vv_i32m2(acc_L33_real, vwmul_vv_i32m2(L32_imag, L32_imag, vl), vl);

  L33_real = vfxrsqrt_v_i16m1(vnclip_wx_i16m1(acc_L33_real, 10, vl), v1, vl);
  vse16_v_i16m1(&L_real[GET_IDX(3, 3, offset, nr, n_subcarrier)], L33_real, vl);

}

