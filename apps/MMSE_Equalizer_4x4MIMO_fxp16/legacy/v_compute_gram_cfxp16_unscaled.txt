#include <riscv_vector.h>
#include "../lib.h"


#define CASE 3

//0 v1 output unscaled
//1 v2 output unscaled


#if CASE == 0

void v_compute_gram_cfxp16_unscaled(
  size_t nt, 
  size_t nr, 
  size_t vl,
  size_t offset,
  size_t n_subcarrier,
  int16_t sigma_n2, 
  int16_t  *H_real,
  int16_t  *H_imag,
  int16_t  *Gram_real, 
  int16_t  *Gram_imag) {

    //Gram_real
    vint32m2_t acc_G_00_real = vmv_v_x_i32m2(0, vl);
    vint32m2_t acc_G_10_real = vmv_v_x_i32m2(0, vl);
    vint32m2_t acc_G_20_real = vmv_v_x_i32m2(0, vl);
    vint32m2_t acc_G_30_real = vmv_v_x_i32m2(0, vl);

    vint32m2_t acc_G_11_real = vmv_v_x_i32m2(0, vl);
    vint32m2_t acc_G_21_real = vmv_v_x_i32m2(0, vl);
    vint32m2_t acc_G_31_real = vmv_v_x_i32m2(0, vl);

    vint32m2_t acc_G_22_real = vmv_v_x_i32m2(0, vl);
    vint32m2_t acc_G_32_real = vmv_v_x_i32m2(0, vl);

    vint32m2_t acc_G_33_real = vmv_v_x_i32m2(0, vl);

    //Gram_imag
    vint32m2_t acc_G_10_imag = vmv_v_x_i32m2(0, vl);
    vint32m2_t acc_G_20_imag = vmv_v_x_i32m2(0, vl);
    vint32m2_t acc_G_30_imag = vmv_v_x_i32m2(0, vl);

    vint32m2_t acc_G_21_imag = vmv_v_x_i32m2(0, vl);
    vint32m2_t acc_G_31_imag = vmv_v_x_i32m2(0, vl);

    vint32m2_t acc_G_32_imag = vmv_v_x_i32m2(0, vl);


    vint16m1_t t0_real, t0_imag;
    vint16m1_t t1_real, t1_imag;
    vint16m1_t t2_real, t2_imag;
    vint16m1_t t3_real, t3_imag;
    
    for (int ptr_row = 0; ptr_row < 3; ptr_row++){
      //G00
      t0_real = vle16_v_i16m1(&H_real[GET_IDX(ptr_row, 0, offset, nr, n_subcarrier)], vl);
      acc_G_00_real = vwmacc_vv_i32m2(acc_G_00_real, t0_real, t0_real, vl);

      t0_imag = vle16_v_i16m1(&H_imag[GET_IDX(ptr_row, 0, offset, nr, n_subcarrier)], vl);
      acc_G_00_real = vwmacc_vv_i32m2(acc_G_00_real, t0_imag, t0_imag, vl);
      

      //G10
      t1_real = vle16_v_i16m1(&H_real[GET_IDX(ptr_row, 1, offset, nr, n_subcarrier)], vl);
      acc_G_10_real = vwmacc_vv_i32m2(acc_G_10_real, t1_real, t0_real, vl);

      t1_imag = vle16_v_i16m1(&H_imag[GET_IDX(ptr_row, 1, offset, nr, n_subcarrier)], vl);
      acc_G_10_real = vwmacc_vv_i32m2(acc_G_10_real, t1_imag, t0_imag, vl);

      acc_G_10_imag = vwmacc_vv_i32m2(acc_G_10_imag, t1_real, t0_imag, vl);
      acc_G_10_imag = vsub_vv_i32m2(acc_G_10_imag, vwmul_vv_i32m2(t1_imag, t0_real, vl), vl);


      //G20
      t2_real = vle16_v_i16m1(&H_real[GET_IDX(ptr_row, 2, offset, nr, n_subcarrier)], vl);
      acc_G_20_real = vwmacc_vv_i32m2(acc_G_20_real, t2_real, t0_real, vl);

      t2_imag = vle16_v_i16m1(&H_imag[GET_IDX(ptr_row, 2, offset, nr, n_subcarrier)], vl);
      acc_G_20_real = vwmacc_vv_i32m2(acc_G_20_real, t2_imag, t0_imag, vl);

      acc_G_20_imag = vwmacc_vv_i32m2(acc_G_20_imag, t2_real, t0_imag, vl);
      acc_G_20_imag = vsub_vv_i32m2(acc_G_20_imag, vwmul_vv_i32m2(t2_imag, t0_real, vl), vl);


      //G30
      t3_real = vle16_v_i16m1(&H_real[GET_IDX(ptr_row, 3, offset, nr, n_subcarrier)], vl);
      acc_G_30_real = vwmacc_vv_i32m2(acc_G_30_real, t3_real, t0_real, vl);

      t3_imag = vle16_v_i16m1(&H_imag[GET_IDX(ptr_row, 3, offset, nr, n_subcarrier)], vl);
      acc_G_30_real = vwmacc_vv_i32m2(acc_G_30_real, t3_imag, t0_imag, vl);

      acc_G_30_imag = vwmacc_vv_i32m2(acc_G_30_imag, t3_real, t0_imag, vl);
      acc_G_30_imag = vsub_vv_i32m2(acc_G_30_imag, vwmul_vv_i32m2(t3_imag, t0_real, vl), vl);

      //G11
      acc_G_11_real = vwmacc_vv_i32m2(acc_G_11_real, t1_real, t1_real, vl);
      acc_G_11_real = vwmacc_vv_i32m2(acc_G_11_real, t1_imag, t1_imag, vl);

      
      //G21
      acc_G_21_real = vwmacc_vv_i32m2(acc_G_21_real, t2_real, t1_real, vl);
      acc_G_21_real = vwmacc_vv_i32m2(acc_G_21_real, t2_imag, t1_imag, vl);

      acc_G_21_imag = vwmacc_vv_i32m2(acc_G_21_imag, t2_real, t1_imag, vl);
      acc_G_21_imag = vsub_vv_i32m2(acc_G_21_imag, vwmul_vv_i32m2(t2_imag, t1_real, vl), vl);

      //G31
      acc_G_31_real = vwmacc_vv_i32m2(acc_G_31_real, t3_real, t1_real, vl);
      acc_G_31_real = vwmacc_vv_i32m2(acc_G_31_real, t3_imag, t1_imag, vl);

      acc_G_31_imag = vwmacc_vv_i32m2(acc_G_31_imag, t3_real, t1_imag, vl);
      acc_G_31_imag = vsub_vv_i32m2(acc_G_31_imag, vwmul_vv_i32m2(t3_imag, t1_real, vl), vl);

      //G22
      acc_G_22_real = vwmacc_vv_i32m2(acc_G_22_real, t2_real, t2_real, vl);
      acc_G_22_real = vwmacc_vv_i32m2(acc_G_22_real, t2_imag, t2_imag, vl);


      //G32
      acc_G_32_real = vwmacc_vv_i32m2(acc_G_32_real, t3_real, t2_real, vl);
      acc_G_32_real = vwmacc_vv_i32m2(acc_G_32_real, t3_imag, t2_imag, vl);

      acc_G_32_imag = vwmacc_vv_i32m2(acc_G_32_imag, t3_real, t2_imag, vl);
      acc_G_32_imag = vsub_vv_i32m2(acc_G_32_imag, vwmul_vv_i32m2(t3_imag, t2_real, vl), vl);


      //G33
      acc_G_33_real = vwmacc_vv_i32m2(acc_G_33_real, t3_real, t3_real, vl);
      acc_G_33_real = vwmacc_vv_i32m2(acc_G_33_real, t3_imag, t3_imag, vl);

    }

    
    //G00
    t0_real = vle16_v_i16m1(&H_real[GET_IDX(3, 0, offset, nr, n_subcarrier)], vl);
    acc_G_00_real = vwmacc_vv_i32m2(acc_G_00_real, t0_real, t0_real, vl);

    t0_imag = vle16_v_i16m1(&H_imag[GET_IDX(3, 0, offset, nr, n_subcarrier)], vl);
    acc_G_00_real = vwmacc_vv_i32m2(acc_G_00_real, t0_imag, t0_imag, vl);

    vint16m1_t G_00_real = vadd_vx_i16m1(vnclip_wx_i16m1(acc_G_00_real, 10, vl), sigma_n2, vl);
    vse16_v_i16m1(&Gram_real[GET_IDX(0,0,offset,nr, n_subcarrier)], G_00_real, vl);
      

    
    //G10
    t1_real = vle16_v_i16m1(&H_real[GET_IDX(3, 1, offset, nr, n_subcarrier)], vl);
    acc_G_10_real = vwmacc_vv_i32m2(acc_G_10_real, t1_real, t0_real, vl);

    t1_imag = vle16_v_i16m1(&H_imag[GET_IDX(3, 1, offset, nr, n_subcarrier)], vl);
    acc_G_10_real = vwmacc_vv_i32m2(acc_G_10_real, t1_imag, t0_imag, vl);
    vse16_v_i16m1(&Gram_real[GET_IDX(1,0,offset,nr, n_subcarrier)], vnclip_wx_i16m1(acc_G_10_real, 10, vl), vl);

    acc_G_10_imag = vwmacc_vv_i32m2(acc_G_10_imag, t1_real, t0_imag, vl);
    acc_G_10_imag = vsub_vv_i32m2(acc_G_10_imag, vwmul_vv_i32m2(t1_imag, t0_real, vl), vl);
    vse16_v_i16m1(&Gram_imag[GET_IDX(1,0,offset,nr, n_subcarrier)], vnclip_wx_i16m1(acc_G_10_imag, 10, vl), vl);
    

    //G20
    t2_real = vle16_v_i16m1(&H_real[GET_IDX(3, 2, offset, nr, n_subcarrier)], vl);
    acc_G_20_real = vwmacc_vv_i32m2(acc_G_20_real, t2_real, t0_real, vl);

    t2_imag = vle16_v_i16m1(&H_imag[GET_IDX(3, 2, offset, nr, n_subcarrier)], vl);
    acc_G_20_real = vwmacc_vv_i32m2(acc_G_20_real, t2_imag, t0_imag, vl);
    vse16_v_i16m1(&Gram_real[GET_IDX(2,0,offset,nr, n_subcarrier)], vnclip_wx_i16m1(acc_G_20_real, 10, vl), vl);

    acc_G_20_imag = vwmacc_vv_i32m2(acc_G_20_imag, t2_real, t0_imag, vl);
    acc_G_20_imag = vsub_vv_i32m2(acc_G_20_imag, vwmul_vv_i32m2(t2_imag, t0_real, vl), vl);
    vse16_v_i16m1(&Gram_imag[GET_IDX(2,0,offset,nr, n_subcarrier)], vnclip_wx_i16m1(acc_G_20_imag, 10, vl), vl);
    

    //G30
    t3_real = vle16_v_i16m1(&H_real[GET_IDX(3, 3, offset, nr, n_subcarrier)], vl);
    acc_G_30_real = vwmacc_vv_i32m2(acc_G_30_real, t3_real, t0_real, vl);

    t3_imag = vle16_v_i16m1(&H_imag[GET_IDX(3, 3, offset, nr, n_subcarrier)], vl);
    acc_G_30_real = vwmacc_vv_i32m2(acc_G_30_real, t3_imag, t0_imag, vl);
    vse16_v_i16m1(&Gram_real[GET_IDX(3,0,offset,nr, n_subcarrier)], vnclip_wx_i16m1(acc_G_30_real, 10, vl), vl);

    acc_G_30_imag = vwmacc_vv_i32m2(acc_G_30_imag, t3_real, t0_imag, vl);
    acc_G_30_imag = vsub_vv_i32m2(acc_G_30_imag, vwmul_vv_i32m2(t3_imag, t0_real, vl), vl);
    vse16_v_i16m1(&Gram_imag[GET_IDX(3,0,offset,nr, n_subcarrier)], vnclip_wx_i16m1(acc_G_30_imag, 10, vl), vl);

    
    //G11
    acc_G_11_real = vwmacc_vv_i32m2(acc_G_11_real, t1_real, t1_real, vl);
    acc_G_11_real = vwmacc_vv_i32m2(acc_G_11_real, t1_imag, t1_imag, vl);

    vint16m1_t G_11_real = vadd_vx_i16m1(vnclip_wx_i16m1(acc_G_11_real, 10, vl), sigma_n2, vl);
    vse16_v_i16m1(&Gram_real[GET_IDX(1,1,offset,nr, n_subcarrier)], G_11_real, vl);
      
    //G21
    acc_G_21_real = vwmacc_vv_i32m2(acc_G_21_real, t2_real, t1_real, vl);
    acc_G_21_real = vwmacc_vv_i32m2(acc_G_21_real, t2_imag, t1_imag, vl);
    vse16_v_i16m1(&Gram_real[GET_IDX(2,1,offset,nr, n_subcarrier)], vnclip_wx_i16m1(acc_G_21_real, 10, vl), vl);

    acc_G_21_imag = vwmacc_vv_i32m2(acc_G_21_imag, t2_real, t1_imag, vl);
    acc_G_21_imag = vsub_vv_i32m2(acc_G_21_imag, vwmul_vv_i32m2(t2_imag, t1_real, vl), vl);
    vse16_v_i16m1(&Gram_imag[GET_IDX(2,1,offset,nr, n_subcarrier)], vnclip_wx_i16m1(acc_G_21_imag, 10, vl), vl);

    
    //G31
    acc_G_31_real = vwmacc_vv_i32m2(acc_G_31_real, t3_real, t1_real, vl);
    acc_G_31_real = vwmacc_vv_i32m2(acc_G_31_real, t3_imag, t1_imag, vl);
    vse16_v_i16m1(&Gram_real[GET_IDX(3,1,offset,nr, n_subcarrier)], vnclip_wx_i16m1(acc_G_31_real, 10, vl), vl);

    acc_G_31_imag = vwmacc_vv_i32m2(acc_G_31_imag, t3_real, t1_imag, vl);
    acc_G_31_imag = vsub_vv_i32m2(acc_G_31_imag, vwmul_vv_i32m2(t3_imag, t1_real, vl), vl);
    vse16_v_i16m1(&Gram_imag[GET_IDX(3,1,offset,nr, n_subcarrier)], vnclip_wx_i16m1(acc_G_31_imag, 10, vl), vl);

    
    //G22
    acc_G_22_real = vwmacc_vv_i32m2(acc_G_22_real, t2_real, t2_real, vl);
    acc_G_22_real = vwmacc_vv_i32m2(acc_G_22_real, t2_imag, t2_imag, vl);
    vint16m1_t G_22_real = vadd_vx_i16m1(vnclip_wx_i16m1(acc_G_22_real, 10, vl), sigma_n2, vl);
    vse16_v_i16m1(&Gram_real[GET_IDX(2,2,offset,nr, n_subcarrier)], G_22_real, vl);

  

    //G32
    acc_G_32_real = vwmacc_vv_i32m2(acc_G_32_real, t3_real, t2_real, vl);
    acc_G_32_real = vwmacc_vv_i32m2(acc_G_32_real, t3_imag, t2_imag, vl);
    vse16_v_i16m1(&Gram_real[GET_IDX(3,2,offset,nr, n_subcarrier)], vnclip_wx_i16m1(acc_G_32_real, 10, vl), vl);

    acc_G_32_imag = vwmacc_vv_i32m2(acc_G_32_imag, t3_real, t2_imag, vl);
    acc_G_32_imag = vsub_vv_i32m2(acc_G_32_imag, vwmul_vv_i32m2(t3_imag, t2_real, vl), vl);
    vse16_v_i16m1(&Gram_imag[GET_IDX(3,2,offset,nr, n_subcarrier)], vnclip_wx_i16m1(acc_G_32_imag, 10, vl), vl);
    
    //G33
    acc_G_33_real = vwmacc_vv_i32m2(acc_G_33_real, t3_real, t3_real, vl);
    acc_G_33_real = vwmacc_vv_i32m2(acc_G_33_real, t3_imag, t3_imag, vl);
    vint16m1_t G_33_real = vadd_vx_i16m1(vnclip_wx_i16m1(acc_G_33_real, 10, vl), sigma_n2, vl);
    vse16_v_i16m1(&Gram_real[GET_IDX(3,3,offset,nr, n_subcarrier)], G_33_real, vl);

    
  }

#elif CASE == 1
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
  int32_t  *Gram_imag) {

    vint16m1_t t0_real, t0_imag, t1_real, t1_imag, t2_real, t2_imag, t3_real, t3_imag;
    vint16m1_t ta_real, ta_imag, tb_real, tb_imag, tc_real, tc_imag, td_real, td_imag;
    vint32m2_t acc_0_real;
    vint32m2_t acc_0_imag;

    
    for (int i = 0; i < 4; i++) {
      acc_0_real = vmv_v_x_i32m2(sigma_n2 << 10, vl);
      //acc_0_imag = vmv_v_x_i32m2(0.0f, vl);

    //G_ii
      //0a
      t0_real = vle16_v_i16m1(&H_real[GET_IDX(0, i, offset, nr, n_subcarrier)], vl);
      acc_0_real = vwmacc_vv_i32m2(acc_0_real, t0_real, t0_real, vl);

      ta_imag = vle16_v_i16m1(&H_imag[GET_IDX(0, i, offset, nr, n_subcarrier)], vl);
      t0_imag = vneg_v_i16m1(ta_imag, vl);
      acc_0_real = vwmacc_vv_i32m2(acc_0_real, ta_imag, ta_imag, vl);

      
      //1b
      t1_real = vle16_v_i16m1(&H_real[GET_IDX(1, i, offset, nr, n_subcarrier)], vl);
      acc_0_real = vwmacc_vv_i32m2(acc_0_real, t1_real, t1_real, vl);

      tb_imag = vle16_v_i16m1(&H_imag[GET_IDX(1, i, offset, nr, n_subcarrier)], vl);
      t1_imag = vneg_v_i16m1(tb_imag, vl);
      acc_0_real = vwmacc_vv_i32m2(acc_0_real, tb_imag, tb_imag, vl);



      //2c
      t2_real = vle16_v_i16m1(&H_real[GET_IDX(2, i, offset, nr, n_subcarrier)], vl);
      acc_0_real = vwmacc_vv_i32m2(acc_0_real, t2_real, t2_real, vl);

      tc_imag = vle16_v_i16m1(&H_imag[GET_IDX(2, i, offset, nr, n_subcarrier)], vl);
      t2_imag = vneg_v_i16m1(tc_imag, vl);
      acc_0_real = vwmacc_vv_i32m2(acc_0_real, tc_imag, tc_imag, vl);


      //3d
      t3_real = vle16_v_i16m1(&H_real[GET_IDX(3, i, offset, nr, n_subcarrier)], vl);
      acc_0_real = vwmacc_vv_i32m2(acc_0_real, t3_real, t3_real, vl);

      td_imag = vle16_v_i16m1(&H_imag[GET_IDX(3, i, offset, nr, n_subcarrier)], vl);
      t3_imag = vneg_v_i16m1(td_imag, vl);
      acc_0_real =vwmacc_vv_i32m2(acc_0_real, td_imag, td_imag, vl);


      vse32_v_i32m2(&Gram_real[GET_IDX(i, i, offset, nr, n_subcarrier)], acc_0_real, vl);

      
      
      for (int j = i - 1; j > -1; j--) {
        acc_0_real = vmv_v_x_i32m2(0, vl);
        acc_0_imag = vmv_v_x_i32m2(0, vl);
        
        ta_real = vle16_v_i16m1(&H_real[GET_IDX(0, j, offset, nr, n_subcarrier)], vl);
        acc_0_real = vwmacc_vv_i32m2(acc_0_real, t0_real, ta_real, vl);

        ta_imag = vle16_v_i16m1(&H_imag[GET_IDX(0, j, offset, nr, n_subcarrier)], vl);
        acc_0_real = vsub_vv_i32m2(acc_0_real, vwmul_vv_i32m2(t0_imag, ta_imag, vl), vl);

        acc_0_imag = vwmacc_vv_i32m2(acc_0_imag, t0_real, ta_imag, vl);
        acc_0_imag = vwmacc_vv_i32m2(acc_0_imag, t0_imag, ta_real, vl);
      
        //1b
        tb_real = vle16_v_i16m1(&H_real[GET_IDX(1, j, offset, nr, n_subcarrier)], vl);
        acc_0_real = vwmacc_vv_i32m2(acc_0_real, t1_real, tb_real, vl);

        tb_imag = vle16_v_i16m1(&H_imag[GET_IDX(1, j, offset, nr, n_subcarrier)], vl);
        acc_0_real = vsub_vv_i32m2(acc_0_real, vwmul_vv_i32m2(t1_imag, tb_imag, vl), vl);

        acc_0_imag = vwmacc_vv_i32m2(acc_0_imag, t1_real, tb_imag, vl);
        acc_0_imag = vwmacc_vv_i32m2(acc_0_imag, t1_imag, tb_real, vl);


        //2c
        tc_real = vle16_v_i16m1(&H_real[GET_IDX(2, j, offset, nr, n_subcarrier)], vl);
        acc_0_real = vwmacc_vv_i32m2(acc_0_real, t2_real, tc_real, vl);

        tc_imag = vle16_v_i16m1(&H_imag[GET_IDX(2, j, offset, nr, n_subcarrier)], vl);
        acc_0_real = vsub_vv_i32m2(acc_0_real, vwmul_vv_i32m2(t2_imag, tc_imag, vl), vl);

        acc_0_imag = vwmacc_vv_i32m2(acc_0_imag, t2_real, tc_imag, vl);
        acc_0_imag = vwmacc_vv_i32m2(acc_0_imag, t2_imag, tc_real, vl);

          

        //3d
        td_real = vle16_v_i16m1(&H_real[GET_IDX(3, j, offset, nr, n_subcarrier)], vl);
        acc_0_real = vwmacc_vv_i32m2(acc_0_real, t3_real, td_real, vl);

        td_imag = vle16_v_i16m1(&H_imag[GET_IDX(3, j, offset, nr, n_subcarrier)], vl);
        acc_0_real = vsub_vv_i32m2(acc_0_real, vwmul_vv_i32m2(t3_imag, td_imag, vl), vl);
        vse32_v_i32m2(&Gram_real[GET_IDX(i, j, offset, nr, n_subcarrier)], acc_0_real, vl);

        acc_0_imag = vwmacc_vv_i32m2(acc_0_imag, t3_real, td_imag, vl);
        acc_0_imag = vwmacc_vv_i32m2(acc_0_imag, t3_imag, td_real, vl);
        vse32_v_i32m2(&Gram_imag[GET_IDX(i, j, offset, nr, n_subcarrier)], acc_0_imag, vl); 
        



      }


    }



    
  }




#elif CASE == 3
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
  int32_t  *Gram_imag) {

    vint32m2_t t0_real, t0_imag, t1_real, t1_imag, t2_real, t2_imag, t3_real, t3_imag;
    vint32m2_t ta_real, ta_imag, tb_real, tb_imag, tc_real, tc_imag, td_real, td_imag;
    vint32m2_t acc_0_real;
    vint32m2_t acc_0_imag;

    
    for (int i = 0; i < 4; i++) {
      acc_0_real = vmv_v_x_i32m2(sigma_n2 << 10, vl);
      //acc_0_imag = vmv_v_x_i32m2(0.0f, vl);

    //G_ii
      
      //0a
      t0_real = vwadd_vx_i32m2(vle16_v_i16m1(&H_real[GET_IDX(0, i, offset, nr, n_subcarrier)], vl), 0 ,vl);
      acc_0_real = vmacc_vv_i32m2(acc_0_real, t0_real, t0_real, vl);
      
      ta_imag = vwadd_vx_i32m2(vle16_v_i16m1(&H_imag[GET_IDX(0, i, offset, nr, n_subcarrier)], vl), 0, vl);
      t0_imag = vneg_v_i32m2(ta_imag, vl);
      acc_0_real = vmacc_vv_i32m2(acc_0_real, ta_imag, ta_imag, vl);

      
      //1b
      
      t1_real = vwadd_vx_i32m2(vle16_v_i16m1(&H_real[GET_IDX(1, i, offset, nr, n_subcarrier)], vl),0, vl);
      acc_0_real = vmacc_vv_i32m2(acc_0_real, t1_real, t1_real, vl);

      tb_imag = vwadd_vx_i32m2(vle16_v_i16m1(&H_imag[GET_IDX(1, i, offset, nr, n_subcarrier)], vl),0, vl);
      t1_imag = vneg_v_i32m2(tb_imag, vl);
      acc_0_real = vmacc_vv_i32m2(acc_0_real, tb_imag, tb_imag, vl);


      
      //2c
      t2_real = vwadd_vx_i32m2(vle16_v_i16m1(&H_real[GET_IDX(2, i, offset, nr, n_subcarrier)], vl),0,vl);
      acc_0_real = vmacc_vv_i32m2(acc_0_real, t2_real, t2_real, vl);

      tc_imag = vwadd_vx_i32m2(vle16_v_i16m1(&H_imag[GET_IDX(2, i, offset, nr, n_subcarrier)], vl),0,vl);
      t2_imag = vneg_v_i32m2(tc_imag, vl);
      acc_0_real = vmacc_vv_i32m2(acc_0_real, tc_imag, tc_imag, vl);

      
      //3d
      t3_real = vwadd_vx_i32m2(vle16_v_i16m1(&H_real[GET_IDX(3, i, offset, nr, n_subcarrier)], vl),0,vl);
      acc_0_real = vmacc_vv_i32m2(acc_0_real, t3_real, t3_real, vl);

      td_imag = vwadd_vx_i32m2(vle16_v_i16m1(&H_imag[GET_IDX(3, i, offset, nr, n_subcarrier)], vl),0,vl);
      t3_imag = vneg_v_i32m2(td_imag, vl);
      acc_0_real =vmacc_vv_i32m2(acc_0_real, td_imag, td_imag, vl);
      
      
      vse32_v_i32m2(&Gram_real[GET_IDX(i, i, offset, nr, n_subcarrier)], acc_0_real, vl);
      
      
      
      for (int j = i - 1; j > -1; j--) {
      
        acc_0_real = vmv_v_x_i32m2(0, vl);
        acc_0_imag = vmv_v_x_i32m2(0, vl);
        
        ta_real = vwadd_vx_i32m2(vle16_v_i16m1(&H_real[GET_IDX(0, j, offset, nr, n_subcarrier)], vl),0,vl);
        acc_0_real = vmacc_vv_i32m2(acc_0_real, t0_real, ta_real, vl);

        ta_imag = vwadd_vx_i32m2(vle16_v_i16m1(&H_imag[GET_IDX(0, j, offset, nr, n_subcarrier)], vl),0,vl);
        acc_0_real = vsub_vv_i32m2(acc_0_real, vmul_vv_i32m2(t0_imag, ta_imag, vl), vl);

        acc_0_imag = vmacc_vv_i32m2(acc_0_imag, t0_real, ta_imag, vl);
        acc_0_imag = vmacc_vv_i32m2(acc_0_imag, t0_imag, ta_real, vl);
      
        //1b
        tb_real = vwadd_vx_i32m2(vle16_v_i16m1(&H_real[GET_IDX(1, j, offset, nr, n_subcarrier)], vl),0,vl);
        acc_0_real = vmacc_vv_i32m2(acc_0_real, t1_real, tb_real, vl);

        tb_imag = vwadd_vx_i32m2(vle16_v_i16m1(&H_imag[GET_IDX(1, j, offset, nr, n_subcarrier)], vl),0,vl);
        acc_0_real = vsub_vv_i32m2(acc_0_real, vmul_vv_i32m2(t1_imag, tb_imag, vl), vl);

        acc_0_imag = vmacc_vv_i32m2(acc_0_imag, t1_real, tb_imag, vl);
        acc_0_imag = vmacc_vv_i32m2(acc_0_imag, t1_imag, tb_real, vl);


        //2c
        tc_real = vwadd_vx_i32m2(vle16_v_i16m1(&H_real[GET_IDX(2, j, offset, nr, n_subcarrier)], vl),0,vl);
        acc_0_real = vmacc_vv_i32m2(acc_0_real, t2_real, tc_real, vl);

        tc_imag = vwadd_vx_i32m2(vle16_v_i16m1(&H_imag[GET_IDX(2, j, offset, nr, n_subcarrier)], vl),0,vl);
        acc_0_real = vsub_vv_i32m2(acc_0_real, vmul_vv_i32m2(t2_imag, tc_imag, vl), vl);

        acc_0_imag = vmacc_vv_i32m2(acc_0_imag, t2_real, tc_imag, vl);
        acc_0_imag = vmacc_vv_i32m2(acc_0_imag, t2_imag, tc_real, vl);

          

        //3d
        td_real = vwadd_vx_i32m2(vle16_v_i16m1(&H_real[GET_IDX(3, j, offset, nr, n_subcarrier)], vl),0,vl);
        acc_0_real = vmacc_vv_i32m2(acc_0_real, t3_real, td_real, vl);

        td_imag = vwadd_vx_i32m2(vle16_v_i16m1(&H_imag[GET_IDX(3, j, offset, nr, n_subcarrier)], vl),0,vl);
        acc_0_real = vsub_vv_i32m2(acc_0_real, vmul_vv_i32m2(t3_imag, td_imag, vl), vl);
        vse32_v_i32m2(&Gram_real[GET_IDX(i, j, offset, nr, n_subcarrier)], acc_0_real, vl);

        acc_0_imag = vmacc_vv_i32m2(acc_0_imag, t3_real, td_imag, vl);
        acc_0_imag = vmacc_vv_i32m2(acc_0_imag, t3_imag, td_real, vl);
        vse32_v_i32m2(&Gram_imag[GET_IDX(i, j, offset, nr, n_subcarrier)], acc_0_imag, vl); 
        
        

      }
      

    }



    
  }



#elif CASE == 4
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
  int32_t  *Gram_imag) {

    vint16m1_t t0_real, t0_imag, t1_real, t1_imag, t2_real, t2_imag, t3_real, t3_imag;
    vint16m1_t ta_real, ta_imag, tb_real, tb_imag, tc_real, tc_imag, td_real, td_imag;
    vint32m2_t acc_0_real;
    vint32m2_t acc_0_imag;

    
    for (int i = 0; i < 4; i++) {
      acc_0_real = vmv_v_x_i32m2(sigma_n2 << 10, vl);
      //acc_0_imag = vmv_v_x_i32m2(0.0f, vl);

    //G_ii
      //0a
      t0_real = vle16_v_i16m1(&H_real[GET_IDX(0, i, offset, nr, n_subcarrier)], vl);
      acc_0_real = vwmacc_vv_i32m2(acc_0_real, t0_real, t0_real, vl);

      ta_imag = vle16_v_i16m1(&H_imag[GET_IDX(0, i, offset, nr, n_subcarrier)], vl);
      t0_imag = vneg_v_i16m1(ta_imag, vl);
      acc_0_real = vwmacc_vv_i32m2(acc_0_real, ta_imag, ta_imag, vl);

      
      //1b
      t1_real = vle16_v_i16m1(&H_real[GET_IDX(1, i, offset, nr, n_subcarrier)], vl);
      acc_0_real = vwmacc_vv_i32m2(acc_0_real, t1_real, t1_real, vl);

      tb_imag = vle16_v_i16m1(&H_imag[GET_IDX(1, i, offset, nr, n_subcarrier)], vl);
      t1_imag = vneg_v_i16m1(tb_imag, vl);
      acc_0_real = vwmacc_vv_i32m2(acc_0_real, tb_imag, tb_imag, vl);



      //2c
      t2_real = vle16_v_i16m1(&H_real[GET_IDX(2, i, offset, nr, n_subcarrier)], vl);
      acc_0_real = vwmacc_vv_i32m2(acc_0_real, t2_real, t2_real, vl);

      tc_imag = vle16_v_i16m1(&H_imag[GET_IDX(2, i, offset, nr, n_subcarrier)], vl);
      t2_imag = vneg_v_i16m1(tc_imag, vl);
      acc_0_real = vwmacc_vv_i32m2(acc_0_real, tc_imag, tc_imag, vl);


      //3d
      t3_real = vle16_v_i16m1(&H_real[GET_IDX(3, i, offset, nr, n_subcarrier)], vl);
      acc_0_real = vwmacc_vv_i32m2(acc_0_real, t3_real, t3_real, vl);

      td_imag = vle16_v_i16m1(&H_imag[GET_IDX(3, i, offset, nr, n_subcarrier)], vl);
      t3_imag = vneg_v_i16m1(td_imag, vl);
      acc_0_real =vwmacc_vv_i32m2(acc_0_real, td_imag, td_imag, vl);


      vse32_v_i32m2(&Gram_real[GET_IDX(i, i, offset, nr, n_subcarrier)], acc_0_real, vl);

      
      
      for (int j = i - 1; j > -1; j--) {
        acc_0_real = vmv_v_x_i32m2(0, vl);
        acc_0_imag = vmv_v_x_i32m2(0, vl);
        
        ta_real = vle16_v_i16m1(&H_real[GET_IDX(0, j, offset, nr, n_subcarrier)], vl);
        acc_0_real = vwmacc_vv_i32m2(acc_0_real, t0_real, ta_real, vl);

        ta_imag = vle16_v_i16m1(&H_imag[GET_IDX(0, j, offset, nr, n_subcarrier)], vl);
        acc_0_real = vsub_vv_i32m2(acc_0_real, vwmul_vv_i32m2(t0_imag, ta_imag, vl), vl);

        acc_0_imag = vwmacc_vv_i32m2(acc_0_imag, t0_real, ta_imag, vl);
        acc_0_imag = vwmacc_vv_i32m2(acc_0_imag, t0_imag, ta_real, vl);
      
        //1b
        tb_real = vle16_v_i16m1(&H_real[GET_IDX(1, j, offset, nr, n_subcarrier)], vl);
        acc_0_real = vwmacc_vv_i32m2(acc_0_real, t1_real, tb_real, vl);

        tb_imag = vle16_v_i16m1(&H_imag[GET_IDX(1, j, offset, nr, n_subcarrier)], vl);
        acc_0_real = vsub_vv_i32m2(acc_0_real, vwmul_vv_i32m2(t1_imag, tb_imag, vl), vl);

        acc_0_imag = vwmacc_vv_i32m2(acc_0_imag, t1_real, tb_imag, vl);
        acc_0_imag = vwmacc_vv_i32m2(acc_0_imag, t1_imag, tb_real, vl);


        //2c
        tc_real = vle16_v_i16m1(&H_real[GET_IDX(2, j, offset, nr, n_subcarrier)], vl);
        acc_0_real = vwmacc_vv_i32m2(acc_0_real, t2_real, tc_real, vl);

        tc_imag = vle16_v_i16m1(&H_imag[GET_IDX(2, j, offset, nr, n_subcarrier)], vl);
        acc_0_real = vsub_vv_i32m2(acc_0_real, vwmul_vv_i32m2(t2_imag, tc_imag, vl), vl);

        acc_0_imag = vwmacc_vv_i32m2(acc_0_imag, t2_real, tc_imag, vl);
        acc_0_imag = vwmacc_vv_i32m2(acc_0_imag, t2_imag, tc_real, vl);

          

        //3d
        td_real = vle16_v_i16m1(&H_real[GET_IDX(3, j, offset, nr, n_subcarrier)], vl);
        acc_0_real = vwmacc_vv_i32m2(acc_0_real, t3_real, td_real, vl);

        td_imag = vle16_v_i16m1(&H_imag[GET_IDX(3, j, offset, nr, n_subcarrier)], vl);
        acc_0_real = vsub_vv_i32m2(acc_0_real, vwmul_vv_i32m2(t3_imag, td_imag, vl), vl);
        vse32_v_i32m2(&Gram_real[GET_IDX(i, j, offset, nr, n_subcarrier)], acc_0_real, vl);

        acc_0_imag = vwmacc_vv_i32m2(acc_0_imag, t3_real, td_imag, vl);
        acc_0_imag = vwmacc_vv_i32m2(acc_0_imag, t3_imag, td_real, vl);
        vse32_v_i32m2(&Gram_imag[GET_IDX(i, j, offset, nr, n_subcarrier)], acc_0_imag, vl); 
        



      }


    }



    
  }









#endif 