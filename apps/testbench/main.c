// Author: Zexu.Tan <zexu.tan@tu-dresden.de>

#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <riscv_vector.h>

#include "runtime.h"
#include <math.h>

#ifndef SPIKE
#include "printf.h"
#else
#include <stdio.h>
#endif



#include <math.h>
#define EPS 1e-3f


int main(void) {
  size_t vl = vsetvl_e64m1(256);

  double i_0[vl];
  double i_1[vl];
  double i_2[vl];
  double i_3[vl];
  double output[vl];
  double output_1[vl];

  for (int i = 0; i < 256; i++){
    i_0[i] = 0.0001;
    i_1[i] = 0.0001;
    i_2[i] = 0.0001;
    i_3[i] = 0.0001;
  }

  start_timer();
  vfloat64m1_t v0, v1, v2;
  vfloat64m1_t acc_0, acc_1;
  double b = 1.1;

  v0 = vle64_v_f64m1(&i_0[0], vl);
  v0 = vfmacc_vv_f64m1(v0, v0, v0, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v0, vl);
  
  v1 = vle64_v_f64m1(&i_1[0], vl);

  //v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);

  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);

  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);

  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);

  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);

  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);

  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);

  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);
  v2 = vfmacc_vv_f64m1(v2, v0, v1, vl);







  vse64_v_f64m1(&output_1[0], v2, vl);




    
  stop_timer();

  int64_t runtime = get_timer();

  float performance = (2.0 * 32 * vl) / runtime;
  float utilization = (100.0 * performance) / (2.0 * NR_LANES);

  printf("vl = %d \n", vl);
  printf("The execution took %d cycles.\n", runtime);
  printf("The performance is %.5f FLOP/cycle (%.5f%% utilization).\n", performance, utilization);

  

  return 0;
}
