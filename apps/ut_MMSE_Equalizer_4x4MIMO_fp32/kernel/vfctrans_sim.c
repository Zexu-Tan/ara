// Copyright 2025 Vodafone Sponsored Chair of Communication System, TU Dresden and Synopsys

// Author: Zexu Tan:zexu.tan@tu-dresden.de, Vodafone Chair, TU Dresden


#include <riscv_vector.h>
#include "arith_lib.h"


void vfctrans_vv_f32m1_sim(vfloat32m1_t a_real, vfloat32m1_t a_imag, size_t vl) {
  a_imag = vfneg_v_f32m1(a_imag, vl);
}
