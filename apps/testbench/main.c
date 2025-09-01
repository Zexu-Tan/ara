// Author: Zexu.Tan <zexu.tan@tu-dresden.de>

#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <riscv_vector.h>

#include "runtime.h"

#ifndef SPIKE
#include "printf.h"
#else
#include <stdio.h>
#endif

#include "kernel/arith_lib.h"
#include "kernel/dataset.h"
#include <math.h>
#define EPS 1e-3f

int main(void) {
  size_t vl = vsetvl_e32m1(N_BEAM);

  float m_A1_real[NT][NR][N_BEAM] = {0};
  float m_A1_imag[NT][NR][N_BEAM] = {0};







  return 0;
}
