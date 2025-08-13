// Copyright 2020 ETH Zurich and University of Bologna.
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
    const int32_t N_subcarriers = N_BEAM;
    const size_t vl = N_ITR_BEAM;
    
    vfloat32m1_t t_real
  
    vfloat32m1_t H_H_0_0_real = vle32_v_f32m1(H_real[0][0], vl);
    vfloat32m1_t H_H_0_0_imag = vle32_v_f32m1(H_imag[0][0], vl);
    vfloat32m1_t H_H_0_0_imag = vfneg_v_f32m1(H_H_0_0_imag, vl);
    
    vfloat32m1_t H_0_0_real = vle32_v_f32m1(H_real[0][0], vl);
    vfloat32m1_t H_0_0_imag = vle32_v_f32m1(H_imag[0][0], vl);
    
    vfloat32m1_t t_0_0_real, t_0_0_imag;
    vcfmul_vv_f32m1_sim(H_H_0_0_real, H_H_0_0_imag, H_0_0_real, H_0_0_imag, &t_0_0_real)
  
    
    vfloat32m1_t H_H_0_1_real = vle32_v_f32m1(H_real[1][0], vl);
    vfloat32m1_t H_H_0_1_imag = vle32_v_f32m1(H_imag[1][0], vl);
    vfloat32m1_t H_H_0_0_imag = vfneg_v_f32m1(H_H_0_0_imag, vl);
    
    
    

  
  
    
    
    return 0;
}
