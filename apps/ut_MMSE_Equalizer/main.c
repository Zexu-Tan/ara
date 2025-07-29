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
    const size_t vl_itr_subcarriers = N_ITR_BEAM;
    
    vfloat32m1_t H_H_i_real[Nt];
    vfloat32m1_t H_H_i_imag[Nt];
    vfloat32m1_t H_j_real[Nt];
    vfloat32m1_t H_j_imag[Nt];
    
    

  
  
    
    
    return 0;
}
