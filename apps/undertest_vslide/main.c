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

#define N 8
int32_t data0[N] __attribute__((aligned(32 * NR_LANES), section(".l2")));
int32_t data1[N] __attribute__((aligned(32 * NR_LANES), section(".l2")));


int main() {
  printf("TEST: vslide");
  
  
  int32_t data0[N] = {1, 2, 3, 4, 5, 6, 7, 8};
  int32_t data1[N] = {1, 2, 3, 4, 5, 6, 7, 8};
  int32_t slided_result[N];

  size_t vl = vsetvl_e32m1(N); // 设置向量长度为4


  vint32m1_t v0 = vle32_v_i32m1(data0, vl);
  vint32m1_t v1 = vle32_v_i32m1(data1, vl);


  vint32m1_t vidx = vid_v_i32m1(vl);
  vbool32_t vslide_mask = vmsge_vx_i32m1_b32(vidx, 4, vl);
  

  int t0 = 4;

  v1 = vslideup(vslide_mask, v1, v0, t0, vl);

  vse32(slided_result, v1, vl);


  return 0;
}
