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



//Attention Intrinsic of Strided Segment load/store not supported by Clang16/GCC13, avoid to use it

//Conclusion vlsseg2e32.v is two times longer than the linear cases of Interleaved stride load

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

/*
int32_t data0[N] __attribute__((aligned(32 * NR_LANES), section(".l2")));
int32_t data1[N] __attribute__((aligned(32 * NR_LANES), section(".l2")));


int main() {
  printf("TEST: stride load");
  
  
  int32_t data0[N] = {11, 12, 13, 14, 15, 16, 17, 18};
  int32_t data1[N] = {1, 2, 3, 4, 5, 6, 7, 8};
  int32_t stride_loaded_result[4];

  size_t vl = vsetvl_e32m1(4);

  ptrdiff_t stride = 2 * sizeof(int32_t);

  vint32m1_t vec = vlse32_v_i32m1(data0, stride, vl);


  vse32(stride_loaded_result, vec, vl);

  for (size_t i = 0; i < vl; i++) {
        printf("stride_loaded_result[%zu] = %d\n", i, stride_loaded_result[i]);
  }



  return 0;
}
*/

#define AVL 4

int main(void) {


/*
    int32_t data[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    int32_t even[4];
    int32_t odd[4];
    int64_t avg_runtime = 0;
    const ptrdiff_t stride = 2 * sizeof(int32_t);

    size_t vl;


  //declaration before using it


  
    for (int i = 0; i < 100; i++){
      asm volatile("vsetvli %0, %1, e32, m1, ta, ma" :"=r"(vl) : "r"(AVL));
      asm volatile("vmv.v.x v0, x0");
      asm volatile("vmv.v.x v1, x0");

      
      start_timer();
      asm volatile("vlsseg2e32.v v0, (%0), %1 " :: "r"(data),"r"(stride));
      stop_timer();


      asm volatile("vse32.v v0, (%0)" :: "r"(odd));
      asm volatile("vse32.v v1, (%0)" :: "r"(even));

      avg_runtime += get_timer();
    }

    avg_runtime = avg_runtime / 100;
    
    printf("----------Assembly, Strided segmented, vlsseg2e32.v(vlsseg2e32_v_i32m1x2)----------\n");
    printf("average_runtime = %d\n", avg_runtime);
 
  */




    int32_t data[N] = {1, 2, 3, 4, 5, 6, 7, 8};
    int32_t even[4];
    int32_t odd[4];

    const ptrdiff_t stride = 2 * sizeof(int32_t);

    int64_t avg_runtime = 0;

    for (int i = 0; i < 200; i++){

      size_t vl = vsetvl_e32m1(AVL);

      start_timer();
      vint32m1_t vec_odd = vlse32_v_i32m1(data, stride, vl);
      vint32m1_t vec_even = vlse32_v_i32m1(data + 1, stride, vl);
      stop_timer();

      vse32(odd, vec_odd, vl);
      vse32(even, vec_even, vl);

      avg_runtime += get_timer();
    
    }

    avg_runtime = avg_runtime / 200;
    

    printf("----------Intrinsic, Strided, vlse32_v_i32m1(vlse32.v)----------\n");
    printf("average_runtime = %d\n", avg_runtime);

    return 0;

}