// Copyright 2025 Vodafone Sponsored Chair of Communication System, TUD and Synopsys
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

// Author: Zexu Tan, Vodafone Chair, TU Dresden


#include <riscv_vector.h>
#include "arith_lib.h"

void vcfmul_vv_f32m1_sim(    
    vfloat32m1_t vr_ar,
    vfloat32m1_t vr_ai,
    vfloat32m1_t vr_br,
    vfloat32m1_t vr_bi,
    vfloat32m1_t *vr_cr,
    vfloat32m1_t *vr_ci,
    size_t vl)
{
    // vr_cr = ar*br - ai*bi
    vfloat32m1_t t1 = vfmul_vv_f32m1(vr_ar, vr_br, vl);
    vfloat32m1_t t2 = vfmul_vv_f32m1(vr_ai, vr_bi, vl);
    *vr_cr = vfsub_vv_f32m1(t1, t2, vl);

    // vr_cr = ar*bi + ai*br
    vfloat32m1_t t3 = vfmul_vv_f32m1(vr_ar, vr_bi, vl);
    vfloat32m1_t t4 = vfmul_vv_f32m1(vr_ai, vr_br, vl);
    *vr_ci = vfadd_vv_f32m1(t3, t4, vl);
}


