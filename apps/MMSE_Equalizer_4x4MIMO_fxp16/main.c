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

#include "kernel/lib.h"
#include "kernel/dataset.h"

#define SCALED 1
#define DEBUG 1

int main(void) {
    size_t vl = vsetvl_e16m1(N_SUBCARRIER_ITR);
    size_t offset = 0;

    int16_t Gram_real[NT][NR][N_SUBCARRIER] = {0};
    int16_t Gram_imag[NT][NR][N_SUBCARRIER] = {0};

    int32_t Gram_real_unscaled[NT][NR][N_SUBCARRIER] = {0};
    int32_t Gram_imag_unscaled[NT][NR][N_SUBCARRIER] = {0};

    //match with python saturation
    asm volatile("csrwi vxrm, 2" ::: "memory");

    start_timer();
    
    #if SCALED == 1
    v_compute_gram_cfxp16(
        NT, 
        NR, 
        vl, 
        offset, 
        N_SUBCARRIER,
        sigma_n2_fxp,
        (int16_t*)H_real_fxp, 
        (int16_t*)H_imag_fxp,
        (int16_t*)Gram_real,
        (int16_t*)Gram_imag);

    #elif SCALED == 0
     v_compute_gram_cfxp16_unscaled(
        NT, 
        NR, 
        vl, 
        offset, 
        N_SUBCARRIER,
        sigma_n2_fxp,
        (int16_t*)H_real_fxp, 
        (int16_t*)H_imag_fxp,
        (int32_t*)Gram_real_unscaled,
        (int32_t*)Gram_imag_unscaled);
    
    #endif
    stop_timer();

    int64_t runtime = get_timer();

    
    #if SCALED == 1

    float performance = (2 * 2 * (4.0 * NT * (NT + 1) / 2 - NT * 2.0) * NT )* vl / runtime;

    #elif SCLAED == 0

    float performance = (2 * 2 * (4.0 * NT * (NT + 1) / 2 - NT * 2.0) * NT +  16 * 4 * 2 )* vl / runtime;

    #endif

   // float performance = （2 * 2 * (4.0 * NT * (NT + 1) / 2 - NT * 2.0) * NT +  16 * 4 * 2 ）* vl / runtime;
        // 一个32的macc相当于4个16的ops

    float utilization = 100.0 * performance / (8.0 * NR_LANES);


    printf("========Kernel: MMSE_Equalizer_4x4MIMO_CFXP16========\n");
    printf("Sigma_n2_fxp = %d \n", sigma_n2_fxp);
    int k_debug = 1;
    printf("vl = %d \n", vl);

    printf("DLEN = %db \n", NR_LANES * 64);

    printf("VLEN = %db \n", VLEN);

    printf("The execution took %d cycles.\n", runtime);
    printf("The performance is %f FLOP/cycle (%f%% utilization).\n", performance/4.0, utilization);


    
#ifdef DEBUG

    printf("===================v_compute_gram()=====================\n");
    printf("Gram_real\n");
    for (int i = 0; i < NR; i++){
        for (int j = 0; j < NT; j++){
            printf("  %d    ", Gram_real[i][j][k_debug]);
        }
        printf("\n");
    }
    
    printf("\n");

    printf("Gram_ref_real\n");

    for (int i = 0; i < NR; i++){
        for (int j = 0; j < NT; j++){
            printf("  %d    ", Gram_ref_real[i][j][k_debug]);
        }
        printf("\n");
    }
    printf("\n");
    

    printf("Gram_imag\n");
    for (int i = 0; i < NR; i++){
        for (int j = 0; j < NT; j++){
            printf("  %d    ", Gram_imag[i][j][k_debug]);
        }
        printf("\n");
    }
    
    printf("\n");

    printf("Gram_ref_imag\n");

    for (int i = 0; i < NR; i++){
        for (int j = 0; j < NT; j++){
            printf("  %d    ", Gram_ref_imag[i][j][k_debug]);
        }
        printf("\n");
    }
    printf("\n");



#endif

  
  
return 0;
}
