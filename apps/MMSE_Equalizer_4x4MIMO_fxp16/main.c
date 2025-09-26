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

#define RESCALED 1
#define DEBUG 1

int main(void) {
    size_t vl = vsetvl_e16m1(N_SUBCARRIER_ITR);
    size_t offset = 0;

    int16_t Gram_real[NT][NR][N_SUBCARRIER] = {0};
    int16_t Gram_imag[NT][NR][N_SUBCARRIER] = {0};

    int16_t L_real[NT][NR][N_SUBCARRIER] = {0};
    int16_t L_imag[NT][NR][N_SUBCARRIER] = {0};

    int16_t z_real[NT][1][N_SUBCARRIER] = {0};
    int16_t z_imag[NT][1][N_SUBCARRIER] = {0};
    //int16_t u_real[NT][1][N_SUBCARRIER] = {0};
    //int16_t u_imag[NT][1][N_SUBCARRIER] = {0};
    //int16_t x_real[NT][1][N_SUBCARRIER] = {0};
    //int16_t x_imag[NT][1][N_SUBCARRIER] = {0};

    

    //match with python saturation
    asm volatile("csrwi vxrm, 2" ::: "memory");

    start_timer();
    
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

    stop_timer();


    int64_t v_compute_gram_runtime = get_timer();

    
    v_LL_decomp_cfxp16(  
        NT, 
        NR, 
        vl,
        offset,
        N_SUBCARRIER,
        (int16_t *)Gram_real,
        (int16_t *)Gram_imag,
        (int16_t *)L_real,
        (int16_t *)L_imag
    );


    v_compute_z_cfxp16(
        NT, 
        NR, 
        vl,
        offset,
        N_SUBCARRIER,
        (int16_t *)H_real_fxp, 
        (int16_t *)H_imag_fxp,
        (int16_t *)Y_ref_real,
        (int16_t *)Y_ref_imag,
        (int16_t *)z_real,
        (int16_t *)z_imag
    );
        
    

    float performance = (2 * 2 * (4.0 * NT * (NT + 1) / 2 - NT * 2.0) * NT )* vl / v_compute_gram_runtime;
    float utilization = 100.0 * performance / (8.0 * NR_LANES);

    printf("The execution of gram matrix computation took %d cycles.\n", v_compute_gram_runtime);
    printf("The performance is %f FLOP/cycle (%f%% utilization).\n", performance/4.0, utilization);



    
#ifdef DEBUG   
    
    int k_debug = 0;

    printf("===================DEBUG:v_compute_gram()=====================\n");
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

    printf("===================DEBUG:v_LL_decomp()=====================\n");

    printf("    SKIPPED, Cycle Performance Simulation, Functionally Incorrect!\n");
    /*
    printf("L_real\n");
    for (int i = 0; i < NR; i++){
        for (int j = 0; j < NT; j++){
            printf("  %d    ", L_real[i][j][k_debug]);
        }
        printf("\n");
    }
    
    printf("\n");

    printf("L_ref_real\n");

    for (int i = 0; i < NR; i++){
        for (int j = 0; j < NT; j++){
            printf("  %d    ", L_ref_real[i][j][k_debug]);
        }
        printf("\n");
    }
    printf("\n");
    

    printf("L_imag\n");
    for (int i = 0; i < NR; i++){
        for (int j = 0; j < NT; j++){
            printf("  %d    ", L_imag[i][j][k_debug]);
        }
        printf("\n");
    }
    
    printf("\n");

    printf("L_ref_imag\n");

    for (int i = 0; i < NR; i++){
        for (int j = 0; j < NT; j++){
            printf("  %d    ", L_ref_imag[i][j][k_debug]);
        }
        printf("\n");
    }
    printf("\n");
    */
    printf("===================DEBUG:v_compute_z()=====================\n");


    printf("z_real\n");
    for (int i = 0; i < NR; i++){
        for (int j = 0; j < 1; j++){
            printf("  %d    ", z_real[i][0][k_debug]);
        }
        printf("\n");
    }
    
    printf("\n");

    printf("z_ref_real\n");

    for (int i = 0; i < NR; i++){
        for (int j = 0; j < 1; j++){
            printf("  %d    ", Z_ref_real[i][0][k_debug]);
        }
        printf("\n");
    }
    printf("\n");
    

    printf("z_imag\n");
    for (int i = 0; i < NR; i++){
        for (int j = 0; j < 1; j++){
            printf("  %d    ", z_imag[i][0][k_debug]);
        }
        printf("\n");
    }
    
    printf("\n");

    printf("z_ref_imag\n");

    for (int i = 0; i < NR; i++){
        for (int j = 0; j < 1; j++){
            printf("  %d    ", Z_ref_imag[i][0][k_debug]);
        }
        printf("\n");
    }
    printf("\n");

#endif

  
  
return 0;
}
